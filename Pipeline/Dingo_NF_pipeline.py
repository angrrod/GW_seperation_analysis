### use 
from .Pipeline_Amortized import Pipeline_Amortized
from scenario import GWScenario
import yaml
from pathlib import Path
import subprocess
import os
import hashlib
from unittest.mock import patch
import dingo.gw.training.train_pipeline as dingo_train_pipeline
from dingo.gw.training.train_pipeline import (
    prepare_training_resume,
    train_stages,
)
import numpy as np
from dingo.gw.domains import build_domain
from dingo.gw.noise.asd_dataset import ASDDataset
import torch
from dingo.gw.dataset.generate_dataset import _generate_dataset_main
# from dingo.core.posterior_models.vector_copula_Model import vector_copula_Model
from dingo.core.posterior_models.vector_copula_Model import CopulaNormalizingFlowModel
from dingo.gw.inference.gw_samplers import GWSampler

from utils.paths import get_config_dir, get_dingo_dir_data
from utils.utils import load_config
from copy import deepcopy
from numbers import Number
import wandb


WATCHED_TIME_PARAMETERS = {
    "delta_t_AB",
    "geocent_time_A",
    "geocent_time_B",
    
    # Intrinsic parameters we care about
    "chirp_mass_A",
    "mass_ratio_A",
    "mass_1_A",
    "mass_2_A",

    "chirp_mass_B",
    "mass_ratio_B",
    "mass_1_B",
    "mass_2_B",
}


def _collect_watched_scalars(
    obj,
    path="sample",
):
    """
    Recursively find selected scalar values in a nested
    Dingo sample dictionary.
    """
    found = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            child_path = f"{path}.{key}"

            found.extend(
                _collect_watched_scalars(
                    value,
                    child_path,
                )
            )

    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            found.extend(
                _collect_watched_scalars(
                    value,
                    f"{path}[{index}]",
                )
            )

    else:
        leaf_name = path.rsplit(".", 1)[-1]

        if leaf_name not in WATCHED_TIME_PARAMETERS:
            return found

        if torch.is_tensor(obj):
            if obj.numel() == 1:
                found.append(
                    (
                        path,
                        float(obj.detach().cpu()),
                    )
                )

        elif isinstance(obj, np.ndarray):
            if obj.size == 1:
                found.append(
                    (
                        path,
                        float(obj.reshape(-1)[0]),
                    )
                )

        elif isinstance(obj, Number):
            found.append(
                (
                    path,
                    float(obj),
                )
            )

    return found

def masses_to_chirp_mass_and_q(mass_1, mass_2):
    m1 = np.maximum(mass_1, mass_2)
    m2 = np.minimum(mass_1, mass_2)

    chirp_mass = (m1 * m2) ** (3.0 / 5.0) / (m1 + m2) ** (1.0 / 5.0)
    mass_ratio = m2 / m1

    return chirp_mass, mass_ratio

def columns_are_positive(columns:list[str],samples):
    for column in columns:
        if column in samples.columns:
            bad = samples[column] <= 0
            if bad.any():
                raise ValueError(
                    f"{column} must be strictly positive, but found "
                    f"{bad.sum()} / {len(samples)} non-positive samples. "
                    f"min={samples['delta_t_AB'].min()}"
                )

def add_derived_params_to_df(samples):
    for suffix in ["_A", "_B"]:
        m1_key = f"mass_1{suffix}"
        m2_key = f"mass_2{suffix}"

        if {m1_key, m2_key}.issubset(samples.columns):
            if (samples[[m1_key, m2_key]] <= 0).any().any():
                print(f"Warning: non-positive masses found for {suffix}; chirp/q may be invalid.")

            chirp, q = masses_to_chirp_mass_and_q(
                samples[m1_key].to_numpy(),
                samples[m2_key].to_numpy(),
            )

            samples[f"chirp_mass{suffix}"] = chirp
            samples[f"mass_ratio{suffix}"] = q
    return samples

class DINGO_pipeline(Pipeline_Amortized):
    def __init__(self, logger, scenario:GWScenario, generateData:bool, nbr_cores:int = 1):
        super().__init__(logger, scenario)
        self.nbr_cores    = nbr_cores
        # set up YML's so that scenario is correctly translated to bilby
        # self.pipeline_type = Pipeline_type.DINGO
        self.train_dir = get_dingo_dir_data() / "training_run"
        os.makedirs(self.train_dir, exist_ok=True)
        self.waveform_settings = load_config("waveform_dataset_settings.yml")
        self.asd_settings = load_config("asd_dataset_settings.yml")
        self.train_settings = load_config("training_copula.yml")
        
        # Dingo treats `local` separately from the training settings. Work on a
        # private copy so that configuration dictionaries are never mutated.
        self.train_settings = deepcopy(self.train_settings)
        self.local_settings = self.train_settings.pop("local", {})

        # Dingo's waveform-dataset generator requires the original YAML path.
        # Resolve it through the same config utility used by load_config().
        self.waveform_settings_path = get_config_dir() / "waveform_dataset_settings.yml"
        

        # Dataset outputs are configuration-driven. If a path is absent, retain
        # the old Dingo/data fallback.
        waveform_dataset_path = (
            self.train_settings.get("data", {}).get("waveform_dataset_path")
        )
        asd_dataset_path = self._get_asd_dataset_path_from_training()
        self.artifact_paths = {
            "waveform_dataset": self._resolve_artifact_path(
                waveform_dataset_path, "waveform_dataset.hdf5"
            ),
            "asd_dataset": self._resolve_artifact_path(
                asd_dataset_path, "asd_dataset.hdf5"
            ),
        }
        
        local_cache = os.environ.get("GW_LOCAL_CACHE")
        if local_cache is not None:
            self.local_settings["local_cache_path"] = local_cache

        print(
            "Dingo local cache:",
            self.local_settings.get("local_cache_path")
        )
        if (
            self.local_settings.get("device", "cpu") == "cuda"
            and not torch.cuda.is_available()
        ):
            print(
                "CUDA requested but unavailable. "
                "Falling back to CPU."
            )
            self.local_settings["device"] = "cpu"
            
        if generateData:
            self.set_up() 
            
        self.model = None
        self.waveform_dataset = None

    def run_cmd(self, cmd):
        cmd = [str(c) for c in cmd]
        self.logger.info("$$$ running: " + " ".join(cmd))
        subprocess.run(cmd, check=True)

    def _stage_debug_hook(
        self,
        wfd,
        stage,
    ):
        """
        Run one sample manually through every Dingo transform and
        log where the event-time variables occur.

        This executes in the main process before DataLoader workers
        are created.
        """
        transform_pipeline = wfd.transform

        transforms = getattr(
            transform_pipeline,
            "transforms",
            None,
        )

        if transforms is None:
            print(
                "Could not inspect transform pipeline: "
                f"{type(transform_pipeline)} has no "
                "'transforms' attribute."
            )
            return

        print("\n=== DINGO TRANSFORM CHAIN ===")

        transform_rows = []

        for index, transform in enumerate(transforms):
            transform_name = type(transform).__name__

            print(
                f"{index:02d}: {transform_name}"
            )

            transform_rows.append(
                [
                    index,
                    transform_name,
                ]
            )

        # Obtain a raw WaveformDataset item without applying the
        # transform pipeline automatically.
        original_transform = wfd.transform

        try:
            wfd.transform = None
            sample = deepcopy(wfd[0])
        finally:
            wfd.transform = original_transform

        parameter_rows = []

        # Apply each transform manually and inspect the result.
        for index, transform in enumerate(transforms):
            transform_name = type(transform).__name__

            sample = transform(sample)

            watched_values = _collect_watched_scalars(
                sample
            )

            for parameter_path, value in watched_values:
                print(
                    f"{index:02d} "
                    f"{transform_name:40s} "
                    f"{parameter_path} = {value}"
                )

                parameter_rows.append(
                    [
                        index,
                        transform_name,
                        parameter_path,
                        value,
                    ]
                )

        if wandb.run is not None:
            wandb.log(
                {
                    "epoch": self.model.epoch,
                    "debug/transform_chain": wandb.Table(
                        columns=[
                            "index",
                            "transform",
                        ],
                        data=transform_rows,
                    ),
                    "debug/time_parameter_path": (
                        wandb.Table(
                            columns=[
                                "index",
                                "transform",
                                "parameter_path",
                                "value",
                            ],
                            data=parameter_rows,
                        )
                    ),
                }
            )

    def _epoch_debug_hook(
        self,
        model,
        epoch,
        train_loader,
        test_loader,
        train_loss,
        test_loss,
        learning_rates,
    ):
        metrics = {}

        # ---------------------------------
        # Basic epoch information
        # ---------------------------------

        metrics["debug/train_loss"] = float(
            train_loss
        )

        metrics["debug/test_loss"] = float(
            test_loss
        )

        metrics["debug/loss_gap"] = float(
            test_loss - train_loss
        )

        # ---------------------------------
        # Optimizer parameter groups
        # ---------------------------------

        for index, group in enumerate(
            model.optimizer.param_groups
        ):
            group_name = group.get(
                "group_name",
                f"group_{index}",
            )

            metrics[
                f"debug/lr/{group_name}"
            ] = float(group["lr"])

        # ---------------------------------
        # Parameter norms
        # ---------------------------------

        total_norm_sq = 0.0

        for name, parameter in (
            model.network.named_parameters()
        ):
            if not parameter.requires_grad:
                continue

            value = parameter.detach()

            norm = torch.linalg.vector_norm(
                value
            )

            total_norm_sq += float(
                norm.cpu()
            ) ** 2

        metrics["debug/trainable_parameter_norm"] = (
            total_norm_sq ** 0.5
        )

        # ---------------------------------
        # Reduced-basis layer specifically
        # ---------------------------------

        for name, parameter in (
            model.network.named_parameters()
        ):
            if "layers_rb" in name:
                metrics[
                    f"debug/rb_norm/{name}"
                ] = float(
                    torch.linalg.vector_norm(
                        parameter.detach()
                    ).cpu()
                )

        return metrics

    def _resolve_input_file(self, filename: str) -> Path:
        base = os.environ.get("GW_INP_DIR")
        if base is not None:
            candidate = Path(base) / filename
            if candidate.exists():
                return candidate.resolve()

        candidate = Path(filename)
        if candidate.exists():
            return candidate.resolve()

        raise FileNotFoundError(f"Could not resolve input file: {filename}")

    def _resolve_artifact_path(self, configured_path, fallback_name: str) -> Path:
        """Resolve an output artifact path without depending on the working directory."""
        if configured_path:
            path = Path(configured_path)
            if path.is_absolute():
                path.parent.mkdir(parents=True, exist_ok=True)
                return path

            # Dingo configs historically use paths such as Dingo/data/foo.hdf5.
            # Keep only the configured filename and anchor it to the project's
            # canonical Dingo data directory.
            path = get_dingo_dir_data() / path.name
        else:
            path = get_dingo_dir_data() / fallback_name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _get_asd_dataset_path_from_training(self):
        """Return the ASD dataset path declared by the first training stage."""
        for stage in self.train_settings.get("training", {}).values():
            path = stage.get("asd_dataset_path")
            if path:
                return path
        return None

    def _constrain_sample(self,samples,pos_columns):
        mask = (samples[pos_columns] > 0).all(axis=1)
        print(f"Kept {mask.sum()} / {len(mask)} samples; removed {(~mask).sum()} invalid samples")

        samples = samples.loc[mask].copy()
        return samples

    def _get_svd_cache_signature(self) -> str:
        """
        Build a hash describing the configuration on which the
        embedding-network SVD basis depends.

        If these settings change, the cached basis is not reused.
        """

        relevant_config = {
            # Increment this manually if our caching implementation changes.
            "cache_version": 1,

            # Waveform family, frequency grid, priors, etc.
            "waveform": self.waveform_settings,

            # ASD / detector configuration.
            "asd": self.asd_settings,

            # Dingo data settings used when constructing the SVD.
            "data": self.train_settings["data"],

            # Requested SVD dimensions/settings.
            "svd": (
                self.train_settings
                .get("model", {})
                .get("embedding_kwargs", {})
                .get("svd", {})
            ),
        }

        serialized = yaml.safe_dump(
            relevant_config,
            sort_keys=True,
        )

        return hashlib.sha256(
            serialized.encode("utf-8")
        ).hexdigest()

    def _prepare_training_new_with_svd_cache(self):
        """
        Run Dingo's standard prepare_training_new(), but cache the
        final V_rb_list returned by build_svd_for_embedding_network().

        Dingo itself is not modified.

        First compatible run:
            Dingo builds SVD normally -> cache V_rb_list.

        Later compatible run:
            load cached V_rb_list -> skip expensive SVD construction.
        """

        cache_path = (
            self.train_dir
            / "embedding_svd_V_rb_cache.pt"
        )

        expected_signature = (
            self._get_svd_cache_signature()
        )

        # This is Dingo's ORIGINAL implementation.
        original_svd_builder = (
            dingo_train_pipeline
            .build_svd_for_embedding_network
        )

        def cached_svd_builder(*args, **kwargs):
            """
            Drop-in replacement used only while
            prepare_training_new() is executing.
            """

            # ---------------------------------------------
            # Try cached basis
            # ---------------------------------------------

            if cache_path.exists():
                try:
                    cache = torch.load(
                        cache_path,
                        map_location="cpu",
                        weights_only=False,
                    )

                    cached_signature = cache.get(
                        "signature"
                    )

                    if (
                        cached_signature
                        == expected_signature
                    ):
                        self.logger.info(
                            "$$$ loading cached DINGO "
                            f"embedding SVD from {cache_path}"
                        )

                        V_rb_list = cache[
                            "V_rb_list"
                        ]

                        for i, V in enumerate(
                            V_rb_list
                        ):
                            self.logger.info(
                                "$$$ cached V_rb "
                                f"[{i}] shape = "
                                f"{getattr(V, 'shape', None)}"
                            )

                        return V_rb_list

                    self.logger.info(
                        "$$$ existing SVD cache is "
                        "incompatible with current "
                        "configuration; rebuilding."
                    )

                except Exception as exc:
                    self.logger.warning(
                        "$$$ could not load existing "
                        "SVD cache; rebuilding. "
                        f"Reason: {exc}"
                    )

            # ---------------------------------------------
            # No valid cache -> use Dingo unchanged
            # ---------------------------------------------

            self.logger.info(
                "$$$ building embedding SVD using "
                "standard DINGO implementation"
            )

            V_rb_list = original_svd_builder(
                *args,
                **kwargs,
            )

            # ---------------------------------------------
            # Store the EXACT result Dingo returns
            # ---------------------------------------------

            torch.save(
                {
                    "signature": expected_signature,
                    "V_rb_list": V_rb_list,
                },
                cache_path,
            )

            self.logger.info(
                "$$$ saved DINGO embedding SVD "
                f"cache to {cache_path}"
            )

            return V_rb_list

        # -------------------------------------------------
        # Temporarily replace only the function used inside
        # prepare_training_new().
        #
        # As soon as this `with` block exits, Dingo's
        # original function is restored automatically.
        # -------------------------------------------------

        with patch.object(
            dingo_train_pipeline,
            "build_svd_for_embedding_network",
            cached_svd_builder,
        ):
            pm, wfd = (
                dingo_train_pipeline
                .prepare_training_new(
                    train_settings=self.train_settings,
                    train_dir=str(self.train_dir),
                    local_settings=self.local_settings,
                )
            )

        return pm, wfd

    def generate_waveform_dataset(self):

        settings_file = str(self.waveform_settings_path)
        out_file = str(self.artifact_paths["waveform_dataset"])
        num_signals = 2

        self.logger.info(
            "$$$ generating DINGO waveform dataset directly: "
            f"settings={settings_file}, out_file={out_file}, "
            f"num_processes={self.nbr_cores}, num_signals={num_signals}"
        )

        _generate_dataset_main(
            settings_file=settings_file,
            out_file=out_file,
            num_processes=self.nbr_cores,
            num_signals=num_signals,
        )

    def generate_asd_dataset(self):
        """
        Create a DINGO-compatible ASDDataset HDF5 from a fixed custom PSD/ASD file.

        This does not download strain and does not estimate PSDs from an observing run.
        It simply packages the custom design curve into the ASDDataset format expected
        by DINGO training.
        """
        out_file = self.artifact_paths["asd_dataset"]
        out_file.parent.mkdir(parents=True, exist_ok=True)
        
        dataset_settings = deepcopy(self.asd_settings["dataset_settings"])
        detectors = list(dataset_settings["detectors"])
        if not detectors:
            raise ValueError("ASD config must define at least one detector.")

        asd_paths = self.asd_settings.get("asds", {})
        reference_detector = detectors[0]
        if reference_detector not in asd_paths:
            raise KeyError(
                f"No ASD/PSD file configured for detector {reference_detector!r}."
            )

        psd_file = self._resolve_input_file(asd_paths[reference_detector])
        raw = np.loadtxt(psd_file)

        # Common format: two columns [frequency, PSD or ASD].
        if raw.ndim == 2 and raw.shape[1] >= 2:
            f_raw = raw[:, 0]
            y_raw = raw[:, 1]
        else:
            raise ValueError(
                f"Expected {psd_file} to have at least two columns: frequency and PSD/ASD."
            )

        waveform_domain = self.waveform_settings["domain"]
        domain_dict = {
            "type": "UniformFrequencyDomain",
            "f_min": float(waveform_domain["f_min"]),
            "f_max": float(waveform_domain["f_max"]),
            "delta_f": float(waveform_domain["delta_f"]),
        }
        domain = build_domain(domain_dict)
        f_target = domain()

        # Interpolate onto DINGO's frequency grid.
        y_interp = np.interp(f_target, f_raw, y_raw)

        # IMPORTANT:
        # If your file is PSD, convert PSD -> ASD.
        # If your file is already ASD, remove the sqrt.
        asd = np.sqrt(y_interp)

        if not np.all(np.isfinite(asd)):
            raise ValueError("Custom ASD contains non-finite values after interpolation.")

        if np.any(asd <= 0):
            raise ValueError("Custom ASD must be strictly positive.")

        # DINGO expects shape (num_asd_samples, num_frequency_bins).
        asd_array = np.array([asd], dtype=np.float32)
        
        dataset_settings["source"] = "custom_fixed_design_psd"
        dataset_settings["psd_file"] = str(psd_file)

        dataset_dict = {
            "settings": {
                "dataset_settings": dataset_settings,
                "domain_dict": domain_dict,
            },
            "asds": {
                name: asd_array.copy()
                for name in detectors
            },
            "gps_times": {
                name: np.array([0.0])
                for name in detectors
            },
        }

        dataset = ASDDataset(dictionary=dataset_dict)
        dataset.to_file(str(out_file))

        self.logger.info(f"$$$ wrote custom DINGO ASD dataset: {out_file}")

    def set_up(self):        
        """
        generate the data
        """
        # if not self.artifact_paths["waveform_dataset"].exists():
        self.generate_waveform_dataset()
            
        # if not self.artifact_paths["asd_dataset"].exists():
        self.generate_asd_dataset()

    def prepare_training(self):
        """
        Prepare a new Dingo training run.

        The expensive embedding SVD is cached between compatible
        training runs.
        """

        if (
            self.model is not None
            and self.waveform_dataset is not None
        ):
            return (
                self.model,
                self.waveform_dataset,
            )

        pm, wfd = (
            self._prepare_training_new_with_svd_cache()
        )

        self.model = pm
        self.waveform_dataset = wfd
        self.wfd = wfd

        return pm, wfd

    def train(self,start_from_checkpoint = False, checkpoint_name="model_latest.pt",):
        path = self.train_dir / 'history.txt'
        if not start_from_checkpoint:
            path = self.train_dir / "history.txt"
            if path.exists():
                path.unlink()
        
        if start_from_checkpoint:

            checkpoint_path = (
                self.train_dir / checkpoint_name
            )

            self.model, self.waveform_dataset = (
                prepare_training_resume(
                    checkpoint_name=str(checkpoint_path),
                    local_settings=self.local_settings,
                    train_dir=str(self.train_dir),
                )
            )


        else:
            # THIS is where prepare_training() gets called.
            self.prepare_training()
            
        # self.model.stage_debug_hook = (
        #     self._stage_debug_hook
        # )
        # self.model.epoch_debug_hook = (
        #     self._epoch_debug_hook
        # )


        complete = train_stages(
            pm=self.model,
            wfd=self.waveform_dataset,
            train_dir=str(self.train_dir),
            local_settings=self.local_settings,
        )
        return complete

    def build_context_from_initialized_scenario(
        self,
        parameters = None
    ):
        """
        Build DINGO sampler context from an already initialized GWScenario.
        Assumes scenario.setUpScenario() has already been called in main.
        
        parameters : 
        """
        if parameters is None:
            parameters = self.scenario.injct_params_waves

        ifos_by_name = {ifo.name: ifo for ifo in self.scenario.ifos}
        waveform = {}
        asds = {}
        
        detectors = self.asd_settings["dataset_settings"]["detectors"]

        for name in detectors:
            ifo = ifos_by_name[name]
            waveform[name] = ifo.strain_data.frequency_domain_strain
            freqs = ifo.strain_data.frequency_array
            psd = ifo.power_spectral_density.get_power_spectral_density_array(freqs)
            asds[name] = np.sqrt(psd)

        return {
            "waveform": waveform,
            "asds": asds,
            "parameters" : parameters
        }

    def load_model(self,model_name:str = "model_latest.pt"):
        latest_model_path = self.train_dir / model_name
        device            = self.local_settings.get("device", "cpu")
        
        self.model = CopulaNormalizingFlowModel(
            device             = device,
            model_filename     = str(latest_model_path),
            load_training_info = False,
        )
        self.logger.info(f"$$$ loaded DINGO vector_copula_Model: {latest_model_path}")
        return self.model

    def infer_from_strain(self,num_samples:int,model_name:str = "model_latest.pt"):
        return self.infer(num_samples,self.build_context_from_initialized_scenario(),model_name)

    def infer(self,num_samples:int, injection,model_name:str = "model_latest.pt"):
        """_summary_

        Args:
            num_samples (int): amount of samples
            injection (): dict of dicts corresponding different overlapping signals

        Raises:
            ValueError: if any value is not a valid input

        Returns:
            pd.Dataframe: posterior distribution 
        """
        current_model   = self.load_model(model_name)
        sampler         = GWSampler(model = current_model)
        sampler.context = injection
        sampler.run_sampler(num_samples=num_samples, batch_size=1_000, isJoint = True)
        samples         = sampler.samples

        #post-process
        if {"geocent_time_A", "delta_t_AB"}.issubset(samples.columns):
            samples["geocent_time_B"] = samples["geocent_time_A"] - samples["delta_t_AB"]
        #tests: Decide what to do with these
        pos_columns = ["mass_1_A","mass_1_B","mass_2_A","mass_2_B","delta_t_AB"]
        present_pos_columns = [  #filter those who are needed
            column for column in pos_columns
            if column in samples.columns
        ]
        samples = self._constrain_sample(samples,present_pos_columns)
        
        if samples.isna().any().any():
            raise ValueError("NaNs found in DINGO posterior samples.")

        return samples

    def run(self):
        """
        High level function to be used for the general test pipeline
        """
        raise NotImplementedError
