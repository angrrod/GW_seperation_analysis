### use 
from .Pipeline_Amortized import Pipeline_Amortized
from scenario import GWScenario
import yaml
from pathlib import Path
from utils import get_dingo_dir_yamls, get_dingo_dir_data
import subprocess
import os
from config import DynestyConfig
from threadpoolctl import threadpool_limits
from dingo.gw.training.train_pipeline import (
    prepare_training_new,
    train_stages,
)
import numpy as np
from dingo.gw.domains import build_domain
from dingo.gw.noise.asd_dataset import ASDDataset
import math
import torch
import pyro.distributions as dist
from torch.distributions import constraints
from dingo.gw.dataset.generate_dataset import _generate_dataset_main
# from dingo.core.posterior_models.vector_copula_Model import vector_copula_Model
from dingo.core.posterior_models.vector_copula_Model import CopulaNormalizingFlowModel
from dingo.gw.inference.gw_samplers import GWSampler

def read_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        obj = yaml.safe_load(f)

    if obj is None:
        raise ValueError(f"YAML file is empty: {path}")

    return obj

def write_yaml(path: Path, obj: dict):
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)

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
    def __init__(self, logger, scenario:GWScenario, generateData:bool):
        super().__init__(logger, scenario)
        self.MethodConfig   = DynestyConfig()  #temporary dummy method
        # set up YML's so that scenario is correctly translated to bilby
        # self.pipeline_type = Pipeline_type.DINGO
        self.train_dir = get_dingo_dir_data() / "training_run"
        os.makedirs(self.train_dir, exist_ok=True)
        self.export_configs()
        'Dingo/data/training_run/history.txt'
        with open(self.config_paths["train"], "r") as f:
            self.train_settings = yaml.safe_load(f)

        self.local_settings = self.train_settings.pop("local") #.pop("local")
        
        if generateData:
            self.set_up() 
        pm,wfd     = self.prerp_train()
        self.model = pm
        self.wfd   = wfd

    def run_cmd(self, cmd):
        cmd = [str(c) for c in cmd]
        self.logger.info("$$$ running: " + " ".join(cmd))
        subprocess.run(cmd, check=True)

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

    def _prior_to_yaml_string(self, name, p) -> str:
        cls = p.__class__.__name__

        if cls == "Uniform":
            return (
                f"bilby.core.prior.Uniform("
                f"minimum={float(p.minimum)}, maximum={float(p.maximum)}, "
                f"name='{name}')"
            )

        if cls == "PowerLaw":
            return (
                f"bilby.core.prior.PowerLaw("
                f"alpha={float(p.alpha)}, "
                f"minimum={float(p.minimum)}, maximum={float(p.maximum)}, "
                f"name='{name}')"
            )

        if cls == "Sine":
            return f"bilby.core.prior.Sine(name='{name}')"

        if cls == "Cosine":
            return f"bilby.core.prior.Cosine(name='{name}')"

        if cls == "Constraint":
            return (
                f"bilby.core.prior.Constraint("
                f"minimum={float(p.minimum)}, maximum={float(p.maximum)}, "
                f"name='{name}')"
            )

        if cls == "UniformSourceFrame":
            if name in {"luminosity_distance_A","luminosity_distance_B"}:
                return (
                    f"bilby.gw.prior.UniformSourceFrame("
                    f"minimum={float(p.minimum)}, maximum={float(p.maximum)}, "
                    f"name=luminosity_distance)"
                )
            else:
                return (
                    f"bilby.gw.prior.UniformSourceFrame("
                    f"minimum={float(p.minimum)}, maximum={float(p.maximum)}, "
                    f"name='{name}')"
                )
        raise NotImplementedError(f"Do not know how to export prior {name}: {cls}")

    def _export_prior_yaml_blocks(self,fixed_extrinsic_prior = True) -> tuple[dict, dict]:
        INTRINSIC_BASE_PARAMS = {
            "mass_1",
            "mass_2",
            # "chirp_mass", 
            # "mass_ratio",
            "a_1", 
            "a_2",
            "tilt_1",
            "tilt_2",
            "phi_12", 
            "phi_jl",
        }

        EXTRINSIC_BASE_PARAMS = {
            "luminosity_distance" : 100.0,
            "dec"                 : 0.0,
            "ra"                  : 0.0,
            "theta_jn"            : 0.0,
            "psi"                 : 0.0, 
            "phase"               : 0.0,
            "geocent_time"        : 0.0,
        }
        
        SIGNAL_SUFFIXES = ("_A", "_B")
        
        INTRINSIC_PARAMS = {
            f"{name}{suffix}"
            for name in INTRINSIC_BASE_PARAMS
            for suffix in SIGNAL_SUFFIXES
        }

        EXTRINSIC_PARAMS = {
            f"{name}{suffix}": default
            for name, default in EXTRINSIC_BASE_PARAMS.items()
            for suffix in SIGNAL_SUFFIXES
        }
        
        EXTRINSIC_PARAMS["delta_t_AB"] = 0.0
        prior_dict = self.scenario.prior.getJointPriors()

        intrinsic_prior = {}
        extrinsic_prior = {}

        # DINGO waveform datasets usually want component masses,
        # TODO:: not chirp_mass/mass_ratio.
        suffixes = ["_A","_B"]
        skip = set()
        for suffix in suffixes:
            intrinsic_prior[f"mass_1{suffix}"] = (
                "bilby.core.prior.Uniform("
                "minimum=5.0, maximum=40.0, name='mass_1')"
            )
            intrinsic_prior[f"mass_2{suffix}"] = (
                "bilby.core.prior.Uniform("
                "minimum=4.0, maximum=30.0, name='mass_2')"
            )

            skip.update([f"chirp_mass{suffix}", f"mass_ratio{suffix}", f"mass_1{suffix}", f"mass_2{suffix}"])

        for name, p in prior_dict.items():
            if name in skip:
                continue

            prior_str = self._prior_to_yaml_string(name, p)

            if name in INTRINSIC_PARAMS:
                intrinsic_prior[name] = prior_str
            elif name in EXTRINSIC_PARAMS:
                if fixed_extrinsic_prior:
                    extrinsic_prior[name] = prior_str
                else:
                    extrinsic_prior[name] = EXTRINSIC_PARAMS.get(name)
            else:
                raise ValueError(
                    f"Prior parameter {name!r} is neither intrinsic nor extrinsic. "
                    "Classify it explicitly."
                )

        return intrinsic_prior, extrinsic_prior

    def _patch_waveform_yaml(self):
        cfg = self.scenario.config

        yml = read_yaml(get_dingo_dir_yamls() / "waveform_dataset_settings.yml")

        yml["domain"]["f_min"] = float(cfg.minimum_frequency)
        yml["domain"]["f_max"] = float(cfg.sampling_frequency / 2)
        yml["domain"]["delta_f"] = float(1.0 / cfg.duration)

        yml["waveform_generator"]["approximant"] = cfg.waveform_approximant
        yml["waveform_generator"]["f_ref"] = float(cfg.reference_frequency)
        
        intrinsic_prior, extrinsic_prior = self._export_prior_yaml_blocks(False)

        yml["intrinsic_prior"] = intrinsic_prior | extrinsic_prior
        yml["extrinsic_prior"] = extrinsic_prior
        
        dst = get_dingo_dir_yamls() / "waveform_dataset_settings.yml"
        write_yaml(dst, yml)
        return dst

    def _patch_asd_yaml(self):
        cfg = self.scenario.config

        yml = read_yaml(get_dingo_dir_yamls() / "asd_dataset_settings.yml")
        
        yml["dataset_settings"]["f_s"] = int(cfg.sampling_frequency)
        yml["dataset_settings"]["T"] = float(cfg.duration)
        yml["dataset_settings"]["time_psd"] = max(4 * float(cfg.duration), 1024)
        ce_psd = self._resolve_input_file(cfg.ASD_file_name_CE + "_PSD.txt")
        yml["dataset_settings"]["detectors"] = ["H1", "L1"]
        yml["asds"] = {
            "H1": str(ce_psd),
            "L1": str(ce_psd),
        }

        dst = get_dingo_dir_yamls() / "asd_dataset_settings.yml"
        write_yaml(dst, yml)
        return dst

    def _patch_training_yaml(self):
        cfg = self.scenario.config
        
        training_name = "training_copula.yml"# "training.yml"
        
        train_yml = read_yaml(get_dingo_dir_yamls() / training_name)
        waveform_yml = read_yaml(get_dingo_dir_yamls() / "waveform_dataset_settings.yml")
        
        train_yml.setdefault("data", {})
        train_yml["data"].setdefault("ref_time", 0.0)
        # Required by prepare_training_new()
        train_yml["data"]["waveform_dataset_path"] = str(
            get_dingo_dir_data() / "waveform_dataset.hdf5"
        )

        # Required by build_svd_for_embedding_network()
        _, extrinsic_prior = self._export_prior_yaml_blocks()
        train_yml["data"]["extrinsic_prior"] = extrinsic_prior

        # Usually useful / expected in DINGO train settings
        train_yml["data"].setdefault("train_fraction", 0.95)

        # Keep training domain consistent with waveform dataset.
        train_yml["data"]["domain_update"] = {
            "f_min": float(cfg.minimum_frequency),
            "f_max": float(cfg.sampling_frequency / 2),
        }

        # Keep detector list consistent with your ASD setup.
        train_yml["data"]["detectors"] = ["H1", "L1"]

        # Required later by prepare_training_new()
        train_yml.setdefault("training", {})
        train_yml["training"].setdefault("stage_0", {})
        train_yml["training"]["stage_0"]["asd_dataset_path"] = str(
            get_dingo_dir_data() / "asd_dataset.hdf5"
        )

        # Robust local defaults.
        train_yml.setdefault("local", {})
        train_yml["local"].setdefault("device", "cpu")
        train_yml["local"].setdefault("num_workers", 1)
        train_yml["local"].setdefault("leave_waveforms_on_disk", True)

        dst = get_dingo_dir_yamls() / training_name
        write_yaml(dst, train_yml)
        return dst

    def _constrain_sample(self,samples,pos_columns):
        mask = (samples[pos_columns] > 0).all(axis=1)
        print(f"Kept {mask.sum()} / {len(mask)} samples; removed {(~mask).sum()} invalid samples")

        samples = samples.loc[mask].copy()
        return samples

    def export_configs(self):
        #combines the bilby backend to make it compatible with DINGO, reads the existing config and writes to yml files used in DINGO, to make them consistent
        self.config_paths = {
            "waveform" : self._patch_waveform_yaml(),
            "asd"      : self._patch_asd_yaml(),
            "train"    : self._patch_training_yaml()
        }
        
        self.artifact_paths = {
            "waveform_dataset": get_dingo_dir_data() / "waveform_dataset.hdf5",
            "asd_dataset": get_dingo_dir_data() / "asd_dataset.hdf5",
        }
        
        return self.config_paths

    def generate_waveform_dataset(self):
        if not self.config_paths:
            self.export_configs()

        settings_file = str(self.config_paths["waveform"])
        out_file = str(self.artifact_paths["waveform_dataset"])
        num_processes = int(self.MethodConfig.cores)
        num_signals = 2

        self.logger.info(
            "$$$ generating DINGO waveform dataset directly: "
            f"settings={settings_file}, out_file={out_file}, "
            f"num_processes={num_processes}, num_signals={num_signals}"
        )

        _generate_dataset_main(
            settings_file=settings_file,
            out_file=out_file,
            num_processes=num_processes,
            num_signals=num_signals,
        )

    def generate_asd_dataset(self):
        """
        Create a DINGO-compatible ASDDataset HDF5 from a fixed custom PSD/ASD file.

        This does not download strain and does not estimate PSDs from an observing run.
        It simply packages the custom design curve into the ASDDataset format expected
        by DINGO training.
        """

        cfg = self.scenario.config

        out_file = self.artifact_paths["asd_dataset"]
        out_file.parent.mkdir(parents=True, exist_ok=True)

        psd_file = self._resolve_input_file(cfg.ASD_file_name_CE + "_PSD.txt")

        raw = np.loadtxt(psd_file)

        # Common format: two columns [frequency, PSD or ASD].
        if raw.ndim == 2 and raw.shape[1] >= 2:
            f_raw = raw[:, 0]
            y_raw = raw[:, 1]
        else:
            raise ValueError(
                f"Expected {psd_file} to have at least two columns: frequency and PSD/ASD."
            )

        f_min = float(cfg.minimum_frequency)
        f_max = float(cfg.sampling_frequency / 2)
        delta_f = float(1.0 / cfg.duration)

        domain_dict = {
            "type": "UniformFrequencyDomain",
            "f_min": f_min,
            "f_max": f_max,
            "delta_f": delta_f,
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

        dataset_dict = {
            "settings": {
                "dataset_settings": {
                    "detectors": ["H1", "L1"],
                    "f_s": int(cfg.sampling_frequency),
                    "T": float(cfg.duration),
                    "time_psd": max(4 * float(cfg.duration), 1024),
                    "source": "custom_fixed_design_psd",
                    "psd_file": str(psd_file),
                },
                "domain_dict": domain_dict,
            },
            "asds": {
                "H1": asd_array,
                "L1": asd_array.copy(),
            },
            "gps_times": {
                "H1": np.array([0.0]),
                "L1": np.array([0.0]),
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

    def prerp_train(self):

        pm, wfd = prepare_training_new(
            train_settings=self.train_settings,
            train_dir=str(self.train_dir),
            local_settings=self.local_settings,
        )

        self.model = pm
        self.waveform_dataset = wfd
        return pm,wfd

    def train(self,start_from_checkpoint = False):
        path = self.train_dir / 'history.txt'
        if path.exists():
            path.unlink()
        
        if start_from_checkpoint:
            self.load_model()

        with threadpool_limits(limits=1, user_api="blas"):
            complete = train_stages(
                pm=self.model,
                wfd=self.waveform_dataset,
                train_dir=str(self.train_dir),
                local_settings=self.local_settings,
            )
        return complete

    def build_context_from_initialized_scenario(
        self,
    ):
        """
        Build DINGO sampler context from an already initialized GWScenario.

        Assumes scenario.setUpScenario() has already been called in main.
        """

        ifos_by_name = {ifo.name: ifo for ifo in self.scenario.ifos}
        waveform = {}
        asds = {}
        
        asd_settings = read_yaml(self.config_paths["asd"])
        detectors = asd_settings["dataset_settings"]["detectors"]

        for name in detectors:
            ifo = ifos_by_name[name]
            waveform[name] = ifo.strain_data.frequency_domain_strain
            freqs = ifo.strain_data.frequency_array
            psd = ifo.power_spectral_density.get_power_spectral_density_array(freqs)
            asds[name] = np.sqrt(psd)

        return {
            "waveform": waveform,
            "asds": asds,
            "parameters" : self.scenario.injct_params_waves[0]
        }

    def load_model(self):
        latest_model_path =  self.train_dir / "model_latest.pt"
        device            = self.local_settings.get("device", "cpu")
        
        self.model = CopulaNormalizingFlowModel(
            device             = device,
            model_filename     = str(latest_model_path),
            load_training_info = False,
        )
        self.logger.info(f"$$$ loaded DINGO vector_copula_Model: {latest_model_path}")
        return self.model

    def infer(self,num_samples:int):
        current_model   = self.load_model()
        sampler         = GWSampler(model = current_model)
        sampler.context = self.build_context_from_initialized_scenario()
        sampler.run_sampler(num_samples=num_samples, batch_size=1_000, isJoint = True)
        samples = sampler.samples

        #post-process
        if {"geocent_time_A", "delta_t_AB"}.issubset(samples.columns):
            samples["geocent_time_B"] = samples["geocent_time_A"] + samples["delta_t_AB"]
        #tests: Decide what to do with these
        pos_columns = ["mass_1_A","mass_1_B","mass_2_A","mass_2_B","delta_t_AB"]
        samples = self._constrain_sample(samples,pos_columns)
        samples = add_derived_params_to_df(samples)
        
        if samples.isna().any().any():
            raise ValueError("NaNs found in DINGO posterior samples.")

        return samples

    def run(self):
        """
        High level function to be used for the general test pipeline
        """
        raise NotImplementedError

