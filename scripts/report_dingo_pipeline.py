"""Generate an A4 computational map from the checked-out Dingo configurations.

This is a reporting tool, separate from training. It reads configs/source/HDF5
metadata, constructs CPU models with a short frequency axis to count parameters,
and optionally streams a profiler JSON. It never rewrites the training code or
configs, launches training, loads a full waveform bank, or requires a GPU.

Run from the project root with the project Python environment:
    python scripts/report_dingo_pipeline.py

Outputs: A4 PDF/SVG/PNG, a detailed Markdown report, JSON, and CSV tables in
Output/HPC_diagnostics/pipeline_report. Config arithmetic, synthetic CPU probes,
local HDF5 metadata, and historical CUDA measurements are explicitly separated.
"""

import argparse
import collections
import copy
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def resolve_report_paths(config_dir=None, data_dir=None):
    """Separate the source checkout from DATA configs and SCRATCH datasets.

    Local defaults stay beside the source checkout. On VSC, GW_INP_DIR and
    GW_DATA_DIR take precedence over VSC_DATA/GW_separation and
    VSC_SCRATCH/GW_separation/data. CLI paths can override config/data locations.
    """
    work_dir = Path(os.environ.get(
        'GW_INP_DIR',
        str(Path(os.environ['VSC_DATA']) / 'GW_separation')
        if 'VSC_DATA' in os.environ else str(ROOT),
    )).expanduser().resolve()
    config_dir = Path(config_dir).expanduser().resolve() if config_dir is not None else work_dir / 'configs'
    if data_dir is None:
        data_dir = os.environ.get(
            'GW_DATA_DIR',
            str(Path(os.environ['VSC_SCRATCH']) / 'GW_separation/data')
            if 'VSC_SCRATCH' in os.environ else str(work_dir / 'Output/Dingo/data'),
        )
    return dict(source_root=str(ROOT), work_dir=str(work_dir),
                config_dir=str(config_dir), data_dir=str(Path(data_dir).expanduser().resolve()))


def tensor_row(name, shape, dtype, location, note="", copies=1):
    sizes = {"float32": 4, "float64": 8, "complex64": 8, "complex128": 16}
    return dict(name=name, shape=list(shape), copies=copies, dtype=dtype,
                location=location, bytes=math.prod(shape) * sizes[dtype] * copies,
                note=note)


def mib(size):
    return f"{size / 2**20:,.3f} MiB"


def inspect_hdf5(path):
    """Read shapes/dtypes, not waveform values; file may be a historical artifact."""
    import h5py
    if not path.exists():
        return {"path": str(path), "exists": False}
    result = {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path), "exists": True,
              "file_bytes": path.stat().st_size, "datasets": []}
    with h5py.File(path) as f:
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.shape:
                result["datasets"].append(dict(name=name, shape=list(obj.shape),
                                                dtype=str(obj.dtype), bytes=obj.size * obj.dtype.itemsize))
        f.visititems(visitor)
    return result


def config_arithmetic(waveform, cfg):
    domain = {**waveform["domain"], **cfg["data"].get("domain_update", {})}
    n = waveform["num_samples"]
    batch = cfg["training"]["stage_0"]["batch_size"]
    f = round(domain["f_max"] / domain["delta_f"]) + 1
    first = round(domain["f_min"] / domain["delta_f"])
    length = f - first
    k = cfg["data"].get("svd_size_update", waveform["compression"]["svd"]["size"])
    r = cfg["model"]["embedding_kwargs"]["svd"]["size"]
    d = len(cfg["data"]["detectors"])
    target = len(cfg["data"]["inference_parameters"])
    train = int(n * cfg["data"]["train_fraction"])
    result = dict(N=n, B=batch, K=k, R=r, F=f, first_index=first, L=length, D=d,
                  target_dim=target, domain=domain, train_rows=train, validation_rows=n-train,
                  train_batches=math.ceil(train/batch), validation_batches=math.ceil((n-train)/batch),
                  train_last_batch=train % batch or batch,
                  validation_last_batch=(n-train) % batch or batch,
                  duration_seconds=1/domain["delta_f"], implied_sample_rate=2*domain["f_max"],
                  implied_time_samples=round(2*domain["f_max"]/domain["delta_f"]),
                  noise_std=1/math.sqrt(4*domain["delta_f"]), workers=cfg["local"]["num_workers"],
                  requested_matmul_precision=cfg["local"].get("float32_matmul_precision", "unspecified"))
    rows = [
        tensor_row("Stored coefficients (training precision)", (n,k), "complex64", "CPU bank", copies=4),
        tensor_row("Stored dataset basis V", (f,k), "complex64", "CPU bank"),
        tensor_row("Coefficient batch", (batch,k), "complex64", "CPU -> GPU", copies=4),
        tensor_row("ASD batch", (batch,f), "float32", "CPU -> GPU", copies=d),
        tensor_row("Geometry: distance + antenna factors", (batch,), "float32", "CPU -> GPU", copies=2*(1+2*d)),
        tensor_row("Geometry: detector time offsets", (batch,), "float32", "CPU -> GPU", copies=2*d,
                   note="CPU probe observes float32; the GPU preprocessor explicitly casts offsets to float64 before phase evaluation."),
        tensor_row("Standardized targets", (batch,target), "float32", "CPU -> GPU"),
        tensor_row("GPU reconstruction basis Vh", (k,f), "complex64", "GPU resident", note="Separate allocation from CPU V."),
        tensor_row("GPU frequency grid", (f,), "float64", "GPU resident"),
        tensor_row("GPU inverse storage-whitening multiplier", (f,), "float32", "GPU resident", note="All ones for the current config."),
        tensor_row("One reconstructed polarization", (batch,f), "complex64", "GPU temporary"),
        tensor_row("Four logical reconstructed polarizations", (batch,f), "complex64", "GPU logical total", copies=4,
                   note="Logical sum, not simultaneous peak: sources are processed sequentially."),
        tensor_row("One time-phase temporary before cast", (batch,f), "complex128", "GPU temporary"),
        tensor_row("One time-phase factor after cast", (batch,f), "complex64", "GPU temporary"),
        tensor_row("Two summed detector strains", (batch,d,f), "complex64", "GPU logical total"),
        tensor_row("Packaged strain/ASD network input", (batch,d,3,length), "float32", "GPU"),
        tensor_row("RB projection output", (batch,2*r*d), "float32", "GPU"),
        tensor_row("Embedding output", (batch,128), "float32", "GPU"),
        tensor_row("Loss", (), "float32", "GPU", note="Reduction of model log probabilities."),
    ]
    if "flows" in cfg["model"]["posterior_kwargs"]:
        p = cfg["model"]["posterior_kwargs"]["CopulaKwargs"]["P"]
        rows += [tensor_row("Copula head raw output", (batch,target*p+1), "float32", "GPU"),
                 tensor_row("Copula factor B (view of head output)", (batch,target,p), "float32", "GPU view"),
                 tensor_row("Copula z (view of head output)", (batch,), "float32", "GPU view"),
                 tensor_row("Copula Omega / inverse (each)", (batch,target,target), "float32", "GPU temporary"),
                 tensor_row("Woodbury K / Cholesky (each)", (batch,p,p), "float32", "GPU temporary")]
    result["tensors"] = rows
    result["host_to_device_bytes_per_full_batch"] = sum(row["bytes"] for row in rows if row["location"] == "CPU -> GPU")
    result["prefetch_logical_bytes"] = result["host_to_device_bytes_per_full_batch"] * result["workers"] * 2
    # Leading-order real floating-point operation conventions: 8 per complex MAC,
    # 2 per real MAC. Forward only; these are counts, not performance predictions.
    result["reconstruction_forward_flops"] = 4 * 8 * batch*k*f
    result["projection_forward_flops"] = 2 * batch * d * (3*length) * (2*r)
    result["projection_parameters"] = d * (3*length*(2*r) + 2*r)
    result["dense_uncompressed_bank_bytes_complex64"] = n*4*f*8
    return result


def probe_cpu_preparation(cfg, arithmetic):
    """Run the real CPU preparation transforms on two synthetic coefficient rows."""
    import numpy as np
    from types import SimpleNamespace
    from bilby.gw.detector import InterferometerList
    from dingo.gw.domains import build_domain
    from dingo.gw.gwutils import get_extrinsic_prior_dict
    from dingo.gw.transforms import (SampleExtrinsicParameters, GetDetectorTimes,
        ProjectOntoDetectors, SampleNoiseASD, SelectStandardizeRepackageParameters)
    from dingo.gw.training.gpu_preprocessing import PrepareCompressedBatch
    b, k, f = 2, arithmetic["K"], arithmetic["F"]
    domain = build_domain(arithmetic["domain"])
    parameters = {}
    for name in cfg["data"]["inference_parameters"]:
        parameters[name] = np.ones(b, dtype=np.float32)
    for suffix in ("_A", "_B"):
        parameters["luminosity_distance"+suffix] = np.full(b,100, dtype=np.float32)
        parameters["geocent_time"+suffix] = np.zeros(b, dtype=np.float32)
        parameters["theta_jn"+suffix] = np.zeros(b, dtype=np.float32)
        parameters["phase"+suffix] = np.zeros(b, dtype=np.float32)
    coefficients = {"h_"+pol+suffix: np.zeros((b,k), dtype=np.complex64)
                    for suffix in ("_A", "_B") for pol in ("plus", "cross")}
    ifos = InterferometerList(cfg["data"]["detectors"])
    prior = get_extrinsic_prior_dict(cfg["data"]["extrinsic_prior"], True)
    prefix = [SampleExtrinsicParameters(prior, False), GetDetectorTimes(ifos, cfg["data"]["ref_time"], False)]
    asds = SimpleNamespace(sample_random_asds=lambda n: {
        ifo.name: np.full((n,f), 1e-23, dtype=np.float32) for ifo in ifos})
    targets = cfg["data"]["inference_parameters"]
    standardize = SelectStandardizeRepackageParameters(
        {"inference_parameters": targets, "context_parameters": []},
        {"mean": dict.fromkeys(targets,0.), "std": dict.fromkeys(targets,1.)})
    prepare = PrepareCompressedBatch(prefix, ProjectOntoDetectors(ifos,domain,cfg["data"]["ref_time"]),
                                     SampleNoiseASD(asds), standardize)
    prepared = prepare({"parameters":parameters,"waveform":coefficients})
    observed = {group: {name: {"shape":list(v.shape), "dtype":str(v.dtype),"bytes":v.nbytes}
                        for name,v in values.items()} for group,values in prepared.items()}
    size = sum(v["bytes"] for values in observed.values() for v in values.values())
    assert size//b*arithmetic["B"] == arithmetic["host_to_device_bytes_per_full_batch"]
    return {"source":"real CPU preparation classes; synthetic two-row coefficients and ASDs",
            "batch_size":b,"observed":observed,"bytes_per_row":size//b}


def count_models(configs, arithmetic):
    """Count real modules; correct only the affine projection's shortened axis."""
    import torch
    from dingo.core.nn.nsf import create_nsf_with_rb_projection_embedding_net
    from dingo.core.nn.copulaNSF import CopulaNSFFlowWrapper
    torch.set_num_threads(1)
    outputs = {}
    for name, cfg in configs.items():
        a=arithmetic[name];model_cfg=cfg["model"]
        embedding=copy.deepcopy(model_cfg["embedding_kwargs"])
        short_f=256
        embedding.update(input_dims=[a["D"],3,short_f], V_rb_list=None, added_context=False)
        posterior=copy.deepcopy(model_cfg["posterior_kwargs"])
        torch.manual_seed(17)
        if "flows" in posterior:
            model=CopulaNSFFlowWrapper(posterior,embedding)
            sections={"embedding_resnet":model.embedding_net[1], "copula_head":model.copulaNet}
            sections.update({f"marginal_{i+1}":flow for i,flow in enumerate(model.vector_copula.flows)})
        else:
            posterior.update(input_dim=a["target_dim"],context_dim=embedding["output_dim"])
            model=create_nsf_with_rb_projection_embedding_net(posterior,embedding)
            sections={"embedding_resnet":model.embedding_net[1],"joint_flow":model.flow}
        rows={section:sum(p.numel() for p in module.parameters()) for section,module in sections.items()}
        rows["rb_projection"]=a["projection_parameters"]
        short_projection=sum(p.numel() for p in model.embedding_net[0].parameters())
        total=sum(p.numel() for p in model.parameters())-short_projection+rows["rb_projection"]
        assert sum(rows.values())==total
        info={"method":"Actual CPU module construction with frequency axis 256; exact affine parameter-count correction to configured L.",
              "sections":rows,"total":total,"weights_bytes_float32":4*total,
              "stage_0_trainable":total-rows["rb_projection"],"stage_1_trainable":total}
        info["stage_0_weights_grads_adam_bytes"]=4*total+12*info["stage_0_trainable"]
        info["stage_1_weights_grads_adam_bytes"]=16*total
        model.eval()
        x=torch.randn(2,a["D"],3,short_f);theta=torch.zeros(2,a["target_dim"])
        with torch.no_grad():
            projection=model.embedding_net[0](x)
            context=model.embedding_net[1](projection)
            info["synthetic_projection_shape"]=list(projection.shape)
            info["synthetic_context_shape"]=list(context.shape)
            if "flows" in posterior:
                raw=model.copulaNet.net(context)
                info["synthetic_copula_head_shape"]=list(raw.shape)
                calls=[0 for _ in model.vector_copula.flows]
                def count_transform(index):
                    def hook(module, inputs, result):
                        calls[index]+=1
                    return hook
                handles=[flow.flow._transform.register_forward_hook(count_transform(i))
                         for i,flow in enumerate(model.vector_copula.flows)]
                try:
                    density=model(theta,x)
                finally:
                    for handle in handles:handle.remove()
                info['marginal_transform_calls']=calls
                info["observed_wrapper_log_prob_shape"]=list(density.shape)
                info["expected_wrapper_log_prob_shape"]=list(theta.shape[:-1])
                info["observed_wrapper_finite"]=bool(torch.isfinite(density).all())
            else:
                density=model.flow.log_prob(theta, context=context)
                info["direct_flow_log_prob_shape"]=list(density.shape)
                try:
                    density=model(theta,x)
                    info["observed_wrapper_log_prob_shape"]=list(density.shape)
                except Exception as error:
                    info["observed_wrapper_error"]=f"{type(error).__name__}: {error}"
        outputs[name]=info
        print(name, 'parameters:',f'{total:,}', 'CPU shape probe complete',flush=True)
        del model,sections,x,theta,context,projection,density
        gc.collect()
    return outputs


def profile_summary(path):
    """Stream large profiler traces without materializing millions of Python dicts."""
    decoder=json.JSONDecoder()
    def events():
        with path.open() as source:
            buffer=source.read(1<<20)
            marker=buffer.index('"traceEvents"')
            header=json.loads(buffer[:marker].rstrip().rstrip(',')+'}')
            result["device"]=header["deviceProperties"][0]["name"]
            result["cuda_runtime_version"]=header.get("cuda_runtime_version")
            pos=buffer.index('[',marker)+1
            while True:
                if pos>len(buffer)-(1<<16):
                    buffer=buffer[pos:]+source.read(1<<20);pos=0
                while pos<len(buffer) and buffer[pos] in ' \r\n\t,':pos+=1
                if pos<len(buffer) and buffer[pos]==']':return
                try:event,end=decoder.raw_decode(buffer,pos)
                except json.JSONDecodeError:
                    extra=source.read(1<<20)
                    if not extra:raise
                    buffer=buffer[pos:]+extra;pos=0;continue
                pos=end
                yield event
    result={"path":str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            "evidence":"historical CUDA profiler, not a current-config benchmark"}
    aggregate=collections.defaultdict(lambda:[0,0.]);copies=collections.defaultdict(lambda:[0,0,0.])
    regions=collections.defaultdict(list);count=collections.Counter();shapes=collections.Counter();steps=[];gpu=[];kernels=[]
    for e in events():
        count[e.get("cat","")]+=1
        if e.get("ph")!="X":continue
        category,name,d=e.get("cat"),e.get("name"),e.get("dur",0)
        if category=="user_annotation":
            if name.startswith("ProfilerStep"):steps.append((e["ts"],e["ts"]+d))
            if name in ("preprocessing_and_transfer","forward_and_loss","backward","optimizer"):
                regions[name].append(d)
        if category in ("kernel","cuda_runtime","gpu_memcpy"):
            aggregate[(category,name)][0]+=1;aggregate[(category,name)][1]+=d
        if category in ("kernel","gpu_memcpy","gpu_memset"):
            gpu.append((e["ts"],e["ts"]+d))
        if category=="kernel":kernels.append(d)
        if category=="gpu_memcpy":
            copies[name][0]+=1;copies[name][1]+=e.get("args",{}).get("bytes",0);copies[name][2]+=d
        if category=="cpu_op" and name=="aten::mm":
            shapes[str(e.get("args",{}).get("Input Dims",[]))]+=1
    result["captured_steps"]=len(steps)
    result["wall_seconds"]=sum(b-a for a,b in steps)/1e6
    result["categories"]=dict(count)
    result["transfers"]={name:{"calls":n,"bytes":b,"duration_ms":d/1000,
                               "active_copy_GB_per_second":b/(d*1000) if d else 0}
                         for name,(n,b,d) in copies.items()}
    result["kernel_median_us"]=statistics.median(kernels)
    result["kernel_under_10us_percent"]=100*sum(d<10 for d in kernels)/len(kernels)
    result["cpu_region_mean_ms"]={name:statistics.mean(ds)/1000 for name,ds in regions.items()}
    result["runtime"]={name:{"calls":n,"duration_ms":d/1000} for (cat,name),(n,d) in aggregate.items() if cat=="cuda_runtime"}
    result["top_kernels"]=[{"name":name,"calls":n,"duration_ms":d/1000}
                            for (cat,name),(n,d) in sorted(aggregate.items(),key=lambda v:-v[1][1]) if cat=="kernel"][:12]
    result["matrix_shapes"]=[{"shapes":shape,"calls":n} for shape,n in shapes.most_common(12)]
    intervals=sorted((max(a,x),min(b,y)) for a,b in steps for x,y in gpu if x<b and y>a)
    end=-math.inf;busy=0
    for a,b in intervals:
        if b>end:busy+=b-max(end,a);end=b
    result["gpu_active_seconds_in_steps"]=busy/1e6
    result["warning"]="Stack/shape/memory recording perturbs execution. Region elapsed times are CPU launch scopes; do not treat them as exclusive CUDA time."
    return result


def write_csv(path, rows):
    if not rows:return
    with path.open('w',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def make_diagram(report, output):
    os.environ.setdefault('MPLCONFIGDIR',tempfile.mkdtemp(prefix='dingo-report-mpl-'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    a=report['arithmetic']['copula'];b=a['B'];k=a['K'];f=a['F'];length=a['L'];r=a['R']
    fig,ax=plt.subplots(figsize=(11.6929,8.2677))
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1);ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
    fig.patch.set_facecolor('white')
    ax.text(.025,.965,'DINGO two-source training | CPU / PCIe / GPU',fontsize=17,weight='bold',color='#16324f')
    ax.text(.025,.932,f'CURRENT CONFIG: N={a["N"]:,} A/B pairs | batch={b:,} | H1 + L1 | 64 s, 10–1024 Hz | Δf=1/64 Hz | float32 + high matmul precision',fontsize=8.5)
    colors={'cpu':('#eaf2fb','#346b9c'),'gpu':('#e9f6ef','#38835e'),'model':('#f0ecfa','#7861a5'),'warn':('#fff3e3','#b47829')}
    def box(x,y,w,h,title,body,kind='gpu'):
        fill,edge=colors[kind]
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.006',linewidth=1.1,edgecolor=edge,facecolor=fill))
        ax.text(x+.009,y+h-.022,title,fontsize=9,weight='bold',va='top',color=edge)
        ax.text(x+.009,y+h-.047,body,fontsize=7.4,va='top',linespacing=1.35)
    def arrow(points,dashed=False):
        for p,q in zip(points[:-2],points[1:-1]):ax.plot([p[0],q[0]],[p[1],q[1]],color='#526777',lw=1,ls='--' if dashed else '-')
        ax.annotate('',xy=points[-1],xytext=points[-2],arrowprops=dict(arrowstyle='->',color='#526777',lw=1,linestyle='--' if dashed else '-'))
    box(.025,.755,.275,.14,'1  OFFLINE / CPU: generate + compress',
        f'IMRPhenomXPHM: A and B → h+, h×\nEach series: {f:,} complex frequency bins\nShared rank-{k:,} dataset SVD\n4 coefficient banks: [{a["N"]:,}, {k:,}]\nTraining-precision payload: {a["N"]*4*k*8/2**30:.2f} GiB','cpu')
    box(.025,.575,.275,.14,'2  CPU: load bank + split / workers',
        f'HDF5 → local SSD cache → host RAM\n190,000 train / 10,000 validation rows\nSeed 42 split; {a["workers"]} workers per loader\nPinned buffers; persistent workers\nNo full-bank GPU decompression','cpu')
    box(.025,.395,.275,.14,'3  CPU: prepare one batch',
        f'4 × [{b:,}, {k:,}] complex64 coefficients\nSample extrinsics + detector geometry\n2 × [{b:,}, {f:,}] float32 ASDs\nTargets [{b:,}, 30] float32\nGeometry: 14 float32 vectors / row','cpu')
    box(.025,.245,.275,.11,'4  PCIe: copy batch → GPU',
        f'Coefficients: {4*b*k*8/2**20:.2f} MiB; ASDs: {2*b*f*4/2**20:.2f} MiB\nTotal: {a["host_to_device_bytes_per_full_batch"]/2**20:.2f} MiB / full batch\nBasis copied at setup, reused on GPU','warn')
    box(.35,.755,.30,.14,'5  GPU: reconstruct polarizations',
        f'4 matmuls: [B,{k}] @ [{k},{f}]\nEach output: [B,{f}] complex64\nGPU basis: {k*f*8/2**20:.2f} MiB resident\nOne polarization: {b*f*8/2**20:.2f} MiB\nDataset whitening inverse = identity here')
    box(.35,.565,.30,.145,'6  GPU: project + time shift + sum',
        'For each source s and detector d:\n(h+ F+ + h× F×) × (100 / distance)\n× exp(−2π i f Δt); then h_d = h_d,A + h_d,B\nTwo detector strains: [B,65537] each\nTime exponential uses complex128 → complex64')
    box(.35,.385,.30,.135,'7  GPU: whiten + noise + package',
        f'h_d / (ASD_d × 4) + (N(0,1) + i N(0,1))\nKeep bins 640…65536 → {length:,} bins\nX: [B,2,3,{length}] float32\nChannels: Re(h), Im(h), 1/(ASD × 10²³)\nNetwork input: {b*2*3*length*4/2**20:.2f} MiB')
    box(.35,.19,.30,.145,'8  GPU: embedding (both models)',
        f'Per detector: [B,{3*length}] @ [{3*length},{2*r}]\nConcatenate → [B,{2*r*a["D"]}]\n24 residual blocks → context [B,128]\nRB projection: {a["projection_parameters"]:,} parameters\nStage 0 frozen; stage 1 trainable','model')
    cop=report['models']['copula'];full=report['models']['full']
    box(.705,.735,.27,.16,'9a  GPU: copula model',
        f'Targets [B,30] → two [B,15] marginals\nEach: 30 flow steps, width 1024, 5 blocks\nHead [B,481] → factor [B,30,16] + z\nWoodbury K [B,16,16]; Ω [B,30,30]\nTotal weights: {cop["total"]/1e6:.1f} M\nCPU probe: transform calls {cop["marginal_transform_calls"]}','model')
    box(.705,.54,.27,.14,'9b  GPU: full joint model',
        f'Targets [B,30] + context [B,128]\nOne joint flow: 30 steps\nConditioners: width 1024, 5 residual blocks\nTotal weights: {full["total"]/1e6:.1f} M\nAlternative to 9a; not chained after it','model')
    box(.705,.325,.27,.16,'10  GPU: density → scalar loss',
        f'Intended per-example log q(θ | X): [B]\nLoss = −mean(log q)\nObserved local copula probe: {cop["observed_wrapper_log_prob_shape"]}\nFull wrapper probe: {full.get("observed_wrapper_log_prob_shape", "context error")}\nShape checks use the current checkout\nCheck phase / inclination application too','warn')
    box(.705,.115,.27,.16,'11  GPU + CPU: gradients / update',
        'GPU backward → parameter gradients\nGPU Adam: weights + first/second moments\nCPU scheduler, loss logging, checkpoints\nValidation: same data path, no backward\nWeights + grads + Adam (stage 1):\n'+f'copula {cop["stage_1_weights_grads_adam_bytes"]/2**30:.2f} GiB; full {full["stage_1_weights_grads_adam_bytes"]/2**30:.2f} GiB','model')
    arrow([(.162,.755),(.162,.715)]);arrow([(.162,.575),(.162,.535)]);arrow([(.162,.395),(.162,.355)])
    arrow([(.30,.30),(.325,.30),(.325,.825),(.35,.825)])
    for start,end in [(.755,.710),(.565,.520),(.385,.335)]:arrow([(.50,start),(.50,end)])
    arrow([(.65,.26),(.68,.26),(.68,.81),(.705,.81)])
    arrow([(.68,.61),(.705,.61)])
    arrow([(.975,.80),(.99,.80),(.99,.505),(.84,.505),(.84,.485)])
    arrow([(.84,.54),(.84,.485)]);arrow([(.84,.325),(.84,.275)])
    ax.text(.025,.175,'READ THIS SHEET WITH pipeline_report.md',fontsize=8,weight='bold',color='#16324f')
    ax.text(.025,.145,'Shapes/bytes: calculated from current configs.\nCounts: real module construction + exact RB count.\nWarnings: observed source / synthetic CPU probes.\nTemporary sizes are not a peak-memory estimate.',fontsize=7.2,linespacing=1.5)
    if report.get('profile'):
        prof=report['profile'];n=prof['captured_steps'];h=prof['transfers'].get('Memcpy HtoD (Pinned -> Device)',{})
        footer=f'HISTORICAL TRACE 62491090: H100, B=512; {h.get("bytes",0)/n/2**20:.2f} MiB host→GPU / batch; {h.get("duration_ms",0)/n:.2f} ms copy time. Not a TF32/B=1024 measurement.'
        ax.text(.025,.055,footer,fontsize=7.7,color='#16324f')
    ax.text(.025,.025,'Frequency-domain data: 64 s is the reciprocal grid spacing; 131072 implied real time samples at 2048 Hz. No time-domain series is passed to the network.',fontsize=7.3)
    for extension in ('pdf','svg','png'):
        fig.savefig(output/f'pipeline_A4.{extension}',dpi=180,facecolor='white')
    plt.close(fig)


def write_markdown(report, output):
    a=report['arithmetic']['copula'];b=a['B'];n=a['N'];f=a['F'];l=a['L'];k=a['K'];r=a['R']
    lines=['# Dingo CPU / PCIe / GPU computational reference','',
           'Companion: `pipeline_A4.pdf` (one landscape A4 page), `pipeline_report.json`, `tensor_sizes.csv`, `model_parameters.csv`.', '',
           'This report describes the current checkout. It does not change training code, configs, checkpoints, or the HDF5 banks.', '',
           '## Evidence and scope','',
           '- **Calculated**: current YAML settings and array arithmetic; not an assertion about every remote HDF5 file.',
           '- **CPU probe**: real preparation/model classes with synthetic inputs. Model probes use batch 2 and 256 frequency bins to avoid large allocations; architecture widths, steps and rank are unchanged.',
           '- **Parameter counts**: real instantiated modules; the shortened RB affine layer is corrected exactly to the configured frequency count.',
           '- **Measured CUDA**: only the named historical profiler window. Instrumentation perturbs timings.',
           '- **Dataset metadata**: read from the resolved data directory without loading polarization arrays. Missing files are recorded as absent; inspected shapes/dtypes appear below. Configured row counts are not inferred from a different local dataset.', '',
           '## Resolved paths', '', '```json',json.dumps(report['paths'],indent=2),'```', '',
           '## 1. Configuration and discretization','',
           f'- Generation: IMRPhenomXPHM, f_ref=20 Hz, {n:,} joint A/B rows = {2*n:,} component-source waveforms.',
           f'- Frequency grid: Δf={a["domain"]["delta_f"]} Hz, f_max=1024 Hz; F=f_max/Δf+1={f:,} complex bins including DC and Nyquist.',
           f'- Retain f≥10 Hz: min index={a["first_index"]}; L={l:,} bins passed to the model.',
           '- Reciprocal grid spacing is 64 s; the corresponding real FFT convention implies 2048 Hz and 131,072 time samples. Training uses frequency arrays, not a 131,072-step time-series network.',
           f'- Batch size={b:,}; {a["workers"]} workers per loader; one GPU in the current Slurm script.',
           f'- Split: {a["train_rows"]:,} train and {a["validation_rows"]:,} validation, using random_split generator seed **42**.',
           f'- Per epoch: {a["train_batches"]} training batches (last={a["train_last_batch"]}) and {a["validation_batches"]} validation batches (last={a["validation_last_batch"]}). This assumes drop_last=False and a full {n:,}-row generated file.',
           '- The split seed is fixed; extrinsic/noise draws and model initialization are separate randomness sources.', '',
           '## 2. Offline generation: parameters → polarizations → stored coefficients','',
           'Source: `configs/waveform_dataset_settings.yml`; `Pipeline/Dingo_NF_pipeline.py:generate_waveform_dataset`; `External/dingo/dingo/gw/dataset/generate_dataset.py`.', '',
           '1. Draw constrained masses and spins for A and B; derived masses may be added by prior conversion. Mass constraints are rejection/selection rules, not additional inferred coordinates.',
           '2. Strip A/B suffixes and call the waveform generator for each source. It generates h_plus and h_cross directly; polarizations are not derived from detector strains.',
           '3. Generation config fixes distance=100, inclination theta_jn=0, orbital phase=0, geocentric time=0. The overlap time convention is t_B=t_A−delta_t_AB, giving t_B=0 during generation.',
           f'4. Dataset SVD training uses 50,000 joint rows plus 10,000 validation rows. All four polarization sets are stacked to learn a shared rank-{k} basis: 200,000 training and 40,000 validation frequency series if generation succeeds as requested.',
           f'5. With V:[{f},{k}], compression is h:[F] @ V → c:[K]; reconstruction is c @ Vᴴ. The stored bank contains four arrays [N,K], not one summed detector signal.',
           '- This config requests SVD compression only. No storage whitening or heterodyning is configured. Generation failures can change temporary counts; generation requests replacement rows to reach the requested main count.',
           '- Saved generation arrays may be complex128. Training explicitly loads coefficients/basis as complex64 and parameter columns as float32. File bytes therefore require HDF5 dtype inspection.',
           f'- Main coefficient bank in training precision: {mib(n*4*k*8)} = {n*4*k*8/2**30:.3f} GiB. At six million joint rows: {6_000_000*4*k*8/2**30:.3f} GiB before parameters/overhead.',
           f'- A hypothetical fully reconstructed main bank at complex64 would be {a["dense_uncompressed_bank_bytes_complex64"]/2**30:.2f} GiB. It is never reconstructed as a whole for GPU training.', '',
           '## 3. ASD creation and loading','',
           'Source: `configs/asd_dataset_settings.yml`; `Pipeline/Dingo_NF_pipeline.py:generate_asd_dataset`; `External/dingo/dingo/gw/noise/asd_dataset.py`.', '',
           '- The custom generation method reads the reference detector\'s configured two-column curve, interpolates to the waveform grid, and takes sqrt because the input is treated as a PSD.',
           '- It copies that reference curve to each configured detector. The current H1/L1 configs point to the same Cosmic Explorer design PSD, so this is consistent for these configs.',
           '- One float32 ASD curve per detector is stored: each [1,F]. The method packages design noise; it does not estimate new ASDs from 1024 s of observed data despite the generic PSD-estimation fields in the YAML.',
           '- During batch preparation, ASD sampling selects rows and materializes [B,F] for each detector. Even one fixed curve is currently expanded and transferred every batch.', '',
           '## 4. CPU training preparation and PCIe','',
           'Source: `waveform_dataset.py:__getitems__`; `torchutils.py:build_train_and_test_loaders`; `gpu_preprocessing.py:PrepareCompressedBatch`.', '',
           '- Local SSD caching copies the HDF5 file once. With leave_waveforms_on_disk=False, the compressed bank is loaded into host RAM; ordinary training then indexes RAM, not the shared filesystem for each sample.',
           '- Dataset reads select B rows and skip automatic decompression when defer_decompression=True.',
           '- Sample distance, sky position, polarization, time, inclination and phase using the training priors. t_B=t_A−delta_t_AB is derived, not independently drawn.',
           '- Compute arrival times from the sky position, fixed ref_time and detector vertex. Compute antenna responses F_plus/F_cross on the CPU using Bilby geometry.',
           '- Prepare 14 geometry vectors: two distance ratios, eight antenna factors, four detector-time offsets. The CPU probe checks the resulting dtypes.',
           '- Standardize the 30 targets using (value−mean)/std; extrinsic_parameters supersedes stored parameters. Targets include delta_t_AB, while t_B and per-detector times are derived auxiliary values.',
           '- The dataset unbatches its nested NumPy dictionary. DataLoader collates it to tensors and uses pinned host buffers. These indexing/collation/pinning steps can make copies.',
           f'- Expected full-batch host→GPU tensor payload: **{mib(a["host_to_device_bytes_per_full_batch"])}**. This is logical tensor payload, not PCIe protocol traffic or a measured transfer time.',
           '- non_blocking=True allows asynchronous transfer from pinned buffers; copies and work on the current stream remain ordered. There is no dedicated next-batch CUDA-stream prefetcher.',
           f'- With the default prefetch_factor=2 and {a["workers"]} workers, up to 32 scheduled batches correspond to {a["prefetch_logical_bytes"]/2**30:.2f} GiB of logical payload. This is not a peak-RSS prediction: batch construction, pinning, outstanding tasks, shared memory, and validation affect residency.',
           '- Train and validation each create their own persistent worker pool. Process-tree RSS can double-count the shared bank; use job accounting/PSS for unique host-memory attribution.', '',
           '## 5. GPU reconstruction, geometry application, whitening and packaging','',
           'Source: `gpu_preprocessing.py:GPUWaveformPreprocessor`.', '',
           f'1. At stage setup, move Vᴴ:[K,F] complex64, frequencies:[F] float64 and storage-whitening multiplier:[F] float32 to GPU buffers. Basis payload={mib(k*f*8)}; it is reused, not transferred every batch.',
           f'2. For A and B separately: reconstruct h+ and h× with [B,{k}] @ [{k},{f}] → [B,{f}] complex64. Four matrix multiplications per batch. Inverse storage whitening is multiplication by ones in this config.',
           '3. For each source s and detector d, compute r_s=100/distance_s and h_d,s(f)=(F_plus,d,s h_plus,s + F_cross,d,s h_cross,s) r_s exp(−2π i f (t_d,s−t_reference,s)).',
           '4. Detector-time offsets arrive as float32 in the CPU probe, are explicitly cast to float64, and are expanded against the float64 frequency grid. The cast cannot recover precision already lost in CPU float32 time arithmetic. The complex exponential has a complex128 temporary, then is cast to the polarization dtype complex64.',
           '5. Sum h_d,A+h_d,B before whitening and noise. Four source/detector projections produce two summed detector arrays.',
           f'6. Whiten and normalize: h_d / (ASD_d × noise_std), with noise_std=1/sqrt(4Δf)={a["noise_std"]:g}. Add independent standard-normal real and imaginary components.',
           f'7. Slice bins [{a["first_index"]}:] and stack channels [Re(h), Im(h), 1/(ASD×1e23)] → X:[B,2,3,{l}] float32.',
           f'- One full batch X occupies {mib(b*2*3*l*4)}. Temporary arrays are additional allocations; their logical sizes must not be summed into an assumed simultaneous peak.',
           f'- Reconstruction leading-order forward work: {a["reconstruction_forward_flops"]/1e12:.3f} TFLOP/batch using 8 real FLOPs per complex multiply-accumulate. This excludes all other preprocessing and makes no throughput prediction.',
           '- The training helper disables autocast and gradients around preprocessing. Network float32 matmul precision does not turn the complex reconstruction into a reduced-storage TF32 tensor.', '',
           '## 6. Shared embedding: input frequency tensor → 128 context features','',
           'Source: `External/dingo/dingo/core/nn/enets.py`; `resnet.py`.', '',
           f'- Each detector is flattened to [B,3L]=[B,{3*l}]. Its learned affine layer has weight [{2*r},{3*l}] and bias [{2*r}]. Detector outputs concatenate to [B,{2*r*a["D"]}].',
           f'- The embedding projection is initialized from a separately trained rank-{r} detector basis; it is distinct from the storage rank-{k} basis. Initial ASD-channel weights are zero; they can become nonzero in stage 1.',
           '- Initialization of this detector basis uses clean projected/whitened examples (10,000 training and 2,000 validation samples per the current embedding config). Its CPU build/cache is startup work, not a per-epoch operation.',
           f'- RB projection parameters: {a["projection_parameters"]:,}, including biases. Forward affine work: approximately {a["projection_forward_flops"]/1e12:.3f} TFLOP/batch.',
           f'- Residual network: {2*r*a["D"]}→1024 initial linear, then six blocks each at widths 1024, 512, 256 and 128 (24 blocks total), with resize layers between width groups. Each block has two linear layers, BatchNorm and ELU; final output is [B,128].', '',
           '## 7. Conditional densities: full model versus copula model','',
           '### Full joint model','',
           '- Inputs: standardized theta [B,30] and context [B,128]. One conditional flow transforms all 30 coordinates together.',
           '- There are 30 steps, each consisting of a permutation/LU transform and rational-quadratic coupling, followed by one final permutation/LU.',
           '- Each coupling keeps 15 features and transforms 15. Conditioner input is [B,15+128]=[B,143]; width 1024, five two-linear residual blocks with context gating; output [B,345]=15×(3×8−1).',
           '- The eight-bin spline uses widths, heights and seven internal derivatives per transformed coordinate, with linear tails. The flow accumulates log-Jacobian determinants and standard-normal base density to produce intended log q:[B].', '',
           '### Copula model','',
           '- The same context conditions two 15-dimensional marginal flows. Each has 30 steps plus final LU. Its alternating split has 7/8 identity and 8/7 transformed features.',
           '- Conditioner input widths are 135/136 (identity + 128 context); spline-parameter output widths are 184/161 (transformed×23). The conditioner hidden width/block count matches the full model.',
           '- The copula head maps context [B,128] through five width-1024 residual blocks to [B,481]. Split into factor B_factor:[B,30,16] and z:[B]. Batch B and factor B_factor are different quantities.',
           '- z is mapped to positive zeta. Omega_tilde=zeta I_30+B_factor B_factor^T has shape [B,30,30]. Block normalization uses the two 15-dimensional blocks. Woodbury calculations build K=zeta I_16+B_factor^T B_factor and its Cholesky [B,16,16], plus [B,30,30] inverse-related arrays.',
           '- Marginal base coordinates concatenate to Q:[B,30]. The copula correction combines a log determinant with Q^T(Omega^-1−I)Q, then adds marginal log densities.',
           '- Dingo obtains each marginal density and its base coordinates in one shared transform evaluation. A and B retain separate parameters; the copula reuses each marginal\'s latent output with gradients attached. BatchNorm running statistics update once per marginal evaluation.',
           f'- Synthetic CPU probe, transform calls per marginal: {report["models"]["copula"]["marginal_transform_calls"]}.', '',
           '## 8. Source behavior and shape checks','',
           '- **Inclination/phase augmentation:** waveform generation fixes theta_jn and phase to zero. The inspected CPU/GPU projection applies distance, sky/polarization responses and arrival-time shifts, but does not apply newly sampled theta_jn or orbital phase to reconstructed polarizations. Those values nevertheless enter standardized targets. This is a source-level consistency concern; regenerate/compare physical waveforms before treating those coordinates as learned from the data.',
           f'- **Full wrapper context:** context arguments remain a tuple until the embedding consumes complete batches. The CPU wrapper probe returns {report["models"]["full"].get("observed_wrapper_log_prob_shape")}; error={report["models"]["full"].get("observed_wrapper_error")}. The previous row-unpacking error is fixed in FlowWrapper, including sampling and unconditional calls.',
           f'- **Copula output shape:** the two-row CPU probe returns {report["models"]["copula"]["observed_wrapper_log_prob_shape"]}, expected [2]. Training marginals remain [B]; both density terms are [1,B] before addition, and the final result is [B]. A shape guard rejects unintended broadcasting. The previous extra-axis bug produced [B,B]; regenerate historical reports to update their labels.',
           '- This tool reports current source behavior; it does not modify training code.', '',
           '## 9. Parameter and optimizer memory','',
           '| Model | Component | Parameters | FP32 parameter bytes |','|---|---|---:|---:|']
    for name,model in report['models'].items():
        for section,count in model['sections'].items():lines.append(f'| {name} | {section} | {count:,} | {mib(count*4)} |')
        lines.append(f'| **{name}** | **TOTAL** | **{model["total"]:,}** | **{model["weights_bytes_float32"]/2**30:.3f} GiB** |')
    lines += ['', '| Model | Trainable stage 0 | Trainable stage 1 | Stage 0 weights+grads+Adam | Stage 1 weights+grads+Adam |',
              '|---|---:|---:|---:|---:|']
    for name,m in report['models'].items():
        lines.append(f'| {name} | {m["stage_0_trainable"]:,} | {m["stage_1_trainable"]:,} | {m["stage_0_weights_grads_adam_bytes"]/2**30:.3f} GiB | {m["stage_1_weights_grads_adam_bytes"]/2**30:.3f} GiB |')
    lines += ['', 'These are FP32 storage arithmetic: 4 bytes per weight plus 4 gradient + 4 first-moment + 4 second-moment bytes per trainable parameter. Adam moments are allocated lazily. This excludes activations, buffers, temporary optimizer tensors, CUDA context, allocator reserve, and preprocessing. It is not a prediction of peak GPU memory.', '',
              'Stage 0 (100 epochs): freeze layers_rb; train the remaining network, Adam lr=1e-4. Stage 1 (400 epochs): unfreeze projection; Adam lr=5e-5. Early stopping can shorten stages. Both use batch 1024. Validation uses the preprocessing and model forward path, with no gradients or optimizer update.', '',
              '## 10. Precision and gradient flow','',
              '- Coefficients/reconstruction/physical strains: complex64 in single-precision training. Frequencies and phase exponent calculation: float64/complex128, with the result cast back. ASD/targets/network inputs/weights: float32.',
              '- Requested local.float32_matmul_precision=high is applied by the custom pipeline before preparation; it permits accelerated internal arithmetic for eligible real float32 matmuls. It does not change tensor storage sizes. Actual kernel choice is hardware/library/shape dependent.',
              '- automatic_mixed_precision is not enabled in either stage. TF32 permission and AMP are separate settings.',
              '- Preprocessing is no_grad. Backward builds gradients for the model only; stage-0 frozen projection stores no weight gradients. The CPU launches operations, GPU executes kernels. Scheduler/logging/checkpoint I/O return to CPU and disk.',
              '- Loss logging and conditional checks can require scalar GPU→CPU transfers. A CPU region duration includes launch/wait overhead, not just kernel execution.', '',
              '## 11. Array inventory','',
              'Sizes are logical tensor payloads. Views, copies, lifetimes and temporary arrays prevent summing this table into a peak-memory estimate.', '',
              '| Array | Shape (one copy) | Copies | Dtype | Location | Payload |','|---|---|---:|---|---|---:|']
    for row in a['tensors']:
        lines.append(f'| {row["name"]} | {row["shape"]} | {row["copies"]} | {row["dtype"]} | {row["location"]} | {mib(row["bytes"])} |')
    lines += ['', '## 12. Historical measured evidence','']
    if report.get('profile'):
        p=report['profile'];s=p['captured_steps'];lines += [f'Trace: `{p["path"]}`; device={p["device"]}; {s} captured steps; {p["wall_seconds"]:.3f} s CPU-step wall span.',
            'This is job 62491090, batch 512, prior to the current TF32/batch-1024 experiment. Its matrix input dimensions verify batch 512. Do not label it as a benchmark of the current settings.', '',
            '| Transfer category | Calls | Total payload | CUDA copy time | Active-copy rate |','|---|---:|---:|---:|---:|']
        for name,v in p['transfers'].items():lines.append(f'| {name} | {v["calls"]:,} | {mib(v["bytes"])} | {v["duration_ms"]:.3f} ms | {v["active_copy_GB_per_second"]:.3f} GB/s |')
        h=p['transfers']['Memcpy HtoD (Pinned -> Device)'];sync=p['runtime'].get('cudaStreamSynchronize',{})
        lines += ['',f'- Host→GPU per batch: {mib(h["bytes"]/s)}, {h["duration_ms"]/s:.3f} ms. Total payload divided by captured wall time: {h["bytes"]/p["wall_seconds"]/1e9:.3f} GB/s. Active-copy bandwidth and whole-window average describe different things.',
            f'- Large pinned copies occupy {100*h["duration_ms"]/1000/p["wall_seconds"]:.3f}% of this instrumented window. This does not suggest sustained PCIe bandwidth saturation.',
            f'- Kernels: {p["categories"].get("kernel",0):,}, median {p["kernel_median_us"]:.3f} µs; {p["kernel_under_10us_percent"]:.2f}% shorter than 10 µs.',
            f'- Stream synchronizations: {sync.get("calls",0):,}; recorded CPU runtime total {sync.get("duration_ms",0):.3f} ms. This is not all removable overhead.',
            f'- GPU activity union inside step intervals: {p["gpu_active_seconds_in_steps"]:.3f} s. Profiling overhead prevents interpreting this fraction as normal-run utilization.',
            '- Operator-table CPU/CUDA parent/child rows overlap; percentages above 100% are not hardware utilization. Allocation columns are cumulative allocation traffic, not peak live bytes.',
            '- Unprofiled later epochs in the same log: typical training 450 s for 190,000 rows (~422 samples/s), validation 9 s. First-epoch profiler export caused a ~236 s timer jump. These timings cover the earlier frozen-projection stage only.']
    lines += ['', '## 13. Local artifact metadata','', '```json',json.dumps(report['local_artifacts'],indent=2),'```','',
              '## 14. CPU probe results','', '```json',json.dumps({n:{k:v for k,v in m.items() if 'shape' in k or 'error' in k or 'finite' in k} for n,m in report['models'].items()},indent=2),'```','',
              'CPU preparation dtypes/shapes and exact config hashes are in pipeline_report.json. Byte-count agreement is asserted by the reporting script.', '',
              '## 15. Reproduce and update','', 'From the project root:', '', '```bash',
              '.venv/bin/python scripts/report_dingo_pipeline.py',
              '# On VSC, use the Python environment containing this Dingo checkout:',
              'python scripts/report_dingo_pipeline.py --profile /path/to/new_trace.json',
              '# Faster rebuild from configs only, without parsing a large trace:',
              'python scripts/report_dingo_pipeline.py --skip-profile',
              '# Explicit paths for a separate code / DATA / SCRATCH layout:',
              'python scripts/report_dingo_pipeline.py --config-dir /path/to/configs --data-dir /path/to/datasets --skip-profile', '```', '',
              'A new historical trace can be supplied, but compare its actual shapes/settings to the config snapshot; the script does not assume they match. The default historical prose identifies job 62491090; update that interpretation when analyzing a different experiment.', '',
              'The report does not generate a new physical dataset or establish posterior correctness. All source issues above are observations for review, not unsolicited changes.']
    (output/'pipeline_report.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-dir',type=Path,help='YAML directory; defaults to GW_INP_DIR/configs or VSC_DATA/GW_separation/configs on VSC.')
    parser.add_argument('--data-dir',type=Path,help='HDF5 metadata directory; defaults to GW_DATA_DIR or VSC_SCRATCH/GW_separation/data on VSC.')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--profile',type=Path)
    parser.add_argument('--skip-profile',action='store_true')
    args=parser.parse_args()
    import yaml
    paths=resolve_report_paths(args.config_dir,args.data_dir)
    config_dir=Path(paths['config_dir']);data_dir=Path(paths['data_dir'])
    work_dir=Path(paths['work_dir'])
    files=[config_dir/name for name in ('training_copula.yml','training_full.yml','waveform_dataset_settings.yml','asd_dataset_settings.yml')]
    for path in files:
        if not path.is_file():
            parser.error(f'Missing configuration: {path}. Set --config-dir to the YAML directory or GW_INP_DIR to the DATA project directory.')
    output=(args.output_dir or work_dir/'Output/HPC_diagnostics/pipeline_report').resolve()
    profile=args.profile or work_dir/'Output/HPC_diagnostics/profiler/job_62491090_rank_0_epoch_1.json'
    output.mkdir(parents=True,exist_ok=True)
    print('Configuration directory:',config_dir,flush=True)
    print('Dataset metadata directory:',data_dir,flush=True)
    configs={name:yaml.safe_load((config_dir/f'training_{name}.yml').read_text()) for name in ('copula','full')}
    wave=yaml.safe_load((config_dir/'waveform_dataset_settings.yml').read_text())
    arithmetic={name:config_arithmetic(wave,cfg) for name,cfg in configs.items()}
    report={'evidence_labels':['config-calculated','synthetic CPU probe','local HDF5 metadata','historical CUDA profile'],
            'paths':paths,
            'config_paths':{p.name:str(p) for p in files},
            'config_sha256':{f'configs/{p.name}':hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            'arithmetic':arithmetic,
            'local_artifacts':[inspect_hdf5(data_dir/name) for name in ('waveform_dataset.hdf5','asd_dataset.hdf5')]}
    report['cpu_preparation_probe']=probe_cpu_preparation(configs['copula'],arithmetic['copula'])
    report['models']=count_models(configs,arithmetic)
    if not args.skip_profile and profile.exists():
        print('Streaming historical CUDA trace; this may take about a minute...',flush=True)
        report['profile']=profile_summary(profile.resolve())
    (output/'pipeline_report.json').write_text(json.dumps(report,indent=2)+'\n')
    write_csv(output/'tensor_sizes.csv',arithmetic['copula']['tensors'])
    write_csv(output/'model_parameters.csv',[dict(model=n,component=c,parameters=count,float32_bytes=4*count)
              for n,m in report['models'].items() for c,count in m['sections'].items()])
    write_markdown(report,output)
    make_diagram(report,output)
    print('Report written to',output,flush=True)


if __name__=='__main__':
    main()
