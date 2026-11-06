# CPU embedding SVD preparation

Run once before a new configuration's GPU training:

```bash
sbatch scripts/prepare_embedding_svd.sh
```

The default prepares both `training_full.yml` and `training_copula.yml`. Identical
preparation settings share a cache. To select one:

```bash
sbatch scripts/prepare_embedding_svd.sh --model full
```

After preparation finishes successfully, submit the existing training job:

```bash
sbatch scripts/Dingo_test.sh
```

The preparation job runs on Mindwell's CPU partition, without requesting a GPU.
The training wrapper requires a completed compatible cache and never silently
rebuilds it on the GPU allocation. Existing model-checkpoint resume is unchanged.

## Paths and configuration

Both processes use `utils/paths.py`. The cache root defaults to
`/gpfs1/scratch/371/vsc37106/GW_separation/data/embedding_svd` on Mindwell and
`/lustre1/scratch/371/vsc37106/GW_separation/data/embedding_svd` on wICE, or beneath
`GW_DATA_DIR` when that override is present. `GW_EMBEDDING_SVD_CACHE_DIR` overrides
only the cache root. `GW_WAVEFORM_DATASET_PATH` overrides the input waveform file.
Input waveforms and ASDs must already exist on the selected scratch filesystem.

Each preparation recipe has a separate hash-named directory. Its identity covers
data/preprocessing settings, SVD settings, initialization batch size and seed,
and input filenames, sizes, and modification times. Replacing an input file or
changing settings selects a new cache. This identifies inputs by filesystem
metadata, not by hashing the entire waveform bank. The recipe version must be
bumped if preparation semantics change. Old untagged per-job basis files and
the former `embedding_svd_V_rb_cache.pt` are not imported.

`GW_EMBEDDING_SVD_SEED` defaults to 42. Use the same environment overrides for
preparation and training. `GW_BLAS_THREADS` defaults to 1; this retains the prior
threading choice. Different threading/performance choices need cluster benchmarks.
`GW_CONDA_ENV` optionally selects the CPU preparation environment.

## Restart behavior

The same preparation command resumes its matching directory:

1. Save clean, projected, time-shifted, summed, whitened initialization examples
   without added noise as `.npy` files and labels. Publish the example manifest
   only after all files are complete. Restarts memory-map these exact examples.
2. Build each missing detector basis. Atomically save `svd_H1.hdf5` or
   `svd_L1.hdf5` with compatibility metadata before validation.
3. Validate bases whose validation is unfinished. Atomically replace their files
   with the validation results and completed status. A high mismatch is reported;
   it is not automatically rejected by a quality threshold.
4. Publish `ready.json` only after every detector has completed validation.

If interrupted during L1 construction, H1 is reused. If interrupted during
validation, construction is not repeated. Incomplete temporary writes are never
used as completed checkpoints. A file lock prevents simultaneous writers to the
same recipe. GPU training reads the completed bases in configured detector order
and drops bins below the domain's minimum frequency, just like the original builder.

Preparation retains the same CPU SVD solver and waveform transformations. Splitting
the job does not itself accelerate the SVD. Initialization examples occupy about
23.4 GiB on disk for two detectors, 12,000 rows, 65,537 bins, and complex128;
bases and validation metadata require additional scratch space.
