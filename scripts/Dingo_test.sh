#!/bin/bash
#SBATCH --job-name=DINGO_Test
#SBATCH --account=lp_biolearning
#SBATCH --clusters=mindwell
#SBATCH --partition=gpu_b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
# Use Mindwell's default host memory: 8000 MiB/core (192000 MiB for 24 cores).

#SBATCH --time=3:30:00
#SBATCH --output=HPC_output_logs/TestJob_%j.out
#SBATCH --error=HPC_output_logs/TestJob_%j.err

# Optionally clear any loaded modules
module --force purge

# Source the conda initialization script so that 'conda activate' works
source "$VSC_DATA/miniconda3/etc/profile.d/conda.sh"

# Activate the custom conda environment
conda activate "$VSC_DATA/GW_separation/GW_env"

# First run scripts/prepare_embedding_svd.sh with matching configs on Mindwell.
# GPU initialization loads its completed cache from utils/paths.py's scratch path.

#lower than cpu per tasks
export GW_INP_DIR="$VSC_DATA/GW_separation"

# Capture one PyTorch CPU/CUDA window; disable with --export=ALL,GW_PROFILE_GPU=0.
export GW_PROFILE_GPU="${GW_PROFILE_GPU:-1}"
export GW_GPU_PROFILE_DIR="${GW_GPU_PROFILE_DIR:-$VSC_SCRATCH/GW_separation/gpu_profiles/$SLURM_JOB_ID}"
export GW_GPU_PROFILE_WAIT="${GW_GPU_PROFILE_WAIT:-20}"

export GW_PROFILE_CPU=0
export GW_CPU_PROFILE_INTERVAL=2
export GW_CPU_PROFILE_DIR="$SLURM_SUBMIT_DIR"

# Default numerical-library threading
export GW_BLAS_THREADS="${GW_BLAS_THREADS:-1}"

export OPENBLAS_NUM_THREADS=$GW_BLAS_THREADS
export MKL_NUM_THREADS=$GW_BLAS_THREADS
export BLIS_NUM_THREADS=$GW_BLAS_THREADS
export OMP_NUM_THREADS=$GW_BLAS_THREADS

#wandb metrics
# Include DataLoader child processes in W&B metrics
export WANDB_X_STATS_TRACK_PROCESS_TREE=true

# Increase system metric sampling frequency
export WANDB_X_STATS_SAMPLING_INTERVAL=5

cd "$VSC_DATA/GW_separation"

export GW_LOCAL_CACHE="${VSC_SCRATCH_NODE}/dingo_cache"
mkdir -p "$GW_LOCAL_CACHE"

export GW_USE_LOCAL_CACHE=1

echo "TMPDIR=$TMPDIR"
echo "GW_LOCAL_CACHE=$GW_LOCAL_CACHE"
df -h "$TMPDIR"

PROJECT_DIR="/vsc-hard-mounts/leuven-user/371/vsc37106/GW_separation"

cd "$PROJECT_DIR" || exit 1

export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

cd "$VSC_DATA/GW_separation"

# Run the Python script using srun and pass the parameters explicitly
# srun python "$PROJECT_DIR/scripts/Debug_dingo.py"
srun python -u "$PROJECT_DIR/scripts/Debug_dingo.py" "$@"
