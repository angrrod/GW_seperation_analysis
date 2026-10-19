#!/bin/bash
#SBATCH --job-name=DINGO_Test
#SBATCH --account=lp_biolearning
#SBATCH --clusters=wice            # <-- run on wICE
#SBATCH --partition=gpu_a100 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G

#SBATCH --time=1:20:00
#SBATCH --output=HPC_output_logs/TestJob_%j.out
#SBATCH --error=HPC_output_logs/TestJob_%j.err

# Optionally clear any loaded modules
module --force purge

# Source the conda initialization script so that 'conda activate' works
source "$VSC_DATA/miniconda3/etc/profile.d/conda.sh"

# Activate the custom conda environment
conda activate "$VSC_DATA/GW_separation/GW_env"

#lower than cpu per tasks
export GW_INP_DIR="$VSC_DATA/GW_separation"
export GW_DATA_DIR="$VSC_SCRATCH/GW_separation/data"

export GW_PROFILE_CPU=0
export GW_CPU_PROFILE_INTERVAL=2
export GW_CPU_PROFILE_DIR="$SLURM_SUBMIT_DIR"

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
srun py-spy record \
    --subprocesses \
    --rate 10 \
    --duration 3000 \
    --output "$SLURM_SUBMIT_DIR/dingo_cpu_flame.svg" \
    -- python -u "$PROJECT_DIR/scripts/Debug_dingo.py" "$@"