#!/bin/bash
#SBATCH --job-name=TestJob
#SBATCH --account=lp_biolearning
#SBATCH --clusters=genius            # <-- run on wice
#SBATCH --partition=gpu_v100 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

#SBATCH --time=24:00:00
#SBATCH --output=HPC_output_logs/TestJob_%j.out
#SBATCH --error=HPC_output_logs/TestJob_%j.err

# Optionally clear any loaded modules
module --force purge

# Source the conda initialization script so that 'conda activate' works
source "$VSC_DATA/miniconda3/etc/profile.d/conda.sh"

# Activate the custom conda environment
conda activate "$VSC_DATA/GW_separation/GW_env"

#lower than cpu per tasks
export GW_NPOOL=4  
export GW_INP_DIR="$VSC_DATA/GW_separation"
export GW_DATA_DIR="$VSC_SCRATCH/GW_separation/data"

# Keep W&B's local run files away from the home-directory quota.
export WANDB_DIR="$VSC_SCRATCH/GW_separation/wandb"
mkdir -p "$GW_DATA_DIR" "$WANDB_DIR"

# Default: sync metrics to W&B during the job.
export WANDB_MODE="online"

cd "$VSC_DATA/GW_separation"

# Run the Python script using srun and pass the parameters explicitly
srun python /vsc-hard-mounts/leuven-user/371/vsc37106/GW_separation/ML_Test_true_posterior.py