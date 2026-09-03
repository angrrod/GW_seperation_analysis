#!/bin/bash
#SBATCH --job-name=TestJob
#SBATCH --account=lp_biolearning
#SBATCH --clusters=wice            # <-- run on wICE
#SBATCH --partition=gpu_a100 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G

#SBATCH --time=3:00:00
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

cd "$VSC_DATA/GW_separation"

export GW_LOCAL_CACHE="${VSC_SCRATCH_NODE}/dingo_cache"
mkdir -p "$GW_LOCAL_CACHE"

export GW_USE_LOCAL_CACHE=1

echo "TMPDIR=$TMPDIR"
echo "GW_LOCAL_CACHE=$GW_LOCAL_CACHE"
df -h "$TMPDIR"

# Run the Python script using srun and pass the parameters explicitly
srun python /vsc-hard-mounts/leuven-user/371/vsc37106/GW_separation/Debug_dingo.py