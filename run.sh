#!/bin/bash
#SBATCH --job-name=TestJob
#SBATCH --account=lp_biolearning
#SBATCH --clusters=wice            # <-- run on wICE
#SBATCH --partition=batch_icelake  # optional but recommended (adjust if you use another partition)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=48:00:00
#SBATCH --output=HPC_output_logs/TestJob_%j.out
#SBATCH --error=HPC_output_logs/TestJob_%j.err

# Optionally clear any loaded modules
module --force purge

# Source the conda initialization script so that 'conda activate' works
source "$VSC_DATA/miniconda3/etc/profile.d/conda.sh"

# Activate the custom conda environment
conda activate "$VSC_DATA/GW_separation/GW_env"

#lower than cpu per tasks
export GW_NPOOL=16  
export GW_INP_DIR="$VSC_DATA/GW_separation"

cd "$VSC_DATA/GW_separation"

# Run the Python script using srun and pass the parameters explicitly
srun python /vsc-hard-mounts/leuven-user/371/vsc37106/GW_separation/main.py "$@"