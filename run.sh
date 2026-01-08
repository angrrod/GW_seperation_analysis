#!/bin/bash
#SBATCH --job-name=TestJob
#SBATCH --account=intro_vsc37106
#SBATCH --clusters=genius         # Specify the target cluster; adjust if needed
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --output=HPC_output_logs/TestJob_%j.out
#SBATCH --error=HPC_output_logs/TestJob_%j.err

# Optionally clear any loaded modules
module --force purge

#lower than cpu per tasks
export GW_NPOOL=6  
export GW_INP_DIR="$VSC_DATA/GW_separation"

# Source the conda initialization script so that 'conda activate' works
source $VSC_DATA/GW_separation/miniconda3/bin/activate

# Activate the custom conda environment
conda activate "$PWD/GW_separation/GW_env"

# Capture the first two command-line arguments as parameters
MODEL_TYPE=$1

# Run the Python script using srun and pass the parameters explicitly
srun python /vsc-hard-mounts/leuven-user/371/vsc37106/GW_separation/main.py /vsc-hard-mounts/leuven-user/371/vsc37106/GW_separation/main.py --run-sampler --method "$MODEL_TYPE"