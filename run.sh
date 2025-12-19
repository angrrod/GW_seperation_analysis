#!/bin/bash
#SBATCH --job-name=TestJob
#SBATCH --account=intro_vsc37106
#SBATCH --clusters=genius         # Specify the target cluster; adjust if needed
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=66:00:00
#SBATCH --output=HPC_output_logs/TestJob_%j.out
#SBATCH --error=HPC_output_logs/TestJob_%j.err

# Optionally clear any loaded modules
module --force purge

# Source the conda initialization script so that 'conda activate' works
source $VSC_DATA/miniconda3/etc/profile.d/conda.sh

# Activate the custom conda environment
conda activate ThesisEnv

# Make sure HPC can access the API key
export WANDB_API_KEY="a60c2702a6cb6c266aa5ebc8d6ba83520768fde2"

# Capture the first two command-line arguments as parameters
MODEL_TYPE=$1
NFOLDS=$2

# Run the Python script using srun and pass the parameters explicitly
srun python /vsc-hard-mounts/leuven-user/371/vsc37106/Thesis/sampeling.py /vsc-hard-mounts/leuven-user/371/vsc37106/Thesis/sampeling.py --ModelType "$MODEL_TYPE" --Nfolds "$NFOLDS"