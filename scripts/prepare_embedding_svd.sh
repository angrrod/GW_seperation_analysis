#!/bin/bash
#SBATCH --job-name=embedding_svd
#SBATCH --account=lp_biolearning
#SBATCH --clusters=mindwell
#SBATCH --partition=batch_graniterapids
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=HPC_output_logs/EmbeddingSVD_%j.out
#SBATCH --error=HPC_output_logs/EmbeddingSVD_%j.err

set -euo pipefail
module --force purge
source "$VSC_DATA/miniconda3/etc/profile.d/conda.sh"
conda activate "${GW_CONDA_ENV:-$VSC_DATA/GW_separation/GW_env}"
export GW_INP_DIR="${GW_INP_DIR:-$VSC_DATA/GW_separation}"
export PYTHONPATH="$GW_INP_DIR:$GW_INP_DIR/External/dingo:${PYTHONPATH:-}"
# Paths and checkpoint locations are resolved by utils/paths.py on native scratch.
export GW_BLAS_THREADS="${GW_BLAS_THREADS:-1}"
export OPENBLAS_NUM_THREADS="$GW_BLAS_THREADS"
export MKL_NUM_THREADS="$GW_BLAS_THREADS"
export OMP_NUM_THREADS="$GW_BLAS_THREADS"
cd "$GW_INP_DIR"
# Slurm may increase CPUs to satisfy --mem, leaving the original cpu TRES value.
# Let srun use the effective SLURM_CPUS_PER_TASK allocation consistently.
unset SLURM_TRES_PER_TASK
echo "Embedding preparation CPUs per task: ${SLURM_CPUS_PER_TASK:-unknown}"
srun python -u scripts/prepare_embedding_svd.py --blas-threads "$GW_BLAS_THREADS" "$@"
