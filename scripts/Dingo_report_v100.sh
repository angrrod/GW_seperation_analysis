#!/bin/bash
#SBATCH --job-name=DINGO_Report
#SBATCH --account=lp_biolearning
#SBATCH --clusters=wice
#SBATCH --partition=gpu_v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=HPC_output_logs/PipelineReport_%j.out
#SBATCH --error=HPC_output_logs/PipelineReport_%j.err

# Submit from the project root (Slurm needs the log directory beforehand):
#   mkdir -p HPC_output_logs
#   sbatch scripts/Dingo_report_v100.sh
# If this shell script is in the main directory on VSC:
#   sbatch Dingo_report_v100.sh
# Keep the Python helper in scripts/report_dingo_pipeline.py.
# Optional arguments are passed to the report, e.g. --skip-profile or
# --profile /path/to/a/downloaded_trace.json.
# This generates a report using CPU probes and existing traces. It does not
# train a model or capture a new GPU profile, despite reserving a V100 node.

set -eo pipefail

module --force purge
source "$VSC_DATA/miniconda3/etc/profile.d/conda.sh"
conda activate "$VSC_DATA/GW_separation/GW_env"
set -u

# Code is in the user checkout; configs live in DATA and datasets in SCRATCH.
export GW_INP_DIR="${GW_INP_DIR:-$VSC_DATA/GW_separation}"
export GW_DATA_DIR="${GW_DATA_DIR:-$VSC_SCRATCH/GW_separation/data}"

# Prefer the checkout from which sbatch was submitted. An explicit override
# supports submitting elsewhere or keeping multiple checkouts on VSC.
if [[ -n "${GW_REPORT_PROJECT_DIR:-}" ]]; then
    PROJECT_DIR="$GW_REPORT_PROJECT_DIR"
elif [[ -f "$SLURM_SUBMIT_DIR/scripts/report_dingo_pipeline.py" ]]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
    PROJECT_DIR="$VSC_DATA/GW_separation"
fi

cd "$PROJECT_DIR"
if [[ ! -f scripts/report_dingo_pipeline.py ]]; then
    echo "Missing $PROJECT_DIR/scripts/report_dingo_pipeline.py" >&2
    echo "Transfer the Python helper or set GW_REPORT_PROJECT_DIR to the checkout." >&2
    exit 1
fi

# Use this checkout's Dingo and copula implementation with the installed deps.
export PYTHONPATH="$PROJECT_DIR/External/dingo:$PROJECT_DIR/vector_copula_vi_v2:$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# Avoid a stale CPU count inherited from a different allocation (e.g. 12 vs 4).
# The allocation above requests one GPU per node and four CPUs per task.
unset SLURM_TRES_PER_TASK

REPORT_DIR="$GW_INP_DIR/Output/HPC_diagnostics/pipeline_report_${SLURM_JOB_ID}"
mkdir -p "$REPORT_DIR"
echo "Project: $PROJECT_DIR"
echo "Configs: $GW_INP_DIR/configs"
echo "Datasets: $GW_DATA_DIR"
echo "Report output: $REPORT_DIR"
echo "CPU shape/parameter probes; no new GPU timing measurements."

srun python -u scripts/report_dingo_pipeline.py --output-dir "$REPORT_DIR" "$@"
