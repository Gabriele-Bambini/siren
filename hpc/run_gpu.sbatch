#!/bin/bash
#SBATCH --job-name=siren_gpu
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
# Usage: sbatch -J <name> hpc/run_gpu.sbatch scripts/v2_e3_enformer.py [args...]
set -eu
export SIREN_ROOT="${SIREN_ROOT:-$HOME/siren_repro}"
export PYTHONHASHSEED=0
# shellcheck disable=SC1091
source "${SIREN_VENV:-$HOME/venvs/siren}/bin/activate"
cd "$SIREN_ROOT"
mkdir -p logs results/v2
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python "$@"
