#!/bin/bash
#SBATCH --job-name=siren_cpu
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
# Usage: sbatch -J <name> hpc/run_cpu.sbatch scripts/v2_e1_loco.py [args...]
set -eu
export SIREN_ROOT="${SIREN_ROOT:-$HOME/siren_repro}"
export PYTHONHASHSEED=0
# shellcheck disable=SC1091
source "${SIREN_VENV:-$HOME/venvs/siren}/bin/activate"
cd "$SIREN_ROOT"
mkdir -p logs results/v2
python "$@"
