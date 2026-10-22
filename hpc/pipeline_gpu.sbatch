#!/bin/bash
#SBATCH --job-name=siren_gpu_pipeline
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
# Enformer branch: fidelity control (A8) -> scoring -> analysis (analysis forces INCONCLUSIVE if the control fails).
set -eu
export SIREN_ROOT="${SIREN_ROOT:-$HOME/siren_repro}"
export PYTHONHASHSEED=0
# shellcheck disable=SC1091
source "${SIREN_VENV:-$HOME/venvs/siren}/bin/activate"
cd "$SIREN_ROOT"
mkdir -p logs results/v2
pip freeze > "results/v2/env_freeze_gpu.txt" 2>/dev/null || true
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
STATUS=results/v2/pipeline_gpu_status.tsv
printf "script\texit_code\tseconds\n" > "$STATUS"
for s in v2_e3_sanity v2_e3_enformer v2_e3_analysis; do
  t0=$(date +%s)
  set +e
  python "scripts/$s.py" > "logs/$s.log" 2>&1
  rc=$?
  set -e
  printf "%s\t%s\t%s\n" "$s" "$rc" "$(( $(date +%s) - t0 ))" >> "$STATUS"
  if [ "$rc" -ne 0 ]; then echo "STOP: $s failed (rc=$rc)"; break; fi
done
cat "$STATUS"
