#!/bin/bash
#SBATCH --job-name=siren_cpu_pipeline
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
# CPU experiments E1, E2, E4, E5, E6 in sequence (QoS 'stud' allows only 2 running / 4 submitted jobs per user).
set -eu
export SIREN_ROOT="${SIREN_ROOT:-$HOME/siren_repro}"
export PYTHONHASHSEED=0
# shellcheck disable=SC1091
source "${SIREN_VENV:-$HOME/venvs/siren}/bin/activate"
cd "$SIREN_ROOT"
mkdir -p logs results/v2
pip freeze > "results/v2/env_freeze_cpu.txt" 2>/dev/null || true
STATUS=results/v2/pipeline_cpu_status.tsv
printf "script\texit_code\tseconds\n" > "$STATUS"
for s in v2_e5_sign v2_e1_loco v2_e4_conformal v2_e2_transfer v2_e6_zoo v2_e7_classifiers; do
  t0=$(date +%s)
  set +e
  python "scripts/$s.py" > "logs/$s.log" 2>&1
  rc=$?
  set -e
  printf "%s\t%s\t%s\n" "$s" "$rc" "$(( $(date +%s) - t0 ))" >> "$STATUS"
done
cat "$STATUS"
