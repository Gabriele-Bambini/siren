#!/bin/bash
#SBATCH --job-name=siren_A
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
# Phase A: re-run the v1.0 experiment scripts unchanged (only SIREN_ROOT path made configurable)
# and record exit status per script, so "what works / what fails" is a machine-readable table.
set -u
export SIREN_ROOT="${SIREN_ROOT:-$HOME/siren_repro}"
export PYTHONHASHSEED=0
cd "$SIREN_ROOT" || exit 2
mkdir -p logs results
STATUS=results/phaseA_status.tsv
printf "script\texit_code\tseconds\n" > "$STATUS"
run() {
  local s="$1"; local t0=$(date +%s)
  echo "=== $s ($(date -Is)) ==="
  python3 "scripts/$s" > "logs/phaseA_${s%.py}.log" 2>&1
  local rc=$?
  printf "%s\t%s\t%s\n" "$s" "$rc" "$(( $(date +%s) - t0 ))" >> "$STATUS"
  echo "rc=$rc"
}
python3 -m pytest tests -q > logs/phaseA_pytest.log 2>&1; printf "pytest\t%s\t0\n" "$?" >> "$STATUS"
python3 scripts/demo_synthetic.py > logs/phaseA_demo.log 2>&1; printf "demo_synthetic\t%s\t0\n" "$?" >> "$STATUS"
for s in hpc_w2_gate.py hpc_w2b_abc_baseline.py hpc_w2c_full_baseline.py hpc_backtest_bench.py \
         hpc_w7_multids.py hpc_diag_w7b.py hpc_w7c_transfer.py hpc_gridsearch.py \
         hpc_s2_ranker.py hpc_s3_s7.py hpc_s4_twostage.py hpc_figs23.py hpc_fig_transfer.py; do
  run "$s"
done
echo "PHASE_A_DONE $(date -Is)"
