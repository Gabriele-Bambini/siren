#!/bin/bash
# Re-run every experiment with the corrected (2026-10 audit) code, in dependency order.
#
#   cd <checkout of this repo on the cluster>      # code comes from here
#   export SIREN_HOME=~/siren_hpc                  # data/raw, data/processed, results
#   sbatch scripts/hpc_rerun_all.sbatch            # or: bash scripts/hpc_rerun_all.sbatch
#
# Before anything else the current results/ directory is copied to
# results_preaudit_<timestamp>/ (the scripts overwrite files with the same names).
# Each step is logged; a failing step does not stop the others and the summary at
# the end lists every exit code. Outputs to commit are collected in
# $SIREN_HOME/rerun_<timestamp>.tar.gz. See supplementary/AUDIT_TREE.md.
#
#SBATCH --job-name=siren-rerun
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=siren_rerun_%j.log

set -uo pipefail
export SIREN_HOME="${SIREN_HOME:-$HOME/siren_hpc}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_DIR" || exit 1
[ -f scripts/hpc_reference_eval.py ] || { echo "run from the repository checkout"; exit 1; }
PY="${PYTHON:-python3}"
TS="$(date +%Y%m%d_%H%M%S)"
RES="$SIREN_HOME/results"
declare -a SUMMARY=()

step() {  # step <name> <command...>
    local name="$1"; shift
    echo; echo "=== [$(date +%H:%M:%S)] $name"
    "$@"
    local rc=$?
    SUMMARY+=("$(printf '%-34s exit=%s' "$name" "$rc")")
    return 0
}

echo "repo=$REPO_DIR commit=$(git rev-parse --short HEAD 2>/dev/null || echo n/a) SIREN_HOME=$SIREN_HOME"
mkdir -p "$RES"
cp -r "$RES" "$SIREN_HOME/results_preaudit_$TS"
echo "backup: $SIREN_HOME/results_preaudit_$TS"

step "unit tests"                 "$PY" -m pytest tests -q -p no:cacheprovider
[ -d "$SIREN_HOME/data/processed" ] || step "preprocess" "$PY" scripts/hpc_preprocess.py

# 1. corrected statistics on the EXISTING W3b pairs (old, mis-targeted readout)
if [ -f "$RES/w3b_pairs.tsv.gz" ]; then
    step "W3b stats (old readout)"   "$PY" scripts/hpc_w3b_stats.py
    cp "$RES/w3b_summary.json" "$RES/w3b_summary_old_readout.json" 2>/dev/null
    cp "$RES/w3b_pairs.tsv.gz" "$RES/w3b_pairs_old_readout.tsv.gz" 2>/dev/null
fi

# 2. reference class (Table 1) and the experiments that feed it
step "reference eval (Table 1)"   "$PY" scripts/hpc_reference_eval.py
step "W2 gate"                    "$PY" scripts/hpc_w2_gate.py
step "W2b"                        "$PY" scripts/hpc_w2b_abc_baseline.py
step "W2c"                        "$PY" scripts/hpc_w2c_full_baseline.py
step "diag W7b"                   "$PY" scripts/hpc_diag_w7b.py
step "backtest + bench"           "$PY" scripts/hpc_backtest_bench.py
step "gridsearch"                 "$PY" scripts/hpc_gridsearch.py
step "W7 multi-dataset"           "$PY" scripts/hpc_w7_multids.py
step "W7c transfer"               "$PY" scripts/hpc_w7c_transfer.py

# 3. Enformer with the corrected readout (CPU, ~1-1.5 h)
step "W3 probe (Enformer)"        "$PY" scripts/hpc_w3_enformer_probe.py
step "W3 stats"                   "$PY" scripts/hpc_w3_stats.py
step "W3 audit"                   "$PY" scripts/hpc_w3_audit.py
step "W3b inference (Enformer)"   "$PY" scripts/hpc_w3b_enformer.py
step "W3b stats"                  "$PY" scripts/hpc_w3b_stats.py

# 4. figures (read the JSONs written above)
step "Fig 2-3"                    "$PY" scripts/hpc_figs23.py
step "Fig 1"                      "$PY" scripts/hpc_fig_transfer.py

# collect what should be committed to results/ (pair tables stay on the cluster)
OUT="$SIREN_HOME/rerun_$TS.tar.gz"
( cd "$RES" && tar czf "$OUT" --ignore-failed-read \
    reference_eval.json reference_eval.md w2_gate_results.json w2_gate_report.md \
    w2b_abc_results.json w2c_full_results.json w2c_full_report.md diag_w7b.json \
    bench_compare_results.json gridsearch_results.json w7_results.json w7c_results.json \
    w3_enformer_summary.json w3_audit.json w3b_summary.json w3b_summary_old_readout.json \
    w3b_provenance.json w3b_report.md fig2_precision_at_k.png fig2_precision_at_k.pdf \
    fig2_precision_curves.json fig3_baserate_trap.png fig3_baserate_trap.pdf \
    fig_transfer_matrix.png fig_transfer_matrix.pdf fig_transfer_matrix.json 2>/dev/null )

echo; echo "=== summary"
printf '%s\n' "${SUMMARY[@]}"
echo "outputs to commit: $OUT"
