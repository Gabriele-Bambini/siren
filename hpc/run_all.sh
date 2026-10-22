#!/bin/bash
# Rebuild every v2 result from the raw benchmark files (PREREG_v2 goal G1).
# Usage (login node, after `bash hpc/setup_env.sh`):  bash hpc/run_all.sh
# Submits the CPU experiments in parallel, the GPU Enformer scoring job, and the dependent analysis jobs.
set -euo pipefail
export SIREN_ROOT="${SIREN_ROOT:-$HOME/siren_repro}"
cd "$SIREN_ROOT"
mkdir -p logs results/v2
sub() { sbatch --parsable "$@"; }
J_E1=$(sub -J siren_e1 hpc/run_cpu.sbatch scripts/v2_e1_loco.py)
J_E2=$(sub -J siren_e2 hpc/run_cpu.sbatch scripts/v2_e2_transfer.py)
J_E5=$(sub -J siren_e5 hpc/run_cpu.sbatch scripts/v2_e5_sign.py)
J_E4=$(sub -J siren_e4 hpc/run_cpu.sbatch scripts/v2_e4_conformal.py)
J_E6=$(sub -J siren_e6 hpc/run_cpu.sbatch scripts/v2_e6_zoo.py)
J_E3F=$(sub -J siren_e3f hpc/run_gpu.sbatch scripts/v2_e3_sanity.py)
J_E3S=$(sub -J siren_e3s --dependency=afterok:"$J_E3F" hpc/run_gpu.sbatch scripts/v2_e3_enformer.py)
J_E3A=$(sub -J siren_e3a --dependency=afterok:"$J_E3S" hpc/run_cpu.sbatch scripts/v2_e3_analysis.py)
echo "submitted: E1=$J_E1 E2=$J_E2 E4=$J_E4 E5=$J_E5 E6=$J_E6 E3sanity=$J_E3F E3score=$J_E3S E3analysis=$J_E3A"
echo "then:  python scripts/claims.py build render check"
