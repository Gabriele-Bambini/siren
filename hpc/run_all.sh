#!/bin/bash
# Rebuild every v2 result from the raw benchmark files (PREREG_v2 goal G1).
# Usage (login node, after `bash hpc/setup_env.sh`):  bash hpc/run_all.sh
# Two jobs only: QoS 'stud' allows 4 submitted / 2 running jobs per user.
set -euo pipefail
export SIREN_ROOT="${SIREN_ROOT:-$HOME/siren_repro}"
cd "$SIREN_ROOT"
mkdir -p logs results/v2
J_CPU=$(sbatch --parsable hpc/pipeline_cpu.sbatch)
J_GPU=$(sbatch --parsable hpc/pipeline_gpu.sbatch)
echo "submitted: cpu pipeline (E5,E1,E4,E2,E6,E7)=$J_CPU  gpu pipeline (E3 sanity,score,analysis,posthoc,bridge,random-shuffle sensitivity)=$J_GPU"
echo "when both finish:  python scripts/claims.py build render check   (results/v2/*.json -> claims -> manuscript)"
