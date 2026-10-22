#!/bin/bash
# Create the pinned virtualenv used by every HPC job in this repository.
# Usage: bash hpc/setup_env.sh        (run once on the login node; ~4 GB download)
set -euo pipefail
VENV="${SIREN_VENV:-$HOME/venvs/siren}"
HPC_DIR="${SIREN_ROOT:-$HOME/siren_repro}/hpc"   # not $(dirname $0): SLURM runs a copy of this script
python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r "$HPC_DIR/requirements.lock.txt"
python - <<'PY'
import torch, lightgbm, sklearn, numpy, pandas
print("torch", torch.__version__, "cuda_build", torch.version.cuda, "numpy", numpy.__version__,
      "pandas", pandas.__version__, "sklearn", sklearn.__version__, "lightgbm", lightgbm.__version__)
PY
pip freeze > "$HPC_DIR/requirements.freeze.txt"
echo "ENV_READY $VENV"
