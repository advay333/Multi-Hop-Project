#!/usr/bin/env bash
# One-time setup on the Prajna LOGIN node (installing packages is allowed there; compute is not).
#   bash slurm/setup_env.sh
# Creates the project venv at ~/mhtable-venv with a CUDA 12.8 torch build (Prajna's driver is
# 570 / CUDA 12.8 — a cu13 wheel would not run) and the project installed in editable mode.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=""
for c in python3.12 python3.11 python3.10; do
  if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  echo "No python >= 3.10 found. Try:  source /lustre-flash/apps/spack/share/spack/setup-env.sh"
  echo "then  spack find python  and  spack load python@3.11  (or similar), and re-run this script."
  exit 1
fi
echo "using $($PY --version) at $(command -v $PY)"

VENV="$HOME/mhtable-venv"
[ -d "$VENV" ] || "$PY" -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[dev,train]"
python -c "import torch, mhtable; print('torch', torch.__version__, 'cuda build', torch.version.cuda); print('mhtable ok')"
