#!/usr/bin/env bash
# One-time: add the real reader (vLLM) and the encoder stack to ~/mhtable-venv, on the Prajna
# LOGIN node (after slurm/setup_env.sh).
#   bash slurm/setup_reader.sh
# Prajna's driver is 570 (CUDA 12.8). A CUDA 12.x build of vLLM/torch runs on it (CUDA minor-
# version compatibility); a CUDA 13 build does not. vLLM's default wheels have moved between
# cu128 / cu129 / cu130 over releases, so this asks uv for the cu128 torch backend, falls back to
# cu129, and refuses to finish if torch is not a CUDA 12 build.
set -euo pipefail
cd "$(dirname "$0")/.."
source "$HOME/mhtable-venv/bin/activate"
pip install -q uv

ok=""
for backend in cu128 cu129; do
  echo "== trying vllm with torch backend $backend"
  if uv pip install vllm --torch-backend="$backend"; then ok="$backend"; break; fi
done
[ -n "$ok" ] || { echo "vllm install failed for cu128 and cu129 — paste the error to the agent"; exit 1; }

pip install -e ".[dev,train,p4]"   # transformers, sentence-transformers (after vllm, so its torch pin stays)

python - <<'EOF'
import sys, torch
cuda = torch.version.cuda or "none"
try:
    import vllm
    v = vllm.__version__
except Exception as e:  # noqa: BLE001
    v = f"import failed: {e}"
print(f"torch {torch.__version__} (CUDA build {cuda}) | vllm {v}")
if not cuda.startswith("12."):
    sys.exit(f"torch is a CUDA {cuda} build; Prajna's driver (570) needs CUDA 12.x — tell the agent")
print("OK — note the vllm version above: every P9 run must use this same version")
EOF
