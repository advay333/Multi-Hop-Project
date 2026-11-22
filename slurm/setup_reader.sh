#!/usr/bin/env bash
# One-time: add the real reader (vLLM) and the encoder stack to ~/mhtable-venv, on the Prajna
# LOGIN node (after slurm/setup_env.sh). Safe to re-run: it replaces any earlier vllm/torch.
#   bash slurm/setup_reader.sh
# Prajna's driver is 570 (CUDA 12.8). vLLM's default PyPI wheel is a CUDA 13 build; it imports on
# the login node but fails on a GPU node with "libcudart.so.13: cannot open shared object file"
# (reader_smoke, 2026-11-22). The fix is vLLM's CUDA 12.9 wheel from wheels.vllm.ai together
# with torch built for cu129. A 12.9 runtime runs on the 12.8 driver (CUDA minor-version
# compatibility, as long as the GPU's own kernels are compiled in: A40 = sm_86, L40S = sm_89).
# For vLLM 0.24.0 only cu129 and cu130 builds exist (no cu128; checked 2026-11-22).
# Every compared P9 run must use this same vLLM version.
set -euo pipefail
cd "$(dirname "$0")/.."
source "$HOME/mhtable-venv/bin/activate"
VLLM_VERSION="${VLLM_VERSION:-0.24.0}"
CU="${CU:-cu129}"
pip install -q uv

echo "== removing any earlier vllm / torch build"
uv pip uninstall vllm torch torchvision torchaudio || true

echo "== installing vllm ${VLLM_VERSION}+${CU} with torch for ${CU}"
uv pip install "vllm==${VLLM_VERSION}+${CU}" \
  --extra-index-url "https://wheels.vllm.ai/${VLLM_VERSION}/${CU}" \
  --torch-backend="${CU}" \
  --index-strategy unsafe-best-match

pip install -e ".[dev,train,p4]"   # transformers, sentence-transformers (after vllm, so its torch pin stays)

# Python C headers for Triton's runtime C build (the system lacks python3.11-devel and we have no
# root). uv downloads a standalone Python of the same minor version, which ships the headers;
# slurm/gpu_env.sh points gcc at them (C_INCLUDE_PATH) only when the system copy is missing.
PYV="$(python -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
echo "== Python $PYV headers into ~/mhtable-pyinclude"
uv python install "$PYV" --install-dir "$HOME/mhtable-python"
hdr="$(find "$HOME/mhtable-python" -path "*/include/python$PYV/Python.h" | head -1)"
[ -n "$hdr" ] || { echo "FAIL: no Python.h found under ~/mhtable-python - tell the agent"; exit 1; }
ln -sfn "$(dirname "$hdr")" "$HOME/mhtable-pyinclude"
ls -l "$HOME/mhtable-pyinclude"

python - <<'EOF'
import pathlib, sys, torch
cuda = torch.version.cuda or "none"
import vllm  # importing works on the login node even for a wrong build; the .so check below does not need a GPU
print(f"torch {torch.__version__} (CUDA build {cuda}) | vllm {vllm.__version__}")
bad = [p.name for p in pathlib.Path(vllm.__file__).parent.rglob("*.so") if b"libcudart.so.13" in p.read_bytes()]
if not cuda.startswith("12."):
    sys.exit(f"FAIL: torch is a CUDA {cuda} build; Prajna's driver (570) needs CUDA 12.x - tell the agent")
if bad:
    sys.exit(f"FAIL: vllm is still a CUDA 13 build (needs libcudart.so.13: {bad[:3]}) - tell the agent")
print("OK - vllm and torch are CUDA 12 builds. Note the versions above: every P9 run must use them.")
EOF
