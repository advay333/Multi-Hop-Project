#!/usr/bin/env bash
# GPU smoke test of the real reader on Prajna (after setup_reader.sh and fetch_models.sh):
# the oracle selector + Qwen2.5-7B (vLLM, bf16) on 100 dev questions per tier. Expected (P4 gate
# / Track C on molab): oracle EM about 50%. History 2026-11-22: run 1-2 failed on the CUDA 13
# vLLM wheel (fixed in setup_reader.sh), run 3 on Triton's C build (missing Python.h, fixed by
# gpu_env.sh + headers from setup_reader.sh). Compute nodes have no internet (confirmed).
#   mkdir -p slurm/logs && sbatch slurm/reader_smoke.sh
#SBATCH --job-name=reader-smoke
#SBATCH --account=cs769_02
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/reader_smoke_%J.out
#SBATCH --error=slurm/logs/reader_smoke_%J.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "== node $(hostname)"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
source slurm/gpu_env.sh   # venv, HF_HUB_OFFLINE=1, Python headers for Triton + a C build check
python -c "import torch, vllm; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu ok', torch.cuda.is_available(), '| vllm', vllm.__version__)"

python -m mhtable.evaluate --config configs/experiments/eval.yaml --selector oracle --limit 100 \
  --data-dir "$HOME/mhtable-data/v4" --out "$HOME/mhtable-runs" --run-id reader-smoke-oracle-dev100
