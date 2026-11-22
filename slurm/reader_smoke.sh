#!/usr/bin/env bash
# GPU smoke test of the real reader on Prajna (after setup_reader.sh and fetch_models.sh):
# the oracle selector + Qwen2.5-7B (vLLM, bf16) on 100 dev questions per tier. Expected (P4 gate
# / Track C on molab): oracle EM about 50%. Also answers two open questions: does this vLLM build
# run on the CUDA 12.8 driver, and do compute nodes have internet.
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
source "$HOME/mhtable-venv/bin/activate"

echo "== node $(hostname)"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo "== internet from the compute node (HTTP code, or FAILED):"
curl -s -o /dev/null -w "%{http_code}\n" --max-time 15 https://huggingface.co || echo FAILED

export HF_HUB_OFFLINE=1
python -c "import torch, vllm; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu ok', torch.cuda.is_available(), '| vllm', vllm.__version__)"

python -m mhtable.evaluate --config configs/experiments/eval.yaml --selector oracle --limit 100 \
  --data-dir "$HOME/mhtable-data/v4" --out "$HOME/mhtable-runs" --run-id reader-smoke-oracle-dev100
