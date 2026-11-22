#!/usr/bin/env bash
# Download the reader and encoder weights and the frozen bge-m3 embedding cache on the Prajna
# LOGIN node (compute nodes have no internet). Needs `hf auth login` done once.
#   bash slurm/fetch_models.sh
# Everything lands in the HuggingFace cache under ~/.cache/huggingface (on /home, no quota);
# jobs then run with HF_HUB_OFFLINE=1 and find it there. About 15 GB (Qwen) + 2 GB (bge-m3) +
# the embedding cache.
set -euo pipefail
cd "$(dirname "$0")/.."
source "$HOME/mhtable-venv/bin/activate"
hf download Qwen/Qwen2.5-7B-Instruct
hf download BAAI/bge-m3
python -c "from mhtable import storage; print(storage.fetch('cache', 'embeddings/bge-m3/spider_mmqa.npz'))"
du -sh "$HOME/.cache/huggingface" 2>/dev/null || true
