#!/usr/bin/env bash
# Download data-v4 (Spider + MMQA only, no BIRD) on the Prajna LOGIN node — compute nodes have no
# internet. Needs a HuggingFace token: run  hf auth login  once first (stored in ~/.cache, never
# in the repo).
#   bash slurm/fetch_data.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source "$HOME/mhtable-venv/bin/activate"
OUT="$HOME/mhtable-data/v4"
python -m mhtable.storage download data "$OUT" --revision data-v4 --include \
  "db/spider/*" "db/mmqa/*" "gold/spider.jsonl" "gold/mmqa.jsonl" \
  "instances/spider.jsonl" "instances/mmqa.jsonl" "schemas/spider.json" "schemas/mmqa.json" \
  "splits/spider_mmqa/*" "index/spider/*" "index/mmqa/*"
du -sh "$OUT"; ls "$OUT"
