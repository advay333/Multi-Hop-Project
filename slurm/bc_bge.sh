#!/usr/bin/env bash
# P9 gate, step 1: behaviour cloning with bge-m3 on Prajna, then its real-reader dev score.
# One array task per tier (0 = strict, 1 = relaxed); each task:
#   1. trains BC (configs/experiments/bc_bge.yaml; the dev check inside uses the mock reader and
#      join-path accuracy) -> ~/mhtable-runs/bc-bge-<tier>-s0/policy.pt
#   2. evaluates that checkpoint on the full dev tier with the real reader (Qwen2.5-7B, vLLM)
#      -> ~/mhtable-runs/p-bc-<tier>-dev/result.json   (molab Track C: EM 28.9% strict / 28.2% relaxed)
# The budget is counted with the reader's tokenizer, as in PPO and evaluation (Track C's BC used
# the 3.5-characters-per-token estimate).
#   mkdir -p slurm/logs && sbatch --array=0-1 slurm/bc_bge.sh
#   sbatch --array=0 --export=ALL,SMOKE=1 slurm/bc_bge.sh     (smoke: 1 epoch, 50 questions, ~10 min)
#SBATCH --job-name=bc-bge
#SBATCH --account=cs769_02
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/bc_bge_%A_%a.out
#SBATCH --error=slurm/logs/bc_bge_%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/gpu_env.sh   # venv, HF_HUB_OFFLINE=1, Python headers for Triton + a C build check
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
DATA="$HOME/mhtable-data/v4"
RUNS="$HOME/mhtable-runs"
TIERS=(strict relaxed)
TIER="${TIERS[$SLURM_ARRAY_TASK_ID]}"
SMOKE="${SMOKE:-0}"
BC_ID="bc-bge-$TIER-s0"; EVAL_ID="p-bc-$TIER-dev"
TRAIN_X=(); EVAL_X=()
if [ "$SMOKE" = 1 ]; then
  BC_ID="$BC_ID-smoke"; EVAL_ID="$EVAL_ID-smoke"
  TRAIN_X=(--limit 50 train.epochs=1 train.eval_every=1 train.eval_train_questions=20); EVAL_X=(--limit 50)
fi
echo "job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID on $(hostname): $BC_ID, then $EVAL_ID"

python -m mhtable.train --config configs/experiments/bc_bge.yaml --data-dir "$DATA" --out "$RUNS" \
  --run-id "$BC_ID" "${TRAIN_X[@]}" data.revision=data-v4 env.tier="$TIER" \
  env.tokenizer=Qwen/Qwen2.5-7B-Instruct

python -m mhtable.evaluate --config configs/experiments/eval.yaml --selector policy \
  --policy "$RUNS/$BC_ID/policy.pt" --split dev --tiers "$TIER" --chunk 128 --reported \
  --data-dir "$DATA" --out "$RUNS" --run-id "$EVAL_ID" "${EVAL_X[@]}"
