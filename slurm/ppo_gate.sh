#!/usr/bin/env bash
# P9 gate, step 2: PPO from the Prajna BC checkpoint with the real reader in the loop
# (configs/experiments/ppo_real.yaml, reward R1, 300 iterations x 128 episodes). One array task
# per arm:
#   0 strict  actor lr 3e-4 (report default)     1 strict  actor lr 1e-5 (Track A: small steps)
#   2 relaxed actor lr 3e-4                      3 relaxed actor lr 1e-5
# The gate is read from the in-run dev curve (metrics.jsonl: iteration 0 = the BC start, then
# every 25 iterations, same reader and evaluator throughout): PPO's final dev EM >= its
# iteration-0 EM, and the best iteration is not 0. policy.pt is the best dev checkpoint, which
# includes iteration 0, so its re-score (p-<run>-dev, done at the end of each task) can never fall
# below BC - it is a secondary number, not the gate.
#   sbatch --array=0-3%4 slurm/ppo_gate.sh       (after BOTH bc_bge.sh tasks have finished; l40
#                                                 allows 5 submitted jobs per user, so do not
#                                                 queue it behind bc_bge.sh with -d)
#   sbatch --array=0 --export=ALL,SMOKE=1 slurm/ppo_gate.sh   (smoke: 2 iterations on 50 questions,
#                                                             from the smoke BC checkpoint)
# Resubmitting the same task continues from its last.pt (e.g. after the time limit).
#SBATCH --job-name=ppo-gate
#SBATCH --account=cs769_02
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/ppo_gate_%A_%a.out
#SBATCH --error=slurm/logs/ppo_gate_%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/gpu_env.sh   # venv, HF_HUB_OFFLINE=1, Python headers for Triton + a C build check
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
DATA="$HOME/mhtable-data/v4"
RUNS="$HOME/mhtable-runs"
ARMS=(
  "strict|lr3e-4|algo.lr_actor=3.0e-4"
  "strict|lr1e-5|algo.lr_actor=1.0e-5"
  "relaxed|lr3e-4|algo.lr_actor=3.0e-4"
  "relaxed|lr1e-5|algo.lr_actor=1.0e-5"
)
IFS='|' read -r TIER TAG OV <<< "${ARMS[$SLURM_ARRAY_TASK_ID]}"
SMOKE="${SMOKE:-0}"
SUFFIX=""; TRAIN_X=(); EVAL_X=()
if [ "$SMOKE" = 1 ]; then
  SUFFIX="-smoke"; TRAIN_X=(--limit 50 train.iterations=2 train.eval_every=1 train.questions_per_iter=16)
  EVAL_X=(--limit 50)
fi
INIT="$RUNS/bc-bge-$TIER-s0$SUFFIX/policy.pt"
RUN_ID="ppo-r1-$TIER-$TAG-s0$SUFFIX"
[ -f "$INIT" ] || { echo "no BC checkpoint at $INIT - run slurm/bc_bge.sh first"; exit 1; }
RESUME=(); [ -f "$RUNS/$RUN_ID/last.pt" ] && RESUME=(--resume)
echo "job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID on $(hostname): $RUN_ID | init $INIT | $OV ${RESUME[*]}"

if [ -f "$RUNS/$RUN_ID/result.json" ]; then
  echo "training already finished ($RUNS/$RUN_ID/result.json); only evaluating"
else
  python -m mhtable.train --config configs/experiments/ppo_real.yaml --data-dir "$DATA" --out "$RUNS" \
    --run-id "$RUN_ID" "${RESUME[@]}" "${TRAIN_X[@]}" env.tier="$TIER" reward.variant=R1 \
    train.init="$INIT" $OV
fi

python -m mhtable.evaluate --config configs/experiments/eval.yaml --selector policy \
  --policy "$RUNS/$RUN_ID/policy.pt" --split dev --tiers "$TIER" --chunk 128 --reported \
  --data-dir "$DATA" --out "$RUNS" --run-id "p-$RUN_ID-dev" "${EVAL_X[@]}"
