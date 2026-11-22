#!/usr/bin/env bash
# P9 ablation: step-level ("process") GRPO, R1, strict and relaxed, seed 0
# (configs/experiments/grpo_process_real.yaml). It is separate from slurm/p9_grid.sh so the main grid
# is untouched. Same BC start, reader, LR and 128 episodes per iteration as the grid.
#   0 grpo-process R1 strict      1 grpo-process R1 relaxed
# Each task trains (skipped if result.json exists; resumes from last.pt), then scores its best-dev
# checkpoint on the TEST split -> t-<run>-test/result.json.
#   LR and LAM must be given (LAM < 1, or the shaping cancels - see algos/grpo.py):
#   sbatch --partition=a40 --qos=a40 --export=ALL,LR=1e-5,LAM=0.5 --array=0-1 slurm/p9_process.sh
#   smoke:  sbatch --export=ALL,LR=1e-5,LAM=0.5,SMOKE=1 --array=0 slurm/p9_process.sh
#           (needs the smoke BC checkpoint from bc_bge.sh SMOKE=1)
#SBATCH --job-name=p9-process
#SBATCH --account=cs769_02
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/p9_process_%A_%a.out
#SBATCH --error=slurm/logs/p9_process_%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/gpu_env.sh   # venv, HF_HUB_OFFLINE=1, Python headers for Triton + a C build check
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
DATA="$HOME/mhtable-data/v4"
RUNS="$HOME/mhtable-runs"
case "${LR:-}" in   # YAML reads "1e-5" as a string, so the override needs the 1.0e-5 form
  1e-5) LRV=1.0e-5 ;;
  3e-5) LRV=3.0e-5 ;;
  1e-4) LRV=1.0e-4 ;;
  3e-4) LRV=3.0e-4 ;;
  *) echo "set LR (1e-5 | 3e-5 | 1e-4 | 3e-4) with --export=ALL,LR=..."; exit 1 ;;
esac
case "${LAM:-}" in
  0|0.0) LAMV=0.0 ;;
  0.5) LAMV=0.5 ;;
  0.8) LAMV=0.8 ;;
  0.95) LAMV=0.95 ;;
  *) echo "set LAM (0 | 0.5 | 0.8 | 0.95) with --export=ALL,LAM=..."; exit 1 ;;
esac
TIERS=(strict relaxed)
TIER="${TIERS[$SLURM_ARRAY_TASK_ID]}"
SMOKE="${SMOKE:-0}"
SUFFIX=""; TRAIN_X=(); EVAL_X=()
if [ "$SMOKE" = 1 ]; then
  SUFFIX="-smoke"; EVAL_X=(--limit 50)
  TRAIN_X=(--limit 50 train.iterations=2 train.eval_every=1 train.questions_per_iter=2)
fi
RUN_ID="grpoproc-r1-$TIER-lr$LR-lam$LAMV-s0$SUFFIX"
INIT="$RUNS/bc-bge-$TIER-s0$SUFFIX/policy.pt"
[ -f "$INIT" ] || { echo "no BC checkpoint at $INIT - run slurm/bc_bge.sh first"; exit 1; }
RESUME=(); [ -f "$RUNS/$RUN_ID/last.pt" ] && RESUME=(--resume)
echo "job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID on $(hostname): $RUN_ID | init $INIT | lr $LRV lam $LAMV ${RESUME[*]}"

if [ -f "$RUNS/$RUN_ID/result.json" ]; then
  echo "training already finished ($RUNS/$RUN_ID/result.json); only evaluating on test"
else
  python -m mhtable.train --config configs/experiments/grpo_process_real.yaml --data-dir "$DATA" \
    --out "$RUNS" --run-id "$RUN_ID" "${RESUME[@]}" "${TRAIN_X[@]}" env.tier="$TIER" reward.variant=R1 \
    train.init="$INIT" algo.lr_actor="$LRV" algo.process_lam="$LAMV"
fi

python -m mhtable.evaluate --config configs/experiments/eval.yaml --selector policy \
  --policy "$RUNS/$RUN_ID/policy.pt" --split test --tiers "$TIER" --chunk 128 --reported \
  --data-dir "$DATA" --out "$RUNS" --run-id "t-$RUN_ID-test" "${EVAL_X[@]}"
