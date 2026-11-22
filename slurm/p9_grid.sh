#!/usr/bin/env bash
# P9 grid: {PPO, GRPO} x {R1 shaped, R0 sparse} x {strict, relaxed}, seed 0 (user decision
# 2026-11-22: one seed for now). Every run starts from the Prajna BC checkpoint of its tier
# (slurm/bc_bge.sh) and has the real reader in the loop; all arms use the same actor learning
# rate LR (chosen from the gate, slurm/ppo_gate.sh) and spend 128 episodes per iteration.
#   0 ppo  R1 strict    1 ppo  R1 relaxed     <- the gate runs at this LR: not retrained, only
#   2 ppo  R0 strict    3 ppo  R0 relaxed        evaluated on test
#   4 grpo R1 strict    5 grpo R1 relaxed
#   6 grpo R0 strict    7 grpo R0 relaxed
# Each task trains (skipped if result.json exists; resumes from last.pt if present), then scores
# its best-dev checkpoint (policy.pt) on the TEST split of its tier -> t-<run>-test/result.json.
# Selecting on dev and reporting test is the P9 protocol.
#   LR must be given:  --export=ALL,LR=1e-5   (or LR=3e-4)
#   l40 (5 submitted / 4 running per user):  sbatch --export=ALL,LR=1e-5 --array=2-5%4 slurm/p9_grid.sh
#   a40 (6 / 3):  sbatch --partition=a40 --qos=a40 --export=ALL,LR=1e-5 --array=0,1,6,7%3 slurm/p9_grid.sh
#   smoke:        sbatch --export=ALL,LR=1e-5,SMOKE=1 --array=4 slurm/p9_grid.sh   (GRPO, 2 iterations;
#                 needs the smoke BC checkpoint from bc_bge.sh SMOKE=1)
#SBATCH --job-name=p9-grid
#SBATCH --account=cs769_02
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/p9_grid_%A_%a.out
#SBATCH --error=slurm/logs/p9_grid_%A_%a.out
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
ARMS=(
  "ppo|R1|strict"  "ppo|R1|relaxed"
  "ppo|R0|strict"  "ppo|R0|relaxed"
  "grpo|R1|strict" "grpo|R1|relaxed"
  "grpo|R0|strict" "grpo|R0|relaxed"
)
IFS='|' read -r ALGO REW TIER <<< "${ARMS[$SLURM_ARRAY_TASK_ID]}"
SMOKE="${SMOKE:-0}"
SUFFIX=""; TRAIN_X=(); EVAL_X=()
if [ "$SMOKE" = 1 ]; then
  SUFFIX="-smoke"; EVAL_X=(--limit 50)
  QPI=16; [ "$ALGO" = grpo ] && QPI=2   # 16 episodes per iteration either way (GRPO: 2 x 8)
  TRAIN_X=(--limit 50 train.iterations=2 train.eval_every=1 train.questions_per_iter=$QPI)
fi
RUN_ID="$ALGO-${REW,,}-$TIER-lr$LR-s0$SUFFIX"   # e.g. ppo-r1-strict-lr1e-5-s0 (= the gate run)
INIT="$RUNS/bc-bge-$TIER-s0$SUFFIX/policy.pt"
[ -f "$INIT" ] || { echo "no BC checkpoint at $INIT - run slurm/bc_bge.sh first"; exit 1; }
RESUME=(); [ -f "$RUNS/$RUN_ID/last.pt" ] && RESUME=(--resume)
echo "job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID on $(hostname): $RUN_ID | init $INIT | lr $LRV ${RESUME[*]}"

if [ -f "$RUNS/$RUN_ID/result.json" ]; then
  echo "training already finished ($RUNS/$RUN_ID/result.json); only evaluating on test"
else
  python -m mhtable.train --config "configs/experiments/${ALGO}_real.yaml" --data-dir "$DATA" --out "$RUNS" \
    --run-id "$RUN_ID" "${RESUME[@]}" "${TRAIN_X[@]}" env.tier="$TIER" reward.variant="$REW" \
    train.init="$INIT" algo.lr_actor="$LRV"
fi

python -m mhtable.evaluate --config configs/experiments/eval.yaml --selector policy \
  --policy "$RUNS/$RUN_ID/policy.pt" --split test --tiers "$TIER" --chunk 128 --reported \
  --data-dir "$DATA" --out "$RUNS" --run-id "t-$RUN_ID-test" "${EVAL_X[@]}"
