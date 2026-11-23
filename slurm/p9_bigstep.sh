#!/usr/bin/env bash
# P9 follow-up: do the RL arms gain more when allowed to move further from BC? (PLAN P9)
# The lr 1e-5 grid barely moved: KL to BC 0.01-0.12 after 300 iterations, 74-89% of test questions
# solved with BC's actions. lr 3e-4 overshot: approx. KL 0.3-0.6 per update. Policy entropy was
# ~0.07-0.1 nats, and GRPO had 56-87% zero-advantage groups (all 8 rollouts identical).
#   big   = lr 1e-4, target_kl 0.02 (each update stops its epochs once approx. KL > 0.03), KL-to-BC 0.005
#   bigx  = big + entropy bonus 0.05 -> 0.005 (more exploration; more non-identical rollouts per group)
# Reward R1, seed 0, same BC start, reader, 300 iterations x 128 episodes as the grid.
#   0 ppo  big  strict     1 ppo  big  relaxed
#   2 grpo big  strict     3 grpo big  relaxed
#   4 ppo  bigx strict     5 grpo bigx strict
# Each task trains (skipped if result.json exists; resumes from last.pt), then scores its best-dev
# checkpoint on the TEST split -> t-<run>-test/result.json.
#   l40 (4 running / 5 submitted):  sbatch --array=0-3%4 slurm/p9_bigstep.sh
#   a40 (3 / 6):                    sbatch --partition=a40 --qos=a40 --array=4-5 slurm/p9_bigstep.sh
#   smoke:                          sbatch --export=ALL,SMOKE=1 --array=2 slurm/p9_bigstep.sh
#SBATCH --job-name=p9-big
#SBATCH --account=cs769_02
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/p9_big_%A_%a.out
#SBATCH --error=slurm/logs/p9_big_%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/gpu_env.sh   # venv, HF_HUB_OFFLINE=1, Python headers for Triton + a C build check
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
DATA="$HOME/mhtable-data/v4"
RUNS="$HOME/mhtable-runs"
BIG=(algo.lr_actor=1.0e-4 algo.target_kl=0.02 algo.kl_bc_coef=0.005)
EXPLORE=(algo.entropy_coef=0.05 algo.entropy_coef_final=0.005)
ARMS=(
  "ppo|big|strict"  "ppo|big|relaxed"
  "grpo|big|strict" "grpo|big|relaxed"
  "ppo|bigx|strict" "grpo|bigx|strict"
)
IFS='|' read -r ALGO SET TIER <<< "${ARMS[$SLURM_ARRAY_TASK_ID]}"
OV=("${BIG[@]}"); [ "$SET" = bigx ] && OV+=("${EXPLORE[@]}")
SMOKE="${SMOKE:-0}"
SUFFIX=""; TRAIN_X=(); EVAL_X=()
if [ "$SMOKE" = 1 ]; then
  SUFFIX="-smoke"; EVAL_X=(--limit 50)
  QPI=16; [ "$ALGO" = grpo ] && QPI=2   # 16 episodes per iteration either way (GRPO: 2 x 8)
  TRAIN_X=(--limit 50 train.iterations=2 train.eval_every=1 train.questions_per_iter=$QPI)
fi
RUN_ID="$ALGO-r1-$TIER-$SET-s0$SUFFIX"   # e.g. ppo-r1-strict-big-s0
INIT="$RUNS/bc-bge-$TIER-s0$SUFFIX/policy.pt"
[ -f "$INIT" ] || { echo "no BC checkpoint at $INIT - run slurm/bc_bge.sh first"; exit 1; }
RESUME=(); [ -f "$RUNS/$RUN_ID/last.pt" ] && RESUME=(--resume)
echo "job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID on $(hostname): $RUN_ID | init $INIT | ${OV[*]} ${RESUME[*]}"

if [ -f "$RUNS/$RUN_ID/result.json" ]; then
  echo "training already finished ($RUNS/$RUN_ID/result.json); only evaluating on test"
else
  python -m mhtable.train --config "configs/experiments/${ALGO}_real.yaml" --data-dir "$DATA" --out "$RUNS" \
    --run-id "$RUN_ID" "${RESUME[@]}" "${TRAIN_X[@]}" env.tier="$TIER" reward.variant=R1 \
    train.init="$INIT" "${OV[@]}"
fi

python -m mhtable.evaluate --config configs/experiments/eval.yaml --selector policy \
  --policy "$RUNS/$RUN_ID/policy.pt" --split test --tiers "$TIER" --chunk 128 --reported \
  --data-dir "$DATA" --out "$RUNS" --run-id "t-$RUN_ID-test" "${EVAL_X[@]}"
