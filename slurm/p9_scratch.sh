#!/usr/bin/env bash
# P9 follow-up: PPO and GRPO from scratch (random policy, no BC start) on the full Spider+MMQA
# training pool with the real reader in the loop (PLAN P9). From BC, every arm stayed within ~2
# EM pts of BC, so the critic and shaping had little to do. From scratch the reward is sparse and
# exploration matters: that is where a learned critic (PPO) and the shaping (R1) should show.
# Settings = G-8 arm E, the best from-scratch setting: actor lr 3e-4, frozen softmax temperature,
# constant entropy bonus 0.05. 1000 iterations x 128 episodes; the gold-prefix curriculum is
# annealed over 300 iterations instead of 100.
#   0 ppo  R1 strict    1 ppo  R1 relaxed
#   2 ppo  R0 strict    3 ppo  R0 relaxed
#   4 grpo R1 strict    5 grpo R1 relaxed      (GRPO R0 = R1 exactly: not run)
# Each task trains (skipped if result.json exists; resumes from last.pt, e.g. after the time
# limit), then scores its best-dev checkpoint on the TEST split -> t-<run>-test/result.json.
#   l40 (4 running / 5 submitted):  sbatch --array=0-3%4 slurm/p9_scratch.sh
#   a40 (3 / 6):                    sbatch --partition=a40 --qos=a40 --array=4-5 slurm/p9_scratch.sh
#   smoke:                          sbatch --export=ALL,SMOKE=1 --array=4 slurm/p9_scratch.sh
#SBATCH --job-name=p9-scratch
#SBATCH --account=cs769_02
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/p9_scratch_%A_%a.out
#SBATCH --error=slurm/logs/p9_scratch_%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/gpu_env.sh   # venv, HF_HUB_OFFLINE=1, Python headers for Triton + a C build check
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
DATA="$HOME/mhtable-data/v4"
RUNS="$HOME/mhtable-runs"
OV=(train.init=null algo.lr_actor=3.0e-4 algo.freeze_temperature=true
    algo.entropy_coef=0.05 algo.entropy_coef_final=0.05
    train.iterations=1000 train.curriculum_iters=300)
ARMS=(
  "ppo|R1|strict"  "ppo|R1|relaxed"
  "ppo|R0|strict"  "ppo|R0|relaxed"
  "grpo|R1|strict" "grpo|R1|relaxed"
)
IFS='|' read -r ALGO REW TIER <<< "${ARMS[$SLURM_ARRAY_TASK_ID]}"
SMOKE="${SMOKE:-0}"
SUFFIX=""; LIMIT=(); TRAIN_X=(); EVAL_X=()
if [ "$SMOKE" = 1 ]; then
  SUFFIX="-smoke"; LIMIT=(--limit 50); EVAL_X=(--limit 50)
  QPI=16; [ "$ALGO" = grpo ] && QPI=2   # 16 episodes per iteration either way (GRPO: 2 x 8)
  TRAIN_X=(train.iterations=2 train.eval_every=1 train.questions_per_iter=$QPI)
fi
RUN_ID="$ALGO-${REW,,}-$TIER-scratch-s0$SUFFIX"   # e.g. ppo-r1-strict-scratch-s0
RESUME=(); [ -f "$RUNS/$RUN_ID/last.pt" ] && RESUME=(--resume)
echo "job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID on $(hostname): $RUN_ID | ${OV[*]} ${RESUME[*]}"

if [ -f "$RUNS/$RUN_ID/result.json" ]; then
  echo "training already finished ($RUNS/$RUN_ID/result.json); only evaluating on test"
else
  # options (--limit, --resume) before the key=value overrides (argparse); TRAIN_X after OV so the
  # smoke's train.iterations=2 wins
  python -m mhtable.train --config "configs/experiments/${ALGO}_real.yaml" --data-dir "$DATA" --out "$RUNS" \
    --run-id "$RUN_ID" "${RESUME[@]}" "${LIMIT[@]}" env.tier="$TIER" reward.variant="$REW" "${OV[@]}" "${TRAIN_X[@]}"
fi

python -m mhtable.evaluate --config configs/experiments/eval.yaml --selector policy \
  --policy "$RUNS/$RUN_ID/policy.pt" --split test --tiers "$TIER" --chunk 128 --reported \
  --data-dir "$DATA" --out "$RUNS" --run-id "t-$RUN_ID-test" "${EVAL_X[@]}"
