#!/usr/bin/env bash
# P9 follow-up: coverage reward (PLAN P9). Does a denser, reader-free answer signal let RL beat BC?
#   R_ans = (1 - cov) * EM + cov * Suff        (rewards.py; Suff = the sub-table shows the rows
#                                               the answer needs, within budget)
# EM is right/wrong and also fails when the reader errs on a good sub-table; GRPO had 56-87% of
# groups with identical returns. Suff credits the policy's own job and separates near misses.
# Everything else = the big-step runs (slurm/p9_bigstep.sh: lr 1e-4, target_kl 0.02, KL-to-BC
# 0.005, R1, BC start, 300 iterations x 128 episodes, seed 0), so each cell's baseline is the
# matching `<algo>-r1-<tier>-big-s0` run (cov = 0). Dev selection and test scoring stay EM.
#   0 ppo  cov0.5 strict   1 ppo  cov0.5 relaxed   2 grpo cov0.5 strict   3 grpo cov0.5 relaxed
#   4 ppo  cov1   strict   5 ppo  cov1   relaxed   6 grpo cov1   strict   7 grpo cov1   relaxed
# cov 1 = the reader never enters the reward (pure "show the right rows" objective).
# Each task trains (skipped if result.json exists; resumes from last.pt), then scores its best-dev
# checkpoint on the TEST split -> t-<run>-test/result.json.
#   l40 (4 running / 5 submitted):  sbatch --array=0-3%4 slurm/p9_coverage.sh
#   a40 (3 / 6):                    sbatch --partition=a40 --qos=a40 --array=4-7%3 slurm/p9_coverage.sh
#   smoke:                          sbatch --export=ALL,SMOKE=1 --array=2 slurm/p9_coverage.sh
#SBATCH --job-name=p9-cov
#SBATCH --account=cs769_02
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/p9_cov_%A_%a.out
#SBATCH --error=slurm/logs/p9_cov_%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/gpu_env.sh   # venv, HF_HUB_OFFLINE=1, Python headers for Triton + a C build check
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
DATA="$HOME/mhtable-data/v4"
RUNS="$HOME/mhtable-runs"
BIG=(algo.lr_actor=1.0e-4 algo.target_kl=0.02 algo.kl_bc_coef=0.005)
ARMS=(
  "ppo|0.5|strict"  "ppo|0.5|relaxed"  "grpo|0.5|strict"  "grpo|0.5|relaxed"
  "ppo|1.0|strict"  "ppo|1.0|relaxed"  "grpo|1.0|strict"  "grpo|1.0|relaxed"
)
IFS='|' read -r ALGO COV TIER <<< "${ARMS[$SLURM_ARRAY_TASK_ID]}"
SMOKE="${SMOKE:-0}"
SUFFIX=""; TRAIN_X=(); EVAL_X=()
if [ "$SMOKE" = 1 ]; then
  SUFFIX="-smoke"; EVAL_X=(--limit 50)
  QPI=16; [ "$ALGO" = grpo ] && QPI=2   # 16 episodes per iteration either way (GRPO: 2 x 8)
  TRAIN_X=(--limit 50 train.iterations=2 train.eval_every=1 train.questions_per_iter=$QPI)
fi
COVTAG="cov${COV%.0}"                  # cov0.5 | cov1
RUN_ID="$ALGO-r1-$TIER-$COVTAG-s0$SUFFIX"   # e.g. grpo-r1-relaxed-cov0.5-s0
INIT="$RUNS/bc-bge-$TIER-s0$SUFFIX/policy.pt"
[ -f "$INIT" ] || { echo "no BC checkpoint at $INIT - run slurm/bc_bge.sh first"; exit 1; }
RESUME=(); [ -f "$RUNS/$RUN_ID/last.pt" ] && RESUME=(--resume)
echo "job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID on $(hostname): $RUN_ID | init $INIT | coverage $COV ${BIG[*]} ${RESUME[*]}"

if [ -f "$RUNS/$RUN_ID/result.json" ]; then
  echo "training already finished ($RUNS/$RUN_ID/result.json); only evaluating on test"
else
  python -m mhtable.train --config "configs/experiments/${ALGO}_real.yaml" --data-dir "$DATA" --out "$RUNS" \
    --run-id "$RUN_ID" "${RESUME[@]}" "${TRAIN_X[@]}" env.tier="$TIER" reward.variant=R1 \
    reward.coverage="$COV" train.init="$INIT" "${BIG[@]}"
fi

python -m mhtable.evaluate --config configs/experiments/eval.yaml --selector policy \
  --policy "$RUNS/$RUN_ID/policy.pt" --split test --tiers "$TIER" --chunk 128 --reported \
  --data-dir "$DATA" --out "$RUNS" --run-id "t-$RUN_ID-test" "${EVAL_X[@]}"
