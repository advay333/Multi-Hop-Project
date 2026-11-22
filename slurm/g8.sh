#!/usr/bin/env bash
# Gate G-8, round 3: PPO from scratch must overfit 20 training questions (mock reader, hash
# encoder, data-v4 Spider+MMQA). CPU only — no GPU requested. One array task = one arm.
#   sbatch --array=0-4%4 slurm/g8.sh                                  (R1 arms, on l40)
#   sbatch --partition=a40 --qos=a40 --array=5-7%3 slurm/g8.sh       (R0 arms, on a40)
#   sbatch --array=0 --export=ALL,ITERS=5 slurm/g8.sh                 (smoke test, ~5 min)
#SBATCH --job-name=g8
#SBATCH --account=cs769_02
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/g8_%A_%a.out
#SBATCH --error=slurm/logs/g8_%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$HOME/mhtable-venv/bin/activate"
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
DATA="$HOME/mhtable-data/v4"
ITERS="${ITERS:-200}"

FT="algo.freeze_temperature=true"
ENT1="algo.entropy_coef=0.01 algo.entropy_coef_final=0.01"
ENT5="algo.entropy_coef=0.05 algo.entropy_coef_final=0.05"
KL="algo.target_kl=0.02"
ARMS=(
  "base-r1|reward.variant=R1"                                                      # 0 defaults (reference)
  "E-r1|reward.variant=R1 $FT $ENT5"                                               # 1 best of round 1 (0.90 on data-v2)
  "F-r1|reward.variant=R1 $FT $ENT1 $KL algo.lr_actor=1.0e-5"                      # 2 small actor lr
  "G-r1|reward.variant=R1 $FT $ENT1 algo.adam_eps=1.0e-4"                          # 3 large Adam eps
  "H-r1|reward.variant=R1 $FT $ENT1 algo.lr_actor=3.0e-5 algo.adam_eps=1.0e-5"     # 4 both, milder
  "E-r0|reward.variant=R0 $FT $ENT5"                                               # 5
  "F-r0|reward.variant=R0 $FT $ENT1 $KL algo.lr_actor=1.0e-5"                      # 6
  "G-r0|reward.variant=R0 $FT $ENT1 algo.adam_eps=1.0e-4"                          # 7
)
ARM="${ARMS[$SLURM_ARRAY_TASK_ID]}"
NAME="g8v4-${ARM%%|*}"
OV="${ARM#*|}"
if [ "$ITERS" != 200 ]; then NAME="$NAME-smoke$ITERS"; fi
echo "job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID on $(hostname): $NAME | $OV"

python -m mhtable.train --config configs/experiments/ppo.yaml --data-dir "$DATA" --out "$HOME/mhtable-runs" \
  --run-id "$NAME" data.revision=data-v4 train.questions=20 train.reachable_only=true \
  train.iterations="$ITERS" train.eval_on=train $OV
