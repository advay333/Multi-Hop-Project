#!/usr/bin/env bash
# P9 table, baseline rows: every selector scored on the TEST split with the Prajna reader
# (vLLM 0.24.0+cu129, Qwen2.5-7B p1 bf16), so they compare directly with the grid's t-*-test runs.
# One job, the evaluations run one after another (each loads vLLM again, ~1-2 min):
#   oracle (gold path: the ceiling), B1 truncate, B2 random_rows, heuristic -> both tiers each;
#   BC (bc-bge-<tier>-s0 from slurm/bc_bge.sh) -> its own tier.
# Output: ~/mhtable-runs/t-<selector>-test/result.json and t-bc-<tier>-test/result.json.
#   sbatch slurm/baselines_test.sh                       (after both bc_bge.sh tasks have finished)
#   sbatch --export=ALL,SMOKE=1 slurm/baselines_test.sh  (50 questions per tier, smoke BC checkpoints)
#SBATCH --job-name=baselines-test
#SBATCH --account=cs769_02
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/baselines_test_%J.out
#SBATCH --error=slurm/logs/baselines_test_%J.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/gpu_env.sh   # venv, HF_HUB_OFFLINE=1, Python headers for Triton + a C build check
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
DATA="$HOME/mhtable-data/v4"
RUNS="$HOME/mhtable-runs"
SMOKE="${SMOKE:-0}"
SUFFIX=""; X=()
if [ "$SMOKE" = 1 ]; then SUFFIX="-smoke"; X=(--limit 50); fi

evaluate() {  # evaluate <run id> <selector> <tiers...> [-- extra args]
  local rid="$1" sel="$2"; shift 2
  local tiers=() extra=()
  while [ $# -gt 0 ] && [ "$1" != -- ]; do tiers+=("$1"); shift; done
  [ $# -gt 0 ] && shift && extra=("$@")
  echo "== $rid ($sel, ${tiers[*]})"
  python -m mhtable.evaluate --config configs/experiments/eval.yaml --selector "$sel" --split test \
    --tiers "${tiers[@]}" --chunk 128 --data-dir "$DATA" --out "$RUNS" --run-id "$rid" "${X[@]}" "${extra[@]}"
}

evaluate "t-oracle-test$SUFFIX" oracle strict relaxed            # the oracle uses gold: not --reported
for sel in truncate random_rows heuristic; do
  evaluate "t-$sel-test$SUFFIX" "$sel" strict relaxed -- --reported
done
for tier in strict relaxed; do
  evaluate "t-bc-$tier-test$SUFFIX" policy "$tier" -- --reported --policy "$RUNS/bc-bge-$tier-s0$SUFFIX/policy.pt"
done
echo "== done"
