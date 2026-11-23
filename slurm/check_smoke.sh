#!/usr/bin/env bash
# Check smoke runs on the LOGIN node (only reads files; no compute). For each log given, prints:
# the job line, the C build check, any Python traceback / error, whether it reached the end, and
# for the run it trained: train result.json (iterations done, final dev EM) and the test result.
#   bash slurm/check_smoke.sh slurm/logs/p9_big_<JOBID>_2.out slurm/logs/p9_scratch_<JOBID>_4.out
#   bash slurm/check_smoke.sh            (no arguments: the 2 newest p9_big_* and p9_scratch_* logs)
set -uo pipefail
cd "$(dirname "$0")/.."
RUNS="$HOME/mhtable-runs"
source "$HOME/mhtable-venv/bin/activate" 2>/dev/null || true

LOGS=("$@")
if [ ${#LOGS[@]} -eq 0 ]; then
  for pat in p9_big p9_scratch; do
    f=$(ls -t slurm/logs/${pat}_*.out 2>/dev/null | head -1); [ -n "$f" ] && LOGS+=("$f")
  done
fi
[ ${#LOGS[@]} -eq 0 ] && { echo "no logs found"; exit 1; }

for log in "${LOGS[@]}"; do
  echo "================ $log"
  [ -f "$log" ] || { echo "  missing"; continue; }
  grep -m1 '^job ' "$log" | sed 's/^/  /'
  grep -m1 'C build check' "$log" | sed 's/^/  /' || echo "  (no C build check line)"
  # real Python errors; vLLM's harmless import-time AssertionError warnings are skipped
  errs=$(grep -nE 'Traceback|Error|error:|Killed|CANCELLED|TIME LIMIT|OOM' "$log" \
         | grep -vE 'WARNING .*import_utils|AssertionError$' | head -8)
  if [ -n "$errs" ]; then echo "  !! problems:"; echo "$errs" | sed 's/^/     /'; else echo "  no errors in log"; fi
  run=$(grep -m1 '^job ' "$log" | sed -E 's/.*: ([^ |]+) \|.*/\1/')
  [ -z "$run" ] && { echo "  could not read the run id"; continue; }
  python - "$RUNS" "$run" <<'EOF'
import json, os, sys
runs, run = sys.argv[1], sys.argv[2]
def show(path, label):
    if not os.path.exists(path):
        print(f"  !! {label}: missing ({path})"); return None
    return json.load(open(path))
r = show(f"{runs}/{run}/result.json", "train result")
if r is not None:
    m = [json.loads(l) for l in open(f"{runs}/{run}/metrics.jsonl")] if os.path.exists(f"{runs}/{run}/metrics.jsonl") else []
    ev = [x for x in m if x.get("eval")]
    print(f"  train OK: {run}  evals {len(ev)}  last iteration {ev[-1]['iteration'] if ev else '?'}  "
          f"dev EM {ev[-1]['eval']['em'] if ev else '?'}  reader calls {m[-1].get('reader_calls') if m else '?'}")
t = show(f"{runs}/t-{run}-test/result.json", "test result")
if t is not None:
    for tier, v in t["tiers"].items():
        print(f"  test OK: {tier} n={v['n']} EM={v['em']:.3f} tokens={v['tokens']:.0f}")
EOF
  tail -1 "$log" | sed 's/^/  last line: /'
done
echo "================ smoke OK = no '!!' lines above"
