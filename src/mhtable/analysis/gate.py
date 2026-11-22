"""P9 gate summary: does RL keep or beat its behaviour-cloning start? (PLAN.md P9)

Reads each RL run's `metrics.jsonl` (dev evaluations; iteration 0 = the BC checkpoint, before any
update) and prints the start, last and best dev EM. Works while runs are still going, and only
reads files, so it is fine on a login node.

    python -m mhtable.analysis.gate ~/mhtable-runs            (default pattern: ppo-r1-*)
    python -m mhtable.analysis.gate ~/mhtable-runs --pattern 'grpo-*'

Gate per run: last EM >= start EM and the best iteration is not 0. The re-score of policy.pt is
not the gate: that checkpoint is the best dev evaluation including iteration 0, so it can never
fall below BC.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarise(run_dir: Path) -> dict | None:
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return None
    recs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    recs = [r for r in recs if r.get("eval", {}).get("em") is not None]
    if not recs:
        return None
    start = next((r for r in recs if r["iteration"] == 0), None)
    last = recs[-1]
    best = max(recs, key=lambda r: r["eval"]["em"])  # first maximum, as train.py keeps it
    out = {"run": run_dir.name, "evals": len(recs), "last_iteration": last["iteration"],
           "reader_calls": last.get("reader_calls"),
           "start_em": start["eval"]["em"] if start else None, "last_em": last["eval"]["em"],
           "best_em": best["eval"]["em"], "best_iteration": best["iteration"],
           "last_tokens": last["eval"].get("tokens"), "finished": (run_dir / "result.json").exists()}
    if start:
        out["gate"] = "PASS" if out["last_em"] >= out["start_em"] and out["best_iteration"] != 0 else "FAIL"
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", type=Path, help="folder holding the run folders, e.g. ~/mhtable-runs")
    p.add_argument("--pattern", default="ppo-r1-*", help="run folder glob (default: ppo-r1-*)")
    args = p.parse_args(argv)
    rows = [s for d in sorted(args.runs.glob(args.pattern)) if d.is_dir() and (s := summarise(d))]
    if not rows:
        print(f"no runs with metrics.jsonl matching {args.pattern} under {args.runs}")
        return 1
    cols = ["run", "start_em", "last_em", "best_em", "best_iteration", "last_iteration", "evals",
            "reader_calls", "last_tokens", "finished", "gate"]
    print("\t".join(cols))
    for r in rows:
        print("\t".join(str(r.get(c, "")) for c in cols))
    print("\ngate (per run): last_em >= start_em and best_iteration != 0; start = iteration 0 = the BC policy."
          "\nA run that is not finished shows its gate so far.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
