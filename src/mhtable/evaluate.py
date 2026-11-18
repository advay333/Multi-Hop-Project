"""
Evaluation entry point: one selector, one reader, both action tiers (DESIGN.md §6.6, report D).

    python -m mhtable.evaluate --config configs/experiments/eval.yaml --selector oracle \
        [--policy runs/<run>/policy.pt] [--split dev] [--tiers strict relaxed] [--limit 200] \
        [--data-dir D] [--index-dir I] [--out runs] [--run-id NAME] [--reported] [key=value ...]

Selectors: `oracle` (follows the gold trajectory: the ceiling for this reader, never a reported
system) and `policy` (a trained checkpoint, `--policy`). The P6 baselines plug in here as more
selectors. Every selector goes through the same rollout, materialiser, prompt, reader and budget
(PLAN.md §6 rules 5–6). Gold is used for metrics only (`oracle_mode="metric_only"`; the reward
mode is refused here by construction).

Writes `<out>/<run_id>/`:
    config.yaml                  exact settings + code revision
    predictions_<tier>.jsonl     one line per question: answer, gold answer, every metric
    result.json                  per tier: metric suite overall and by slice (metrics.summarise);
                                 reader: calls, cache hits, seconds, calls per hour, cut answers
    reader_cache.jsonl           the reader-answer shard (DESIGN.md §4.4)

`--reported` refuses the mock reader and the oracle selector (DESIGN.md §8 invariant 1); every
result also records `reportable`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mhtable.encoders  # noqa: F401  (registers encoders)
import mhtable.policies  # noqa: F401  (registers policies)
import mhtable.readers  # noqa: F401  (registers readers)
from mhtable.config import load_config
from mhtable.dataset import load_split
from mhtable.metrics import summarise
from mhtable.readers.base import CachedReader
from mhtable.registry import build
from mhtable.rollout import PolicyActor, collect
from mhtable.selectors.oracle import OracleSelector
from mhtable.train import build_env, build_featuriser, git_dirty, git_sha, load_checkpoint, seed_everything

SELECTORS = ("oracle", "policy")


def question_row(inst, gold, tr) -> dict:
    info = tr.info
    return {"instance_id": inst.instance_id, "source": inst.source, "category": inst.category,
            "tables": f"{inst.n_gold_tables}_tables", "monotone": gold.monotone, "question": inst.question,
            "answer": tr.answer, "gold_answer": gold.gold_answer, "em": info["r_ans"], "set_f1": info["set_f1"],
            "path_match": float(info["path_match"]), "sufficient": float(info["sufficient"]),
            "answer_in_subtable": info["answer_in_subtable"], "tokens": tr.subtable_tokens,
            "overflow": float(info["overflow"]), "rows_shown": info["rows_shown"], "steps": len(tr.steps),
            "actions": info.get("actions", []), "visited": info["visited"], "reader_called": tr.reader_called,
            "columns": info.get("columns", []), "subtable": tr.subtable,  # exactly what the reader saw
            "hops": info["hops"], **{k: info[k] for k in ("edge_hit", "edge_pred", "edge_gold",
                                                          "anchor_hit", "anchor_pred", "anchor_gold")}}


def evaluate_tier(cfg, args, tier: str, reader: CachedReader, run_dir: Path, log) -> dict:
    env = build_env(cfg, args.data_dir, args.index_dir, tier=tier)
    try:
        items = load_split(env.data, args.split, tier, tuple(cfg.data.sources), args.limit, cfg.seed)
        golds = {i.instance_id: g for i, g in items}
        if args.selector == "oracle":
            actor = OracleSelector(golds)
        else:
            policy, ck = load_checkpoint(args.policy)
            feat = build_featuriser(cfg, env)
            if ck["spec"] != feat.spec:
                raise SystemExit(f"{args.policy} was trained with features {ck['spec']}, this config gives {feat.spec}")
            policy.eval()
            actor = PolicyActor(policy, feat)
        log(f"[{tier}] {len(items)} {args.split} questions, selector {args.selector}, reader {reader.id}, "
            f"budget counted by {env.counter.id}")
        rows, t0, calls0 = [], time.time(), reader.calls
        path = run_dir / f"predictions_{tier}.jsonl"
        path.unlink(missing_ok=True)
        for k in range(0, len(items), args.chunk):
            part = items[k:k + args.chunk]
            trajs = collect(env, actor, reader, [i for i, _ in part], mode="argmax", oracle_mode="metric_only",
                            golds={i.instance_id: g for i, g in part}, detail=True)
            new = [question_row(inst, gold, tr) for (inst, gold), tr in zip(part, trajs)]
            rows += new
            with open(path, "a", encoding="utf-8") as f:
                for r in new:
                    f.write(json.dumps(r, default=str) + "\n")
            log(f"[{tier}] {len(rows)}/{len(items)} questions, em {sum(r['em'] for r in rows) / len(rows):.3f}, "
                f"reader calls {reader.calls} ({time.time() - t0:.0f}s)")
        out = summarise(rows)
        out["reader_calls"], out["minutes"] = reader.calls - calls0, round((time.time() - t0) / 60, 2)
        return out
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--selector", choices=SELECTORS, required=True)
    p.add_argument("--policy", default=None, help="policy checkpoint (selector policy)")
    p.add_argument("--split", default="dev")
    p.add_argument("--tiers", nargs="+", default=["strict", "relaxed"], choices=["strict", "relaxed"])
    p.add_argument("--limit", type=int, default=None, help="seeded random subset of each tier's questions")
    p.add_argument("--chunk", type=int, default=256, help="questions per rollout / reader batch")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--index-dir", type=Path, default=None)
    p.add_argument("--out", type=Path, default=Path("runs"))
    p.add_argument("--run-id", default=None)
    p.add_argument("--reader-cache", nargs="*", default=[], help="extra reader-answer shards to preload")
    p.add_argument("--reported", action="store_true", help="refuse the mock reader and the oracle selector")
    p.add_argument("overrides", nargs="*", help="config overrides, e.g. reader.dtype=bfloat16")
    args = p.parse_args(argv)
    if args.selector == "policy" and not args.policy:
        p.error("--selector policy needs --policy <checkpoint>")

    cfg = load_config(args.config, args.overrides)
    seed_everything(cfg.seed)
    run_id = args.run_id or f"eval-{args.selector}-{args.split}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = args.out / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    info = {"run.id": run_id, "run.code": git_sha(), "run.code_dirty": git_dirty(), "run.selector": args.selector,
            "run.policy": args.policy, "run.split": args.split, "run.limit": args.limit,
            "run.data_dir": str(args.data_dir) if args.data_dir else None}
    (run_dir / "config.yaml").write_text(cfg.with_overrides(info).to_yaml(), encoding="utf-8")

    def log(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    t0 = time.time()
    base = build("reader", cfg.reader)
    log(f"reader {base.id} ready ({time.time() - t0:.0f}s)")
    reportable = not base.needs_gold and args.selector != "oracle"
    if args.reported and not reportable:
        raise SystemExit("--reported: the mock reader and the oracle selector never produce a reported result")
    reader = CachedReader(base, run_dir / "reader_cache.jsonl", preload=tuple(args.reader_cache))

    result = {"selector": args.selector, "policy": args.policy, "split": args.split, "reader": reader.id,
              "reportable": reportable, "data_revision": cfg.get_path("data.revision"),
              "sources": list(cfg.data.sources), "budget_tokens": cfg.env.budget_tokens,
              "tokenizer": cfg.get_path("env.tokenizer"), "tiers": {}}
    for tier in args.tiers:
        result["tiers"][tier] = evaluate_tier(cfg, args, tier, reader, run_dir, log)
    stats = getattr(base, "stats", None)
    result["reader_stats"] = {"calls": reader.calls, "cache_hits": reader.hits, "seconds": round(reader.seconds, 1),
                              "calls_per_hour": round(3600 * reader.calls / reader.seconds, 1) if reader.seconds else None,
                              **({"backend": stats.to_dict()} if stats is not None else {})}
    result["minutes"] = round((time.time() - t0) / 60, 2)
    (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    head = {t: {k: r.get(k) for k in ("n", "em", "set_f1", "path_match", "em_on_path", "answer_in_subtable",
                                      "em_when_answer_shown", "tokens")} for t, r in result["tiers"].items()}
    print(json.dumps({"headline": head, "reader_stats": result["reader_stats"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
