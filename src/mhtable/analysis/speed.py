"""
Environment speed profile per database (PLAN §4a, Track B): how long look-ahead counts and
materialisation take on real databases, and which queries are slow.

    python -m mhtable.analysis.speed --data-dir <data copy> --family bird --split train \
        --top 8 --budget-s 180 --out speed_before.json

For each database (the `--top` largest files used by the split, or `--dbs ...`) it runs seeded
random-policy episodes until `--budget-s` seconds or `--episodes` episodes, and reports:
  - look-ahead queries: count, mean / max seconds, share that timed out (reported as -1 rows),
    seconds per action kind (SEED / ANCHOR / JOIN), memo hits
  - seconds per `candidate_actions` call and per episode, materialise seconds / timeouts
  - crashes, and SQLite's query plan (EXPLAIN QUERY PLAN) for the slowest look-aheads: an
    "AUTOMATIC ... INDEX" line means a join without a permanent index, "SCAN <table>" a full read
Gold-free: reads instances only (random actions), never gold records.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

from mhtable.env.actions import ActionConfig
from mhtable.env.env import EnvConfig, EnvData, TableEnv
from mhtable.types import Instance

SOURCES = {"spider_mmqa": ("spider", "mmqa"), "bird": ("bird",)}


def load_instances(data_dir: Path, family: str, split: str) -> dict[str, list[Instance]]:
    ids = set((data_dir / "splits" / family / f"{split}.txt").read_text().split())
    by_db: dict[str, list[Instance]] = {}
    for src in SOURCES[family]:
        for line in open(data_dir / "instances" / f"{src}.jsonl", encoding="utf-8"):
            d = json.loads(line)
            if d["instance_id"] in ids:
                inst = Instance.from_dict(d)
                by_db.setdefault(inst.db_ref, []).append(inst)
    return by_db


def plans(env: TableEnv, db_ref: str, k: int = 3) -> list[dict]:
    con = env._conn(db_ref)
    out = []
    for dt, ref, sql in sorted(env.slow_queries, reverse=True):
        if ref != db_ref or len(out) >= k:
            continue
        try:
            plan = [r[-1] for r in con.execute("EXPLAIN QUERY PLAN " + sql).fetchall()]
        except Exception as e:  # noqa: BLE001 - diagnostic only
            plan = [f"error: {e}"]
        out.append({"seconds": round(dt, 3), "sql": sql[:600], "plan": plan})
    return out


def profile_db(env: TableEnv, items: list[Instance], budget_s: float, episodes: int, seed: int) -> dict:
    rng = random.Random(seed)
    env.stats.clear()
    env.slow_queries.clear()
    t0 = time.perf_counter()
    n_ep, n_calls, call_s, crashes, ep_s = 0, 0, 0.0, [], []
    while n_ep < episodes and time.perf_counter() - t0 < budget_s:
        inst = items[rng.randrange(len(items))]
        te = time.perf_counter()
        try:
            s, done = env.reset(inst), False
            while not done:
                tc = time.perf_counter()
                cands = env.candidate_actions(s)
                call_s += time.perf_counter() - tc
                n_calls += 1
                s, done = env.step(s, rng.choice(cands))
            env.materialise(s)
        except Exception:  # noqa: BLE001 - counted and reported
            crashes.append((inst.instance_id, traceback.format_exc(limit=3)))
        ep_s.append(time.perf_counter() - te)
        n_ep += 1
    st = env.stats
    q = st["lookahead_queries"]
    kinds = {k: {"actions": st[f"lookahead_{k}_actions"],
                 "mean_s": round(st[f"lookahead_{k}_seconds"] / st[f"lookahead_{k}_actions"], 4)}
             for k in ("SEED", "ANCHOR", "JOIN") if st[f"lookahead_{k}_actions"]}
    ep_s.sort()
    return {
        "episodes": n_ep, "seconds": round(time.perf_counter() - t0, 1),
        "episode_s": {"mean": round(sum(ep_s) / len(ep_s), 3), "p50": round(ep_s[len(ep_s) // 2], 3),
                      "max": round(ep_s[-1], 3)} if ep_s else None,
        "candidate_actions_mean_s": round(call_s / n_calls, 4) if n_calls else None,
        "lookahead": {"queries": q, "memo_hits": st["lookahead_memo_hits"],
                      "mean_s": round(st["lookahead_seconds"] / q, 4) if q else None,
                      "max_s": round(max((x[0] for x in env.slow_queries), default=0.0), 3),
                      "timeout_rate": round(st["lookahead_timeouts"] / q, 4) if q else None,
                      "by_kind": kinds},
        "materialise": {"calls": st["materialise_calls"], "timeouts": st["materialise_timeouts"],
                        "unordered": st["materialise_unordered"],
                        "mean_s": round(st["materialise_seconds"] / st["materialise_calls"], 4)
                        if st["materialise_calls"] else None},
        "crashes": len(crashes), "crash_examples": crashes[:2],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--index-dir", type=Path, default=None)
    p.add_argument("--family", choices=sorted(SOURCES), default="bird")
    p.add_argument("--split", default="train")
    p.add_argument("--dbs", nargs="*", default=None)
    p.add_argument("--top", type=int, default=8, help="the N largest DB files of the split")
    p.add_argument("--tier", choices=["strict", "relaxed"], default="relaxed")
    p.add_argument("--edge-threshold", type=float, default=0.6667)
    p.add_argument("--budget-s", type=float, default=180.0, help="seconds per database")
    p.add_argument("--episodes", type=int, default=200, help="max episodes per database")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    by_db = load_instances(args.data_dir, args.family, args.split)
    data = EnvData(args.data_dir, args.index_dir or args.data_dir)
    if args.dbs:
        dbs = [d for d in args.dbs if d in by_db]
    else:
        dbs = sorted(by_db, key=lambda d: -data.db_file(d).stat().st_size)[: args.top]
    cfg = EnvConfig(actions=ActionConfig(tier=args.tier, edge_threshold=args.edge_threshold))
    report = {"family": args.family, "split": args.split, "tier": args.tier, "budget_s": args.budget_s,
              "env": {"lookahead_timeout_s": cfg.lookahead_timeout_s, "lookahead_cap": cfg.lookahead_cap},
              "databases": {}}
    totals = Counter()
    for ref in dbs:
        env = TableEnv(cfg, data)
        r = profile_db(env, by_db[ref], args.budget_s, args.episodes, args.seed)
        r["file_gib"] = round(data.db_file(ref).stat().st_size / 2**30, 2)
        r["questions"] = len(by_db[ref])
        r["slowest"] = plans(env, ref)
        env.close()
        report["databases"][ref] = r
        totals["episodes"] += r["episodes"]
        totals["crashes"] += r["crashes"]
        la = r["lookahead"]
        print(f"{ref} ({r['file_gib']} GiB): {r['episodes']} episodes in {r['seconds']}s, "
              f"look-ahead mean {la['mean_s']}s max {la['max_s']}s timeouts {la['timeout_rate']}, "
              f"crashes {r['crashes']}", file=sys.stderr, flush=True)
    report["totals"] = dict(totals)
    text = json.dumps(report, indent=2, default=str)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
