"""
Gate G-6 (oracle coverage) and the random-policy smoke test for the environment (PLAN.md P3).

    python -m mhtable.analysis.coverage --data-dir <data repo copy> --index-dir data/index_build \
        --family spider_mmqa --split train

For each instance in a tier, follow its gold trajectory through the environment and check that
every gold action is among the candidates offered at that step (matched by `Action.key()`).
Reports, for strict and relaxed separately:
  - reachability: fraction of instances whose whole gold trajectory is offered (the gate, >= 85%)
  - per-step coverage by action kind, and why missing actions were not offered
  - gold-edge omega distribution -> the edge threshold giving >= 98% gold-edge recall
Then runs random-policy episodes and logs |A(s)|, episode lengths and crashes.

This is an analysis script: it reads gold records, which the environment itself never does.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

from mhtable.data.norm import norm_value
from mhtable.env.actions import ActionConfig
from mhtable.env.env import EnvConfig, EnvData, TableEnv
from mhtable.types import Action, GoldRecord, Instance

SOURCES = {"spider_mmqa": ("spider", "mmqa"), "bird": ("bird",)}


def usable(gold: GoldRecord) -> bool:
    """Same rule as the oracle's `gold_relation_empty` check (data-v2 predates it)."""
    return bool(gold.gold_entity_sets) and gold.gold_entity_sets[-1]["rows"] > 0


def load(data_dir: Path, family: str, split: str):
    ids = set((data_dir / "splits" / family / f"{split}.txt").read_text().split())
    out = []
    for src in SOURCES[family]:
        golds = {}
        for line in open(data_dir / "gold" / f"{src}.jsonl", encoding="utf-8"):
            d = json.loads(line)
            if d["instance_id"] in ids:
                golds[d["instance_id"]] = GoldRecord.from_dict(d)
        for line in open(data_dir / "instances" / f"{src}.jsonl", encoding="utf-8"):
            d = json.loads(line)
            if d["instance_id"] in ids and usable(golds[d["instance_id"]]):
                out.append((Instance.from_dict(d), golds[d["instance_id"]]))
    return out


def why_missing(env: TableEnv, s, gold: Action, cfg: ActionConfig) -> str:
    """Best-effort reason a gold action was not offered."""
    ctx = env._context(s)
    if gold.kind == "SEED":
        return "seed_not_in_top_k"
    if gold.kind == "JOIN":
        if len(gold.edge) > 1:
            return "edge_composite"
        if cfg.edges == "declared":
            return "edge_not_declared_fk" if gold.edge not in ctx.schema.fks else "join_not_in_top_k"
        omega = {c.edge: c.omega for c in ctx.index.edges}.get(gold.edge)
        if omega is None:
            return "edge_not_inferred"
        return "edge_below_threshold" if omega < cfg.edge_threshold else "join_not_in_top_k"
    a = gold.anchor
    if a.op in ("not_like", "not_in", "is", "not_is", "ilike"):
        return f"anchor_op_{a.op}_not_generated"
    if cfg.tier == "strict" and a.op not in ("=", "<", ">"):
        return "anchor_op_outside_tier"
    numeric_val = len(a.values) == 1 and isinstance(a.values[0], (int, float))
    if a.op == "like":
        return "anchor_like_pattern_not_generated"
    if a.op in ("in",) or (a.op == "=" and not numeric_val):
        idx_vals = {norm_value(v) for v in ctx.index.cells.get(a.column, [])}
        if not all(norm_value(v) in idx_vals for v in a.values):
            return "anchor_value_not_in_cell_index"
        matched = {(c, norm_value(v)) for c, v, _ in ctx.cell_matches}
        if not all((a.column, norm_value(v)) in matched for v in a.values):
            return "anchor_value_not_matched_to_question"
        return "anchor_not_in_top_k"
    # numeric comparison / BETWEEN
    qn = {norm_value(x) for x in ctx.query_numbers}
    if not all(norm_value(v) in qn for v in a.values):
        return "anchor_number_not_in_question"
    st = ctx.index.columns.get(a.column)
    if st is None or not st.numeric:
        return "anchor_numeric_on_text_column"
    return "anchor_numeric_column_not_in_top_k"


def coverage(env: TableEnv, items, cfg: ActionConfig) -> dict:
    n = reach = 0
    step_total, step_hit = Counter(), Counter()
    reasons = Counter()
    sizes = []
    for inst, gold in items:
        n += 1
        s = env.reset(inst)
        ok = True
        for g in gold.gold_trajectory:
            cands = env.candidate_actions(s)
            sizes.append(len(cands))
            step_total[g.kind] += 1
            match = next((a for a in cands if a.key() == g.key()), None)
            if match is None:
                ok = False
                reasons[f"{g.kind}:{why_missing(env, s, g, cfg)}"] += 1
                match = g  # keep following the gold path to measure later steps too
            else:
                step_hit[g.kind] += 1
            s, done = env.step(s, match)
            if done:
                break
        reach += ok
    return {
        "instances": n, "reachable": reach, "reachability": round(reach / n, 4) if n else None,
        "step_coverage": {k: f"{step_hit[k]}/{step_total[k]} = {step_hit[k] / step_total[k]:.3f}" for k in step_total},
        "missing_reasons": dict(reasons.most_common()),
        "candidates_per_state": dict(sorted(Counter(min(x, 30) for x in sizes).items())),
    }


def edge_threshold(env: TableEnv, items, recall: float = 0.98) -> dict:
    omegas = []
    for inst, gold in items:
        idx = env.data.index(inst.db_ref)[0]
        table = {c.edge: c.omega for c in idx.edges}
        omegas += [table.get(e, 0.0) for e in gold.gold_edges]
    omegas.sort()
    cut = omegas[int((1 - recall) * len(omegas))] if omegas else 0.0
    return {"gold_edges": len(omegas), "not_inferred": sum(o == 0.0 for o in omegas),
            f"threshold_for_{int(recall * 100)}pct_recall": cut,
            "omega_quantiles": {q: omegas[int(q * (len(omegas) - 1))] for q in (0.01, 0.02, 0.05, 0.1, 0.5)} if omegas else {}}


def random_rollouts(env: TableEnv, items, episodes: int, seed: int = 0) -> dict:
    rng = random.Random(seed)
    lengths, sizes, crashes, rows = Counter(), [], [], []
    for i in range(episodes):
        inst, _ = items[rng.randrange(len(items))]
        try:
            s, done = env.reset(inst), False
            while not done:
                cands = env.candidate_actions(s)
                sizes.append(len(cands))
                s, done = env.step(s, rng.choice(cands))
            lengths[s.t] += 1
            rows.append(env.materialise(s).rows_total)
        except Exception:
            crashes.append((inst.instance_id, traceback.format_exc(limit=2)))
    sizes.sort()
    return {"episodes": episodes, "crashes": len(crashes), "crash_examples": crashes[:3],
            "episode_length": dict(sorted(lengths.items())),
            "candidates_per_state": {"mean": round(sum(sizes) / len(sizes), 2), "max": sizes[-1],
                                     "p50": sizes[len(sizes) // 2]},
            "final_rows_median": sorted(rows)[len(rows) // 2] if rows else None}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--index-dir", type=Path, required=True)
    p.add_argument("--family", choices=sorted(SOURCES), default="spider_mmqa")
    p.add_argument("--split", default="train")
    p.add_argument("--edge-threshold", type=float, default=None, help="default: tuned for 98%% gold-edge recall")
    p.add_argument("--edges", choices=["inferred", "declared"], default="inferred")
    p.add_argument("--random-episodes", type=int, default=1000)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    data = EnvData(args.data_dir, args.index_dir)
    items = load(args.data_dir, args.family, args.split)
    report = {"family": args.family, "split": args.split, "edges": args.edges}
    t0 = time.time()

    relaxed_items = [x for x in items if x[0].tier["relaxed"]][: args.limit]
    probe = TableEnv(EnvConfig(), data)
    report["edge_omega"] = edge_threshold(probe, relaxed_items)
    tau = args.edge_threshold if args.edge_threshold is not None else report["edge_omega"]["threshold_for_98pct_recall"]
    report["edge_threshold_used"] = tau

    for tier in ("strict", "relaxed"):
        cfg = ActionConfig(tier=tier, edges=args.edges, edge_threshold=tau)
        env = TableEnv(EnvConfig(actions=cfg), data)
        tier_items = [x for x in items if x[0].tier[tier]][: args.limit]
        report[tier] = coverage(env, tier_items, cfg)
        report[tier]["random_policy"] = random_rollouts(env, tier_items, args.random_episodes)
        env.close()
        print(f"[{tier}] done ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)

    text = json.dumps(report, indent=2, default=str)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
