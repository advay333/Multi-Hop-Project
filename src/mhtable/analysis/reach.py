"""
Gate G-6 (oracle-trajectory reachability) without the databases.

    python -m mhtable.analysis.reach --family bird --splits train bird_dev
    python -m mhtable.analysis.reach --family spider_mmqa --splits train dev

Whether a gold action is among the candidates offered depends only on the question, the schema
and the env index, not on executing SQL. So this runs on the laptop for BIRD too (whose
databases are never downloaded locally), reading gold/instances/schemas from the data repo and
the index from `--index-revision`. `analysis.coverage` is the full version with look-ahead
counts and random-policy rollouts (needs the databases).

Reports per split x tier: reachability (the gate, >= 85%), per-kind step coverage, why missing
actions were not offered, *sole blockers* (questions whose only missing reason is X, i.e. what
fixing X alone would convert), and |A(s)| along the gold path.

This is an analysis script: it reads gold records, which the environment itself never does.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from types import SimpleNamespace

from mhtable import storage
from mhtable.analysis.coverage import SOURCES, usable, why_missing
from mhtable.data.schema import SchemaGraph
from mhtable.env.actions import ActionConfig, EpisodeContext, enumerate_actions
from mhtable.env.env import TableEnv
from mhtable.env.index import CellSearch, DbIndex
from mhtable.env.text import LexicalSimilarity
from mhtable.types import Action, Anchor, GoldRecord, Instance, State


def query_text(inst: Instance) -> str:
    """Same text the environment ranks candidates against (question + BIRD evidence)."""
    return inst.question + (f" ({inst.evidence})" if inst.evidence else "")


def as_rebuilt(gold: GoldRecord) -> GoldRecord:
    """Apply gold-analysis changes made after data-v2 to a data-v2 record, so coverage reflects
    the next rebuild: `col LIKE 'text'` without wildcards is now recorded as `col = 'text'`.
    (The rebuild's reconstruction check may still drop a few of these on case differences.)"""
    def fix(a: Anchor) -> Anchor:
        v = a.values[0] if a.values else None
        if a.op in ("like", "ilike") and isinstance(v, str) and "%" not in v and "_" not in v:
            return Anchor(a.column, "=", a.values)
        return a
    gold.gold_trajectory = [Action(x.kind, x.table, fix(x.anchor) if x.anchor else None, x.edge, x.lookahead_rows)
                            for x in gold.gold_trajectory]
    gold.gold_anchors = [fix(a) for a in gold.gold_anchors]
    return gold


class Loader:
    def __init__(self, family: str, data_revision: str, index_revision: str):
        self.f = lambda p: storage.fetch("data", p, revision=data_revision)
        self.fi = lambda p: storage.fetch("data", p, revision=index_revision)
        self.family = family
        self.schemas, self.golds, self.insts = {}, {}, []
        for src in SOURCES[family]:
            self.schemas.update(json.load(open(self.f(f"schemas/{src}.json"), encoding="utf-8")))
            for line in open(self.f(f"gold/{src}.jsonl"), encoding="utf-8"):
                d = json.loads(line)
                self.golds[d["instance_id"]] = as_rebuilt(GoldRecord.from_dict(d))
            self.insts += [Instance.from_dict(json.loads(x)) for x in open(self.f(f"instances/{src}.jsonl"), encoding="utf-8")]
        self._idx: dict = {}

    def split_ids(self, split: str) -> set[str]:
        return set(open(self.f(f"splits/{self.family}/{split}.txt")).read().split())

    def index(self, ref: str):
        if ref not in self._idx:
            d = DbIndex.from_dict(json.load(open(self.fi(f"index/{ref}.json"), encoding="utf-8")))
            self._idx[ref] = (d, CellSearch(d))
        return self._idx[ref]


def reach(loader: Loader, split: str, tier: str, cfg: ActionConfig, limit: int | None = None) -> dict:
    ids = loader.split_ids(split)
    sim = LexicalSimilarity()
    items = [(i, loader.golds[i.instance_id]) for i in loader.insts
             if i.instance_id in ids and i.tier[tier] and usable(loader.golds[i.instance_id])][:limit]
    ok_n, tot, hit, why, sole, sizes = 0, Counter(), Counter(), Counter(), Counter(), []
    misses = []
    for inst, gold in items:
        idx, search = loader.index(inst.db_ref)
        schema = SchemaGraph.from_dict(loader.schemas[inst.db_ref])
        ctx = EpisodeContext.build(schema, idx, search, query_text(inst), inst.tables, sim)
        fake_env = SimpleNamespace(_context=lambda s, c=ctx: c)  # why_missing only needs the context
        s, reasons = State(instance=inst), []
        for g in gold.gold_trajectory:
            cands = enumerate_actions(s, ctx, cfg)
            sizes.append(len(cands))
            tot[g.kind] += 1
            if any(a.key() == g.key() for a in cands):
                hit[g.kind] += 1
            else:
                reasons.append(f"{g.kind}:{why_missing(fake_env, s, g, cfg)}")
            if g.kind == "STOP":
                break
            s = TableEnv.apply(s, g)
        why.update(reasons)
        if not reasons:
            ok_n += 1
        else:
            misses.append({"id": inst.instance_id, "reasons": reasons})
            if len(set(reasons)) == 1:
                sole[reasons[0]] += 1
    n = len(items)
    return {
        "questions": n,
        "reachability": round(ok_n / n, 4) if n else None,
        "reachable": ok_n,
        "steps": {k: round(hit[k] / tot[k], 4) for k in tot},
        "missing": dict(why.most_common(12)),
        "sole_blocker": dict(sole.most_common(12)),
        "action_set": {"mean": round(statistics.mean(sizes), 2) if sizes else 0, "max": max(sizes, default=0)},
        "_misses": misses,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--family", choices=sorted(SOURCES), required=True)
    p.add_argument("--splits", nargs="+", default=["train"])
    p.add_argument("--tiers", nargs="+", default=["strict", "relaxed"])
    p.add_argument("--data-revision", default="data-v2")
    p.add_argument("--index-revision", default="main", help="the env index was pushed after the data-v2 tag")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--misses-out", default=None, help="write per-question miss reasons (jsonl) here")
    args = p.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    loader = Loader(args.family, args.data_revision, args.index_revision)
    report, misses = {}, []
    for split in args.splits:
        for tier in args.tiers:
            r = reach(loader, split, tier, ActionConfig(tier=tier), args.limit)
            misses += [dict(m, split=split, tier=tier) for m in r.pop("_misses")]
            report[f"{split}/{tier}"] = r
            print(f"[{split}/{tier}] reach {r['reachable']}/{r['questions']} = {r['reachability']}  "
                  f"|A| mean {r['action_set']['mean']} max {r['action_set']['max']}", file=sys.stderr, flush=True)
    print(json.dumps(report, indent=1))
    if args.misses_out:
        with open(args.misses_out, "w", encoding="utf-8") as fo:
            for m in misses:
                fo.write(json.dumps(m) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
