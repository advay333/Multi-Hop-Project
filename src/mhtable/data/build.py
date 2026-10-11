"""
Build the `data` HF repo from the `raw` one (PLAN.md P1, DESIGN.md §4.1).

    python -m mhtable.data.build --family spider_mmqa --push --tag data-v1      # molab
    python -m mhtable.data.build --family bird --push                           # molab
    python -m mhtable.data.build --family spider_mmqa --raw-local data/raw --limit 200   # laptop dev

Families are built independently because their databases never overlap: Spider and MMQA must
be split together (MMQA reuses Spider databases), BIRD is separate.

For every question: execute the gold SQL on its database (answer = normalised rows), analyse
the SQL against the action set (strict / relaxed tier flags, question category), and write a
public `Instance` plus a separate `GoldRecord`. Gold edges, anchors and per-hop entity sets are
added by the oracle in P2.

Output layout (mirrors the repo):
    instances/<source>.jsonl   gold/<source>.jsonl   schemas/<source>.json
    db/<source>/<name>.sqlite  splits/<family>/{train,dev,test,bird_dev}.txt
    splits/<family>/db_assignment.json   stats/<family>.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from mhtable import storage
from mhtable.data import split as splitter
from mhtable.data.execute import QueryTimeout, connect_ro, run
from mhtable.data.loaders import RawExample, SourceBundle
from mhtable.data.norm import norm_row, norm_value
from mhtable.data.raw import RawSource
from mhtable.data.schema import SchemaGraph
from mhtable.data.sqlshape import analyse_sql
from mhtable.types import GoldRecord, Instance

FAMILIES = {"spider_mmqa": ("spider", "mmqa"), "bird": ("bird",)}
ANSWER_ROW_CAP = 1000  # gold answers longer than this are flagged, not stored in full


# ---------------------------------------------------------------- per-question processing
def process_db(db_file: Path, schema: SchemaGraph, examples: list[RawExample], horizon: int,
               timeout_s: float) -> list[tuple[Instance, GoldRecord, dict]]:
    """Process every question of one database (one worker task)."""
    out = []
    con = connect_ro(db_file)
    try:
        for ex in examples:
            feats, _ = analyse_sql(ex.sql, schema, horizon)
            exec_reasons, answer = [], []
            try:
                rows = run(con, ex.sql, timeout_s, ANSWER_ROW_CAP + 1)
                answer = [norm_row(r) for r in rows[:ANSWER_ROW_CAP]]
                if len(rows) > ANSWER_ROW_CAP:
                    exec_reasons.append("answer_too_large")
                elif not rows or all(c == "<null>" for r in answer for c in r):
                    exec_reasons.append("empty_answer")
            except QueryTimeout:
                exec_reasons.append("exec_timeout")
            except Exception:
                exec_reasons.append("exec_error")

            inst = Instance(
                instance_id=ex.instance_id, source=ex.source, db_ref=ex.db_ref, db_id=ex.db_id,
                question=ex.question, evidence=ex.evidence, tables=schema.tables,
                n_gold_tables=feats.get("n_tables", 0),
                category=feats.get("category", ""),
                tier={"strict": not feats["reasons"] and not exec_reasons,
                      "relaxed": not feats["relaxed_reasons"] and not exec_reasons})
            gold = GoldRecord(instance_id=ex.instance_id, gold_sql=ex.sql, gold_answer=answer,
                              fail_reasons=sorted(set(feats["reasons"]) | set(exec_reasons)),
                              reference_answer=ex.extra.get("answer_text"))
            audit = {"exec": exec_reasons[0] if exec_reasons else "ok"}
            if "answer_text" in ex.extra:  # MMQA ships an answer string: check our execution agrees
                given = {norm_value(p) for p in str(ex.extra["answer_text"]).split(", ")}
                audit["mmqa_answer_exact_set_match"] = given == {c for r in answer for c in r}
            out.append((inst, gold, audit))
    finally:
        con.close()
    return out


def process_bundle(b: SourceBundle, horizon: int, timeout_s: float, workers: int):
    by_db: dict[str, list[RawExample]] = defaultdict(list)
    for ex in b.examples:
        by_db[ex.db_ref].append(ex)
    tasks = [(b.db_files[ref], b.schemas[ref], exs, horizon, timeout_s) for ref, exs in by_db.items()]
    results = []
    t0 = time.time()
    if workers <= 1:
        for i, t in enumerate(tasks):
            results.extend(process_db(*t))
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(tasks)} databases ({time.time() - t0:.0f}s)", flush=True)
    else:
        with ProcessPoolExecutor(workers) as pool:
            for i, r in enumerate(pool.map(process_db, *zip(*tasks))):
                results.extend(r)
                if (i + 1) % 50 == 0:
                    print(f"  {i + 1}/{len(tasks)} databases ({time.time() - t0:.0f}s)", flush=True)
    order = {ex.instance_id: i for i, ex in enumerate(b.examples)}
    results.sort(key=lambda r: order[r[0].instance_id])
    return results


# ---------------------------------------------------------------- loading
def load_family(family: str, raw: RawSource, out: Path, limit: int | None) -> dict[str, SourceBundle]:
    if family == "spider_mmqa":
        from mhtable.data.loaders import mmqa, spider

        sp = spider.load(raw.extracted(spider.ARCHIVE), limit)
        spider_graphs = {ref.split("/", 1)[1]: g for ref, g in sp.schemas.items()}
        mm = mmqa.load({k: raw.file(p) for k, p in mmqa.FILES.items()}, spider_graphs, out / "db", limit)
        return {"spider": sp, "mmqa": mm}
    if family == "bird":
        from mhtable.data.loaders import bird

        return {"bird": bird.load({s: raw.extracted(p) for s, p in bird.ARCHIVES.items()}, limit)}
    raise KeyError(family)


def _place_db(src: Path, dest: Path) -> None:
    """Put a database file into the output tree without doubling disk use when possible."""
    if dest.exists() or src.resolve() == dest.resolve():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)


# ---------------------------------------------------------------- main
def build(family: str, out: Path, raw: RawSource, limit: int | None = None, horizon: int = 5,
          timeout_s: float = 30.0, workers: int = 1, seed: int = 0) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    bundles = load_family(family, raw, out, limit)

    all_examples: list[RawExample] = []
    stats: dict = {"family": family, "limit": limit, "horizon": horizon, "sources": {}}
    for source, b in bundles.items():
        print(f"[{source}] {len(b.examples)} questions over {len(b.db_files)} databases", flush=True)
        results = process_bundle(b, horizon, timeout_s, workers)
        used = {inst.db_ref for inst, _, _ in results}
        for ref in used:
            _place_db(b.db_files[ref], out / "db" / f"{ref}.sqlite")
        (out / "instances").mkdir(exist_ok=True)
        (out / "gold").mkdir(exist_ok=True)
        (out / "schemas").mkdir(exist_ok=True)
        with open(out / "instances" / f"{source}.jsonl", "w", encoding="utf-8") as fi, \
             open(out / "gold" / f"{source}.jsonl", "w", encoding="utf-8") as fg:
            for inst, gold, _ in results:
                fi.write(json.dumps(inst.to_dict(), ensure_ascii=False) + "\n")
                fg.write(json.dumps(gold.to_dict(), ensure_ascii=False) + "\n")
        (out / "schemas" / f"{source}.json").write_text(
            json.dumps({ref: b.schemas[ref].to_dict() for ref in sorted(used)}, ensure_ascii=False))
        stats["sources"][source] = _source_stats(results, b)
        all_examples += b.examples

    inst_split, db_split = splitter.assign(all_examples, seed)
    sdir = out / "splits" / family
    sdir.mkdir(parents=True, exist_ok=True)
    by_split = defaultdict(list)
    for iid, s in inst_split.items():
        by_split[s].append(iid)
    for s, ids in by_split.items():
        (sdir / f"{s}.txt").write_text("\n".join(sorted(ids)) + "\n")
    (sdir / "db_assignment.json").write_text(json.dumps(dict(sorted(db_split.items())), indent=0))
    stats["splits"] = _split_stats(bundles, inst_split, out)

    (out / "stats").mkdir(exist_ok=True)
    (out / "stats" / f"{family}.json").write_text(json.dumps(stats, indent=2))
    return stats


def _source_stats(results, b: SourceBundle) -> dict:
    insts = [r[0] for r in results]
    audits = [r[2] for r in results]
    s = {
        "questions": len(insts),
        "databases": len({i.db_ref for i in insts}),
        "exec": dict(Counter(a["exec"] for a in audits)),
        "strict": sum(i.tier["strict"] for i in insts),
        "relaxed": sum(i.tier["relaxed"] for i in insts),
        "relaxed_by_category": dict(Counter(i.category for i in insts if i.tier["relaxed"])),
        "fail_reasons": dict(Counter(x for _, g, _ in results for x in g.fail_reasons).most_common(15)),
        **b.stats,
    }
    match = [a["mmqa_answer_exact_set_match"] for a in audits if "mmqa_answer_exact_set_match" in a]
    if match:
        # Strict set equality with MMQA's shipped answer text; formatting alone causes most misses (DESIGN.md §11).
        s["mmqa_answer_exact_set_match_rate"] = round(sum(match) / len(match), 4)
    return s


def _split_stats(bundles, inst_split, out: Path) -> dict:
    tiers = {}
    for source in bundles:
        for line in open(out / "instances" / f"{source}.jsonl", encoding="utf-8"):
            d = json.loads(line)
            tiers[d["instance_id"]] = (source, d["tier"]["relaxed"])
    table = defaultdict(Counter)
    for iid, s in inst_split.items():
        source, relaxed = tiers[iid]
        table[s][f"{source}_all"] += 1
        table[s][f"{source}_relaxed"] += relaxed
    return {s: dict(c) for s, c in sorted(table.items())}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--family", choices=sorted(FAMILIES), required=True)
    p.add_argument("--out", type=Path, default=Path("data/build"))
    p.add_argument("--raw-revision", default=None)
    p.add_argument("--raw-local", type=Path, default=None, help="dev only: read raw files from this folder")
    p.add_argument("--limit", type=int, default=None, help="dev only: first N questions per file")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--timeout", type=float, default=30.0, help="seconds per gold query")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--push", action="store_true", help="upload the output folder to the data repo")
    p.add_argument("--tag", default=None, help="tag the data repo after pushing, e.g. data-v1")
    args = p.parse_args(argv)

    if args.push and (args.raw_local or args.limit):
        p.error("--push is only allowed for full builds from the HF raw repo (no --raw-local / --limit)")

    stats = build(args.family, args.out, RawSource(args.raw_revision, args.raw_local), args.limit,
                  args.horizon, args.timeout, args.workers, args.seed)
    print(json.dumps(stats, indent=2))

    if args.push:
        storage.ensure_repos()
        print("uploading to", storage.repo_id("data"), flush=True)
        storage.push_large("data", args.out)
        if args.tag:
            storage.tag("data", args.tag, f"{args.family} build")
    return 0


if __name__ == "__main__":
    sys.exit(main())
