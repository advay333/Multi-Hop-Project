"""
Build the frozen embedding cache for one family (PLAN.md P4, DESIGN.md §6.5).

    python -m mhtable.encoders.build_cache --family spider_mmqa --revision data-v3 \
        [--model BAAI/bge-m3] [--data-dir D] [--index-dir I] [--out cache_build] [--push]

Texts are exactly `featurise.required_texts` (question + BIRD evidence, `table: columns`,
`table column`, every indexed cell value), with and without header ablation, for every question
of the family that fits either action tier, in every split. Only `instances/`, `schemas/` and
`index/` are read — never `gold/`.

Output, under `<out>/<encoder id>/`:
    parts/part-00000.npz ...   chunks of `--chunk` texts in sorted order; a rerun skips finished
                               parts, so a dead GPU session resumes where it stopped
    <family>.npz               all texts, float16 (keys, vectors)
    <family>.json              manifest: model, dim, counts, data revision, code revision
`--push` uploads `<family>.npz` + `<family>.json` to mhtable-cache at `embeddings/<encoder id>/`.
Training loads it with `encoder.cache: hf:embeddings/bge-m3/<family>.npz` (train.build_featuriser).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from mhtable.encoders.cache import EmbCache
from mhtable.registry import build

SOURCES = {"spider_mmqa": ("spider", "mmqa"), "bird": ("bird",)}


def fetch_inputs(family: str, revision: str | None) -> Path:
    """One snapshot of instances/, schemas/ and index/ for the family (no DB files, no gold)."""
    from huggingface_hub import snapshot_download

    from mhtable import storage

    pats = [f"instances/{s}.jsonl" for s in SOURCES[family]] + [f"schemas/{s}.json" for s in SOURCES[family]]
    pats += [f"index/{s}/*" for s in SOURCES[family]]
    return Path(snapshot_download(storage.repo_id("data"), repo_type="dataset", revision=revision,
                                  allow_patterns=pats, cache_dir=storage._cache_dir()))


def collect_texts(family: str, data_dir: Path, index_dir: Path | None = None, limit_dbs: int | None = None) -> tuple[list[str], dict]:
    from mhtable.env.env import EnvData
    from mhtable.featurise import required_texts
    from mhtable.types import Instance

    data = EnvData(data_dir, index_dir or data_dir)
    by_db: dict[str, list[Instance]] = defaultdict(list)
    n_q = 0
    for src in SOURCES[family]:
        path = data.data_dir / f"instances/{src}.jsonl"
        if not path.exists():  # a local copy holding only some sources (tests, partial builds)
            print(f"  no instances/{src}.jsonl in {data.data_dir}: source skipped", flush=True)
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                inst = Instance.from_dict(json.loads(line))
                if inst.tier.get("strict") or inst.tier.get("relaxed"):
                    by_db[inst.db_ref].append(inst)
                    n_q += 1
    dbs = sorted(by_db)[:limit_dbs] if limit_dbs else sorted(by_db)
    texts: set[str] = set()
    for k, ref in enumerate(dbs, 1):
        schema = data.schema(ref)
        index, _ = data.index(ref)
        for inst in by_db[ref]:
            for ablate in (False, True):
                texts.update(required_texts(inst, schema, index, ablate))
        if k % 100 == 0:
            print(f"  texts from {k}/{len(dbs)} databases: {len(texts):,}", flush=True)
    return sorted(texts), {"questions": sum(len(by_db[r]) for r in dbs), "databases": len(dbs)}


def git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10,
                              cwd=Path(__file__).parent).stdout.strip() or None
    except Exception:
        return None


def build_cache(family: str, encoder, texts: list[str], out: Path, chunk: int = 50_000) -> Path:
    """Encode `texts` in resumable parts and merge them into `<out>/<family>.npz`."""
    parts = out / "parts" / family
    parts.mkdir(parents=True, exist_ok=True)
    files = []
    t0 = time.time()
    for k in range(0, len(texts), chunk):
        p = parts / f"part-{k // chunk:05d}.npz"
        files.append(p)
        if p.exists():
            continue
        block = texts[k:k + chunk]
        vec = encoder.encode(block)
        tmp = p.with_suffix(".tmp.npz")
        np.savez(tmp, keys=np.array(block, dtype=object), vectors=vec.astype(np.float16))
        tmp.replace(p)
        done = min(k + chunk, len(texts))
        rate = done / max(time.time() - t0, 1e-6)
        print(f"  encoded {done:,}/{len(texts):,} texts ({rate:,.0f}/s)", flush=True)
    cache = EmbCache.load(files) if files else EmbCache(encoder.dim)
    if cache.texts() != texts:
        raise RuntimeError("the parts do not match the current text list: delete the parts folder and rerun")
    final = out / f"{family}.npz"
    cache.save(final)
    return final


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--family", choices=sorted(SOURCES), required=True)
    p.add_argument("--revision", default=None, help="mhtable-data tag, e.g. data-v3")
    p.add_argument("--encoder", default="st", help="registered encoder name (st | hash)")
    p.add_argument("--model", default="BAAI/bge-m3")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--data-dir", type=Path, default=None, help="local copy of the data repo (default: HuggingFace)")
    p.add_argument("--index-dir", type=Path, default=None)
    p.add_argument("--out", type=Path, default=Path("cache_build"))
    p.add_argument("--chunk", type=int, default=50_000)
    p.add_argument("--limit-dbs", type=int, default=None, help="smoke runs: only the first N databases")
    p.add_argument("--push", action="store_true", help="upload to mhtable-cache embeddings/<encoder id>/")
    args = p.parse_args(argv)

    t0 = time.time()
    data_dir = args.data_dir or fetch_inputs(args.family, args.revision)
    texts, counts = collect_texts(args.family, data_dir, args.index_dir, args.limit_dbs)
    print(f"{len(texts):,} distinct texts from {counts['questions']:,} questions in {counts['databases']} "
          f"databases ({time.time() - t0:.0f}s)", flush=True)
    cfg = {"name": args.encoder}
    if args.encoder == "st":
        cfg.update(model=args.model, batch_size=args.batch_size)
    encoder = build("encoder", cfg)
    out = args.out / encoder.id
    t1 = time.time()
    final = build_cache(args.family, encoder, texts, out, args.chunk)
    secs = time.time() - t1
    manifest = {"family": args.family, "encoder": encoder.id, "model": args.model if args.encoder == "st" else None,
                "dim": encoder.dim, "texts": len(texts), **counts, "data_revision": args.revision,
                "code": git_sha(), "dtype": "float16", "encode_seconds": round(secs, 1),
                "texts_per_second": round(len(texts) / max(secs, 1e-6), 1),
                "bytes": final.stat().st_size, "built": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / f"{args.family}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    if args.push:
        from mhtable import storage

        for f in (final, out / f"{args.family}.json"):
            sha = storage.push("cache", f, f"embeddings/{encoder.id}/{f.name}",
                               f"embeddings {encoder.id} {args.family} (data {args.revision})")
            print(f"pushed {f.name} -> mhtable-cache embeddings/{encoder.id}/ ({sha[:10]})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
