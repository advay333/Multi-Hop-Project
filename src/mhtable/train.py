"""
Training entry point (DESIGN.md §6.6).

    python -m mhtable.train --config configs/experiments/bc.yaml \
        [--data-dir <local copy of the data repo>] [--index-dir <local copy with index/>] \
        [--out runs] [--run-id NAME] [--limit N] [key=value ...]

Without --data-dir, data comes from HuggingFace at `data.revision`. Everything swappable is
chosen in the config: encoder, policy, reader, algorithm, tier. A run writes
`<out>/<run_id>/`: `config.yaml` (exact settings + code revision), `metrics.jsonl` (one line per
evaluation), `policy.pt` (best checkpoint), `result.json` (final numbers). Nothing is uploaded.

P7 implements `algo.name: bc` (behaviour cloning). The RL arms (P8 PPO, P9 GRPO) plug into the
same setup and start from a BC checkpoint (`load_checkpoint`).

Dev evaluation is the shared rollout in argmax mode with gold used for metrics only:
  path_match   join-path exact accuracy — needs no reader at all (the BC gate metric)
  sufficient   the sub-table holds what the answer needs (row-level, see sufficiency.py)
  em           answer exact match — with the mock reader this equals `sufficient`
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import mhtable.algos.bc  # noqa: F401  (registers "bc")
import mhtable.encoders  # noqa: F401  (registers encoders)
import mhtable.policies  # noqa: F401  (registers policies)
import mhtable.readers  # noqa: F401  (registers readers)
from mhtable.algos.bc import demonstrations
from mhtable.config import Config, load_config
from mhtable.dataset import load_split
from mhtable.encoders.cache import EmbCache
from mhtable.env.actions import ActionConfig
from mhtable.env.env import EnvConfig, EnvData, TableEnv
from mhtable.featurise import FeatConfig, Featuriser
from mhtable.readers.base import CachedReader
from mhtable.registry import build
from mhtable.rollout import PolicyActor, collect


# ---------------------------------------------------------------- setup
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10,
                              cwd=Path(__file__).parent).stdout.strip() or None
    except Exception:
        return None


def git_dirty() -> bool | None:
    """True if the checkout has uncommitted changes (the recorded commit is then not the code that ran)."""
    try:
        out = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, timeout=10,
                             cwd=Path(__file__).parent)
        return bool(out.stdout.strip()) if out.returncode == 0 else None
    except Exception:
        return None


def build_env(cfg: Config, data_dir=None, index_dir=None) -> TableEnv:
    e = cfg.env
    actions = ActionConfig(horizon=e.horizon, tier=e.tier, edges=e.edges, edge_threshold=e.edge_threshold)
    data = EnvData(data_dir, index_dir, revision=cfg.get_path("data.revision"))
    return TableEnv(EnvConfig(actions=actions, budget_tokens=e.budget_tokens), data)


def build_featuriser(cfg: Config, env: TableEnv) -> Featuriser:
    enc = dict(cfg.encoder)
    cache_path = enc.pop("cache", None)  # a pre-built embedding cache (P4); otherwise encode on first use
    cache = EmbCache.load(cache_path) if cache_path else EmbCache.live(build("encoder", enc))
    return Featuriser(env, cache, FeatConfig(header_ablation=bool(cfg.get_path("featurise.header_ablation", False))))


def save_checkpoint(path: Path, policy, spec: dict, cfg: Config, extra: dict | None = None) -> None:
    torch.save({"state_dict": policy.state_dict(), "spec": spec, "policy": cfg.policy.to_dict(),
                "config": cfg.to_dict(), "extra": extra or {}}, path)


def load_checkpoint(path: str | Path):
    """-> (policy, checkpoint dict). Rebuilds the network from the stored policy section and spec."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    policy = build("policy", ck["policy"], spec=ck["spec"])
    policy.load_state_dict(ck["state_dict"])
    return policy, ck


# ---------------------------------------------------------------- evaluation
def evaluate_rollouts(env, actor, reader, items, chunk: int = 64) -> dict:
    """Greedy episodes on `items`; gold is used for metrics only. Overall numbers plus a
    breakdown by source and by number of gold tables."""
    golds = {i.instance_id: g for i, g in items}
    rows = []
    for k in range(0, len(items), chunk):
        part = [i for i, _ in items[k:k + chunk]]
        trajs = collect(env, actor, reader, part, mode="argmax", oracle_mode="metric_only",
                        golds={i.instance_id: golds[i.instance_id] for i in part})
        for inst, tr in zip(part, trajs):
            rows.append({"source": inst.source, "tables": f"{inst.n_gold_tables}_tables",
                         "path_match": float(tr.info["path_match"]), "sufficient": float(tr.info["sufficient"]),
                         "em": tr.info["r_ans"], "overflow": float(tr.info["overflow"]),
                         "steps": len(tr.steps), "tokens": tr.subtable_tokens})
    keys = ("path_match", "sufficient", "em", "overflow", "steps", "tokens")

    def mean(rs):
        return {"n": len(rs), **{k: round(sum(r[k] for r in rs) / len(rs), 4) for k in keys}} if rs else {"n": 0}

    out = mean(rows)
    for field in ("source", "tables"):
        groups = defaultdict(list)
        for r in rows:
            groups[r[field]].append(r)
        out[f"by_{field}"] = {g: mean(rs) for g, rs in sorted(groups.items())}
    return out


# ---------------------------------------------------------------- behaviour cloning
def run_bc(cfg: Config, env, feat, policy, algo, reader, run_dir: Path, limit: int | None, log) -> dict:
    tier, sources, seed = cfg.env.tier, tuple(cfg.data.sources), cfg.seed
    tcfg = cfg.get_path("train", Config({}))
    epochs, eval_every = int(tcfg.get("epochs", 30)), int(tcfg.get("eval_every", 5))
    n_train_eval = int(tcfg.get("eval_train_questions", 300))
    train_items = load_split(env.data, "train", tier, sources, limit, seed)
    dev_items = load_split(env.data, cfg.get_path("data.dev_split", "dev"), tier, sources, limit, seed)
    log(f"questions: train {len(train_items)}, dev {len(dev_items)} (tier {tier}, sources {list(sources)})")

    t0 = time.time()
    train_demos, train_stats = demonstrations(env, feat, train_items, progress=True)
    dev_demos, dev_stats = demonstrations(env, feat, dev_items, progress=True)
    log(f"demonstrations: train {train_stats}, dev {dev_stats} ({time.time() - t0:.0f}s)")
    if not train_demos:
        raise SystemExit("no reachable gold trajectories to imitate")
    train_batch = algo.compute_advantages(train_demos)
    dev_batch = algo.compute_advantages(dev_demos) if dev_demos else None
    del train_demos, dev_demos

    actor = PolicyActor(policy, feat)
    train_sub = list(train_items)
    random.Random(seed).shuffle(train_sub)
    train_sub = train_sub[:n_train_eval]
    best, history = None, []
    before = evaluate_rollouts(env, actor, reader, dev_items) if dev_items else {}
    log(f"untrained policy on dev: {json.dumps({k: before.get(k) for k in ('path_match', 'sufficient', 'em')})}")
    for epoch in range(1, epochs + 1):
        stats = algo.update(train_batch)
        if epoch % eval_every and epoch != epochs:
            continue
        rec = {"epoch": epoch, "train": stats, "reader_calls": reader.calls}
        if dev_batch is not None:
            rec["dev_teacher_forced"] = algo.evaluate(dev_batch)
        rec["dev"] = evaluate_rollouts(env, actor, reader, dev_items) if dev_items else {}
        history.append(rec)
        with open(run_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        score = rec["dev"].get("path_match", -stats["loss"])
        log(f"epoch {epoch}: train loss {stats['loss']:.4f} step-acc {stats['step_acc']:.3f} | dev "
            f"path {rec['dev'].get('path_match')} sufficient {rec['dev'].get('sufficient')} "
            f"teacher-forced step-acc {rec.get('dev_teacher_forced', {}).get('step_acc')}")
        if best is None or score > best["score"]:
            best = {"score": score, "epoch": epoch, "dev": rec["dev"], "dev_teacher_forced": rec.get("dev_teacher_forced")}
            save_checkpoint(run_dir / "policy.pt", policy, feat.spec, cfg, {"epoch": epoch, "dev": rec["dev"]})

    policy, _ = load_checkpoint(run_dir / "policy.pt")  # report the best checkpoint
    actor = PolicyActor(policy, feat)
    result = {"algo": "bc", "tier": tier, "sources": list(sources), "seed": seed, "reader": reader.id,
              "encoder": cfg.encoder.to_dict(), "parameters": policy.n_params(),
              "demonstrations": {"train": train_stats, "dev": dev_stats},
              "best_epoch": best["epoch"], "dev_untrained": before, "dev": best["dev"],
              "dev_teacher_forced": best["dev_teacher_forced"],
              "train_subset": evaluate_rollouts(env, actor, reader, train_sub) if train_sub else {},
              "reader_calls": reader.calls, "minutes": round((time.time() - t0) / 60, 1)}
    return result


# ---------------------------------------------------------------- entry point
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--data-dir", type=Path, default=None, help="local copy of the data repo (default: HuggingFace)")
    p.add_argument("--index-dir", type=Path, default=None, help="folder holding index/ (default: --data-dir)")
    p.add_argument("--out", type=Path, default=Path("runs"))
    p.add_argument("--run-id", default=None)
    p.add_argument("--limit", type=int, default=None, help="use a random subset of each split (smoke runs)")
    p.add_argument("overrides", nargs="*", help="config overrides, e.g. env.tier=strict algo.lr=1e-3")
    args = p.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    seed_everything(cfg.seed)
    run_id = args.run_id or f"{cfg.algo.name}-{cfg.env.tier}-s{cfg.seed}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = args.out / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    run_info = {"run.id": run_id, "run.code": git_sha(), "run.code_dirty": git_dirty(),
                "run.data_dir": str(args.data_dir) if args.data_dir else None,
                "run.index_dir": str(args.index_dir) if args.index_dir else None, "run.limit": args.limit}
    (run_dir / "config.yaml").write_text(cfg.with_overrides(run_info).to_yaml(), encoding="utf-8")

    def log(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    env = build_env(cfg, args.data_dir, args.index_dir)
    feat = build_featuriser(cfg, env)
    policy = build("policy", cfg.policy, spec=feat.spec)
    reader = CachedReader(build("reader", cfg.reader), run_dir / "reader_cache.jsonl")
    algo = build("algo", dict(cfg.algo, seed=cfg.seed) if cfg.algo.name == "bc" else cfg.algo, policy=policy)
    log(f"run {run_id}: policy {cfg.policy.name} ({policy.n_params():,} parameters), reader {reader.id}")
    try:
        if cfg.algo.name == "bc":
            result = run_bc(cfg, env, feat, policy, algo, reader, run_dir, args.limit, log)
        else:
            raise SystemExit(f"algo {cfg.algo.name!r}: the RL training loop is added in P8/P9")
    finally:
        env.close()
    (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
