"""
Inference demo core (PLAN.md P9 "inference demo"): run one question through any selector and
record every step, so a notebook (notebooks/demo.py) can show it live or replay saved traces.

    demo = Demo(load_config("configs/experiments/eval.yaml"))       # nothing heavy is loaded yet
    specs = baseline_specs() + discover_policies()                   # every trained run on HF
    inst, gold = demo.questions("dev", "strict")[0]
    trace = demo.run(specs[0], inst, gold)                           # steps, sub-table, answer
    save_traces("traces.json", [trace]); load_traces("traces.json")  # replay mode

Extensible by design:
  * Selectors are `SelectorSpec`s. `discover_policies()` lists every `policy.pt` under
    `mhtable-runs/prajna/` (or a local runs folder), so runs trained later appear without code
    changes. The run id gives the label and tier; `parse_run_id` knows the naming scheme of every
    P9 script, and unknown names still load (label = run id).
  * The reader is built from `cfg.reader` and can be swapped at any time with
    `demo.set_reader({"name": "hf", "model": "..."})`. The token budget keeps being counted with
    the tokenizer the policies were trained with (pinned when the Demo is created), because the
    policies' look-ahead features are measured in those tokens.

Firewall: gold is used only to score the answer after the reader has answered, and by the oracle
selector, which is marked `uses_gold` (never a result, only the ceiling). A typed question has
no gold at all.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

import mhtable.encoders  # noqa: F401  (registers encoders)
import mhtable.policies  # noqa: F401  (registers policies)
import mhtable.readers  # noqa: F401  (registers readers)
from mhtable.config import Config
from mhtable.featurise import FeatConfig, Featuriser, question_text
from mhtable.metrics import answer_in_subtable, em_norm, format_answer
from mhtable.readers.base import CachedReader
from mhtable.registry import build
from mhtable.rollout import PolicyActor, pad_actions
from mhtable.types import Action, GoldRecord, Instance

TABLE_BASELINES = {"truncate": "rowid", "random_rows": "random"}  # same as evaluate.py


# ---------------------------------------------------------------- selectors
@dataclass(frozen=True)
class SelectorSpec:
    """One thing that picks a sub-table. kind: policy (a checkpoint) | heuristic | oracle | table
    (B1 truncate / B2 random_rows: whole tables, no steps)."""

    name: str
    label: str
    kind: str
    tier: str | None = None  # a policy is trained for one action tier; None = works for both
    path: str | None = None  # policy: "hf:prajna/<run>/policy.pt" or a local file
    uses_gold: bool = False  # oracle: follows the gold path (a ceiling, never a result)
    group: str = ""  # for ordering / grouping in the UI: baseline | bc | grid | process | bigstep | scratch | other


def baseline_specs() -> list[SelectorSpec]:
    return [SelectorSpec("oracle", "Oracle (gold path: ceiling)", "oracle", uses_gold=True, group="baseline"),
            SelectorSpec("truncate", "B1 whole tables, truncated", "table", group="baseline"),
            SelectorSpec("random_rows", "B2 random rows", "table", group="baseline"),
            SelectorSpec("heuristic", "Heuristic (cosine, no learning)", "heuristic", group="baseline")]


_RUN_PATTERNS = [  # (regex, group, label template); named groups: algo, rew, tier, rest
    (r"^bc-bge-(?P<tier>strict|relaxed)-s\d+$", "bc", "BC"),
    (r"^grpoproc-(?P<rew>r[01])-(?P<tier>strict|relaxed)-lr(?P<lr>.+?)-lam(?P<lam>[^-]+)-s\d+$", "process",
     "Process GRPO {REW} λ{lam}"),
    (r"^(?P<algo>ppo|grpo)-(?P<rew>r[01])-(?P<tier>strict|relaxed)-(?P<set>big|bigx)-s\d+$", "bigstep",
     "{ALGO} {REW} ({set})"),
    (r"^(?P<algo>ppo|grpo)-(?P<rew>r[01])-(?P<tier>strict|relaxed)-cov(?P<cov>[^-]+)-s\d+$", "coverage",
     "{ALGO} {REW} coverage {cov}"),
    (r"^(?P<algo>ppo|grpo)-(?P<rew>r[01])-(?P<tier>strict|relaxed)-scratch-s\d+$", "scratch",
     "{ALGO} {REW} from scratch"),
    (r"^(?P<algo>ppo|grpo)-(?P<rew>r[01])-(?P<tier>strict|relaxed)-lr(?P<lr>.+?)-s\d+$", "grid",  # lr may hold "-"
     "{ALGO} {REW} lr{lr}"),
]


def parse_run_id(run: str) -> tuple[str, str, str | None]:
    """run id -> (label, group, tier). Unknown names: (run id, "other", tier if visible)."""
    for pat, group, tmpl in _RUN_PATTERNS:
        m = re.match(pat, run)
        if m:
            d = m.groupdict()
            d.update(ALGO=(d.get("algo") or "").upper(), REW=(d.get("rew") or "").upper())
            return tmpl.format(**d), group, d.get("tier")
    tier = next((t for t in ("strict", "relaxed") if f"-{t}-" in f"-{run}-"), None)
    return run, "other", tier


def policy_spec(run: str, path: str) -> SelectorSpec:
    label, group, tier = parse_run_id(run)
    return SelectorSpec(run, label, "policy", tier=tier, path=path, group=group)


def discover_policies(local_dir: str | Path | None = None, hf_prefix: str = "prajna/",
                      skip=("-smoke", "g8")) -> list[SelectorSpec]:
    """Every `policy.pt` in a local runs folder (`<dir>/<run>/policy.pt`) or, by default, under
    `hf_prefix` in the mhtable-runs repo. Smoke and G-8 runs are skipped."""
    runs: dict[str, str] = {}
    if local_dir is not None:
        for p in sorted(Path(local_dir).glob("*/policy.pt")):
            runs[p.parent.name] = str(p)
    else:
        from mhtable import storage

        for f in storage.api().list_repo_files(storage.repo_id("runs"), repo_type=storage.repo_type("runs")):
            if f.startswith(hf_prefix) and f.endswith("/policy.pt"):
                run = f[len(hf_prefix):].split("/")[0]
                runs[run] = "hf:" + f
    order = {"bc": 0, "grid": 1, "process": 2, "bigstep": 3, "scratch": 4, "other": 5}
    specs = [policy_spec(r, p) for r, p in runs.items() if not any(s in r for s in skip)]
    return sorted(specs, key=lambda s: (order.get(s.group, 9), s.tier or "", s.label))


# ---------------------------------------------------------------- traces
@dataclass
class Candidate:
    text: str
    prob: float
    rows: int  # look-ahead: rows of the relation if this action is taken
    chosen: bool = False


@dataclass
class StepTrace:
    t: int
    kind: str  # SEED / ANCHOR / JOIN / STOP
    text: str  # the action in plain English
    rows_after: int
    sql_after: str  # SQL of the intermediate relation after this step ("" for STOP)
    n_candidates: int
    candidates: list[Candidate] = field(default_factory=list)  # top-k with probabilities (policies only)


@dataclass
class Trace:
    selector: str
    label: str
    tier: str
    instance_id: str
    source: str
    db_id: str
    question: str
    steps: list[StepTrace]
    columns: list[str]
    subtable: str
    tokens: int
    budget: int
    rows_shown: int
    rows_total: int
    overflow: bool
    answer: str
    reader: str
    seconds: float
    uses_gold: bool = False
    gold_answer: str | None = None  # filled only after the reader has answered; None for a typed question
    em: float | None = None
    answer_in_subtable: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Trace":
        d = dict(d)
        d["steps"] = [StepTrace(**{**s, "candidates": [Candidate(**c) for c in s.get("candidates", [])]})
                      for s in d["steps"]]
        return cls(**d)


def save_traces(path: str | Path, traces: list[Trace], meta: dict | None = None) -> None:
    Path(path).write_text(json.dumps({"meta": meta or {}, "traces": [t.to_dict() for t in traces]}, indent=1,
                                     default=str), encoding="utf-8")


def load_traces(path: str | Path) -> tuple[list[Trace], dict]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Trace.from_dict(t) for t in d["traces"]], d.get("meta", {})


# ---------------------------------------------------------------- plain-English actions
def _vals(values) -> str:
    return ", ".join(repr(v) if isinstance(v, str) else str(v) for v in values)


_OPS = {"=": "is", "<": "is less than", ">": "is greater than", "<=": "is at most", ">=": "is at least",
        "!=": "is not", "like": "matches", "not_like": "does not match", "in": "is one of",
        "not_in": "is none of", "between": "is between", "is": "is empty", "not_is": "is not empty"}


def describe(a: Action) -> str:
    """An action as one plain sentence (for the audience; the SQL is shown next to it)."""
    if a.kind == "STOP":
        return "Stop and hand the sub-table to the reader"
    if a.kind == "SEED":
        return f"Start from table `{a.table}`"
    if a.kind == "ANCHOR":
        an = a.anchor
        op = _OPS.get(an.op, an.op)
        vals = " and ".join(map(repr, an.values)) if an.op == "between" else _vals(an.values)
        return f"Keep rows where `{an.column}` {op}{' ' + vals if an.values else ''}"
    if a.kind == "JOIN":
        on = " and ".join(f"`{x}` = `{y}`" for x, y in (a.edge or ()))
        return f"Join table `{a.table}` on {on}"
    return a.kind


# ---------------------------------------------------------------- the session
class Demo:
    """Lazily builds what each request needs: one environment per tier, the embedding cache,
    the reader, and every selector, all cached for the next question."""

    def __init__(self, cfg: Config, data_dir=None, index_dir=None, reader=None, emb_cache=None,
                 live_encoder: bool = True, envs: dict | None = None):
        from mhtable.train import env_tokenizer

        # pin the budget tokenizer to the training one, so swapping the reader does not change features
        self.cfg = cfg.with_overrides({"env.tokenizer": env_tokenizer(cfg)})
        self.data_dir, self.index_dir = data_dir, index_dir
        self._reader = CachedReader(reader) if reader is not None else None
        self._cache, self.live_encoder = emb_cache, live_encoder
        self._envs = dict(envs or {})  # tier -> TableEnv (tests inject one; normally built on demand)
        self._feats, self._splits, self._actors = {}, {}, {}

    # -- environment, data, features
    def env(self, tier: str):
        if tier not in self._envs:
            from mhtable.train import build_env

            self._envs[tier] = build_env(self.cfg, self.data_dir, self.index_dir, tier=tier)
        return self._envs[tier]

    def questions(self, split: str, tier: str, sources=None) -> list[tuple[Instance, GoldRecord]]:
        key = (split, tier, tuple(sources or self.cfg.data.sources))
        if key not in self._splits:
            from mhtable.dataset import load_split

            self._splits[key] = load_split(self.env(tier).data, split, tier, key[2])
        return self._splits[key]

    def emb_cache(self):
        if self._cache is None:
            from mhtable.encoders import EmbCache
            from mhtable.train import resolve_cache_paths

            enc = dict(self.cfg.encoder)
            path = enc.pop("cache", None)
            self._cache = EmbCache.load(resolve_cache_paths(path)) if path else EmbCache.live(build("encoder", enc))
        return self._cache

    def _ensure_encoder(self) -> None:
        """Typed questions (and their cell values) are not in the pre-built cache: encode misses
        live with the configured encoder (bge-m3), loaded on first need."""
        cache = self.emb_cache()
        if cache.encoder is None and self.live_encoder:
            enc = {k: v for k, v in dict(self.cfg.encoder).items() if k != "cache"}
            cache.encoder = build("encoder", enc)

    def featuriser(self, tier: str) -> Featuriser:
        if tier not in self._feats:
            hdr = bool(self.cfg.get_path("featurise.header_ablation", False))
            self._feats[tier] = Featuriser(self.env(tier), self.emb_cache(), FeatConfig(header_ablation=hdr))
        return self._feats[tier]

    # -- reader (swappable)
    @property
    def reader(self) -> CachedReader:
        if self._reader is None:
            self._reader = CachedReader(build("reader", self.cfg.reader))
        return self._reader

    def set_reader(self, overrides: dict | None = None, reader=None) -> str:
        """Swap the LLM: `overrides` are merged into `cfg.reader` (e.g. {"name": "hf", "model":
        "Qwen/Qwen2.5-14B-Instruct"}), or pass a ready reader object. Returns the new reader id."""
        if self._reader is not None:
            close = getattr(self._reader.reader, "close", None)
            if callable(close):
                close()
        self._reader = None
        if reader is not None:
            self._reader = CachedReader(reader)
        elif overrides:
            self.cfg = self.cfg.with_overrides({f"reader.{k}": v for k, v in overrides.items()})
        if self._reader is None:
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return self.reader.id

    # -- selectors
    def actor(self, spec: SelectorSpec, tier: str, golds: dict | None = None):
        if spec.kind == "oracle":
            from mhtable.selectors.oracle import OracleSelector

            return OracleSelector(golds or {})
        key = (spec.name, tier)
        if key not in self._actors:
            if spec.kind == "heuristic":
                from mhtable.selectors.heuristic import HeuristicSelector

                self._actors[key] = HeuristicSelector(self.env(tier), self.emb_cache())
            elif spec.kind == "policy":
                from mhtable.train import load_checkpoint

                path = spec.path
                if path.startswith("hf:"):
                    from mhtable import storage

                    path = storage.fetch("runs", path[3:])
                policy, ck = load_checkpoint(path)
                feat = self.featuriser(tier)
                if ck["spec"] != feat.spec:
                    raise ValueError(f"{spec.name} was trained with features {ck['spec']}, the demo gives {feat.spec}")
                policy.eval()
                self._actors[key] = PolicyActor(policy, feat)
            else:
                raise ValueError(f"no actor for selector kind {spec.kind!r}")
        return self._actors[key]

    # -- questions typed by the audience
    @staticmethod
    def typed_instance(base: Instance, question: str) -> Instance:
        """A new question over the same database (and tables) as `base`; it has no gold."""
        n = abs(hash(question)) % 10**8
        return Instance(f"typed:{base.db_id}:{n}", base.source, base.db_ref, base.db_id, question.strip(),
                        tables=list(base.tables), category="", tier=dict(base.tier))

    # -- one question, one selector
    @torch.no_grad()
    def run(self, spec: SelectorSpec, inst: Instance, gold: GoldRecord | None = None, tier: str | None = None,
            top_k: int = 5) -> Trace:
        tier = tier or spec.tier or "strict"
        if spec.tier and spec.tier != tier:
            raise ValueError(f"{spec.name} is a {spec.tier}-tier policy, asked for {tier}")
        if spec.uses_gold and gold is None:
            raise ValueError(f"{spec.label} follows the gold path: it needs a dataset question, not a typed one")
        if inst.instance_id.startswith("typed:"):
            self._ensure_encoder()
        env, t0 = self.env(tier), time.time()
        steps: list[StepTrace] = []
        if spec.kind == "table":
            m = env.materialise_tables(inst, order=TABLE_BASELINES[spec.name], seed=int(self.cfg.seed))
            final = None
        else:
            actor = self.actor(spec, tier, {inst.instance_id: gold} if gold is not None else None)
            s = env.reset(inst)
            while True:
                cands = env.candidate_actions(s)
                probs = self._probs(actor, s, cands) if isinstance(actor, PolicyActor) else None
                idx = int(np.argmax(probs)) if probs is not None else actor.act([s], [cands], "argmax")[0].action_idx
                a = cands[idx]
                nxt, done = env.step(s, a)
                shown = []
                if probs is not None:
                    order = np.argsort(-probs)[:top_k]
                    if idx not in order:
                        order = np.append(order[: top_k - 1], idx)
                    shown = [Candidate(describe(cands[j]), float(probs[j]), int(cands[j].lookahead_rows), bool(j == idx))
                             for j in map(int, order)]
                steps.append(StepTrace(s.t, a.kind, describe(a), int(nxt.n_rows),
                                       "" if a.kind == "STOP" else nxt.relation_sql, len(cands), shown))
                s = nxt
                if done:
                    break
            final = s
            m = env.materialise(final)
        answer = ""
        if m.rows_shown > 0:
            meta = None
            if getattr(self.reader, "needs_gold", False):  # the mock reader (tests only)
                if gold is None:
                    raise ValueError("the mock reader needs gold; use a real reader for typed questions")
                if final is not None:
                    from mhtable.sufficiency import sufficient

                    suff = sufficient(env, final, m, gold, {})
                else:
                    suff = bool(answer_in_subtable(m.text, gold.gold_answer))
                meta = [{"sufficient": suff, "gold_answer": gold.gold_answer}]
            answers, _ = self.reader.read_batch([question_text(inst)], [m.text], meta)
            answer = answers[0]
        tr = Trace(spec.name, spec.label, tier, inst.instance_id, inst.source, inst.db_id, inst.question, steps,
                   list(m.columns), m.text, int(m.tokens), int(env.cfg.budget_tokens), int(m.rows_shown),
                   int(m.rows_total), bool(m.overflow), answer, self.reader.id, round(time.time() - t0, 2),
                   uses_gold=spec.uses_gold)
        if gold is not None:  # scored only now, after the reader has answered
            tr.gold_answer = format_answer(gold.gold_answer)
            tr.em = float(em_norm(answer, gold.gold_answer))
            tr.answer_in_subtable = float(answer_in_subtable(m.text, gold.gold_answer))
        return tr

    @staticmethod
    def _probs(actor: PolicyActor, s, cands: list[Action]) -> np.ndarray:
        """The policy's full distribution over this state's candidates (argmax = its greedy pick)."""
        st, af = actor.featuriser(s, cands)
        policy = actor.policy
        device = next(policy.parameters()).device
        acts, mask = pad_actions([af])
        logits, _ = policy(torch.from_numpy(st[None]).to(device), acts.to(device), mask.to(device))
        return torch.softmax(logits[0, : len(cands)].float(), dim=-1).cpu().numpy()

    def close(self) -> None:
        for env in self._envs.values():
            env.close()
        self._envs.clear()


# ---------------------------------------------------------------- results table (all runs)
def results_table(runs_prefix: str = "prajna/", local_dir: str | Path | None = None) -> list[dict]:
    """Test EM / tokens of every `t-<run>-test/result.json` (HF mhtable-runs or a local folder),
    one row per (run, tier). New runs appear automatically."""
    rows = []
    if local_dir is not None:
        files = {p.parent.name: p for p in Path(local_dir).glob("t-*-test/result.json")}
    else:
        from mhtable import storage

        names = [f for f in storage.api().list_repo_files(storage.repo_id("runs"), repo_type=storage.repo_type("runs"))
                 if f.startswith(runs_prefix + "t-") and f.endswith("-test/result.json")]
        files = {f.split("/")[-2]: storage.fetch("runs", f) for f in names}
    for name, path in sorted(files.items()):
        res = json.loads(Path(path).read_text(encoding="utf-8"))
        run = name[2:-5]  # t-<run>-test
        label, group, _ = parse_run_id(run)
        if run in {s.name for s in baseline_specs()}:
            label, group = next((s.label, s.group) for s in baseline_specs() if s.name == run), "baseline"
        elif run.startswith("bc-"):
            label, group = "BC", "bc"
        for tier, v in res.get("tiers", {}).items():
            rows.append({"run": run, "label": label, "group": group, "tier": tier, "n": v.get("n"),
                         "em": v.get("em"), "set_f1": v.get("set_f1"), "path_match": v.get("path_match"),
                         "answer_in_subtable": v.get("answer_in_subtable"), "tokens": v.get("tokens")})
    return rows
