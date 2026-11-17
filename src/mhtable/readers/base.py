"""
Reader interface and the caching wrapper (DESIGN.md §6.5, PLAN.md §6 rule 8).

A reader answers a question from a serialised sub-table. `CachedReader` wraps any reader with
  - a cache keyed by sha256(reader id + prompt) (readers decode greedily, so answers are fixed);
  - a call counter: only cache misses are reader calls, the unit all RL curves are plotted in;
  - an optional append-only shard file `reader/<reader>/<run_id>.jsonl` (DESIGN.md §4.4).
The real readers (`hf`, `vllm`) are added in P4; they implement only `id` and `read_batch`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from mhtable.readers.prompt import build_prompt


class Reader:
    id: str = "reader"  # part of every cache key
    needs_gold: bool = False  # True only for the debugging mock reader

    def read_batch(self, questions: list[str], subtables: list[str], meta: list[dict] | None = None) -> list[str]:
        """One answer string per (question, sub-table). `meta` is ignored by real readers and
        never enters the prompt."""
        raise NotImplementedError

    def cache_extra(self, meta: dict | None) -> str:
        """Extra cache-key text for readers whose answer depends on more than the prompt (mock)."""
        return ""


class CachedReader:
    def __init__(self, reader: Reader, shard_path: str | Path | None = None, preload: tuple = ()):
        self.reader = reader
        self.cache: dict[str, str] = {}
        self.calls = 0  # cache misses = real reader calls
        self.hits = 0
        self.shard_path = Path(shard_path) if shard_path else None
        for p in tuple(preload) + ((self.shard_path,) if self.shard_path and self.shard_path.exists() else ()):
            self.load_shard(p)

    @property
    def id(self) -> str:
        return self.reader.id

    @property
    def needs_gold(self) -> bool:
        return self.reader.needs_gold

    def load_shard(self, path: str | Path) -> None:
        with open(path, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                self.cache[d["key"]] = d["answer"]

    def key(self, question: str, subtable: str, meta: dict | None = None) -> str:
        text = self.reader.id + "\n" + build_prompt(question, subtable) + self.reader.cache_extra(meta)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def read_batch(self, questions: list[str], subtables: list[str],
                   meta: list[dict] | None = None) -> tuple[list[str], list[bool]]:
        """Answers, and for each whether it cost a reader call (False on a cache hit; identical
        prompts inside one batch are sent once)."""
        metas = meta or [None] * len(questions)
        keys = [self.key(q, s, m) for q, s, m in zip(questions, subtables, metas)]
        todo: dict[str, int] = {}
        for i, k in enumerate(keys):
            if k not in self.cache and k not in todo:
                todo[k] = i
        called = [False] * len(keys)
        if todo:
            idx = list(todo.values())
            answers = self.reader.read_batch([questions[i] for i in idx], [subtables[i] for i in idx],
                                             [metas[i] for i in idx] if meta else None)
            if len(answers) != len(idx):
                raise RuntimeError(f"reader {self.reader.id} returned {len(answers)} answers for {len(idx)} prompts")
            for i, ans in zip(idx, answers):
                self.cache[keys[i]] = ans
                called[i] = True
            if self.shard_path:
                self.shard_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.shard_path, "a", encoding="utf-8") as f:
                    for i in idx:
                        f.write(json.dumps({"key": keys[i], "answer": self.cache[keys[i]]}) + "\n")
        self.calls += len(todo)
        self.hits += len(keys) - len(todo)
        return [self.cache[k] for k in keys], called
