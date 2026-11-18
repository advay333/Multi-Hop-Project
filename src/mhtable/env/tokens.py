"""
Token counting for the sub-table budget B (report C.3; PLAN.md P4).

    CharEstimate(3.5)        len(text) / 3.5 — the default; keeps mock-reader runs and the unit
                             tests unchanged
    HFTokenCounter(model)    the reader's own tokenizer (transformers), so "fits in B tokens"
                             means exactly that for the reader that will read it

Selected by `env.tokenizer` (null | a model name | "reader" = the reader's model; resolved in
train.build_env). Only `materialise` uses it: look-ahead token estimates stay character-based,
since they are a cheap cardinality feature computed for every candidate.
"""

from __future__ import annotations


class CharEstimate:
    exact = False

    def __init__(self, chars_per_token: float = 3.5):
        self.chars_per_token = chars_per_token
        self.id = f"chars/{chars_per_token}"

    def count(self, text: str) -> float:
        return len(text) / self.chars_per_token

    def count_many(self, texts: list[str]) -> list[float]:
        return [len(t) / self.chars_per_token for t in texts]


class HFTokenCounter:
    exact = True

    def __init__(self, model: str):
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model)
        self.id = f"tok/{model}"

    def count(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False)["input_ids"]) if text else 0

    def count_many(self, texts: list[str]) -> list[int]:
        if not texts:
            return []
        return [len(ids) for ids in self.tok(texts, add_special_tokens=False)["input_ids"]]


def make_counter(tokenizer: str | None, chars_per_token: float = 3.5):
    return HFTokenCounter(tokenizer) if tokenizer else CharEstimate(chars_per_token)
