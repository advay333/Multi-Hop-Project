"""
Text encoders (DESIGN.md §6.5). An encoder turns strings into fixed-size vectors; the policy
never fine-tunes it — embeddings are computed once and looked up from `EmbCache`.

`hash` is a training-free stand-in used by unit tests and laptop smoke runs: a bag of word and
character-trigram hashes. It carries lexical overlap only (no semantics). The real encoder
(bge-m3 via sentence-transformers) is added in P4 with the same interface.
"""

from __future__ import annotations

import hashlib

import numpy as np

from mhtable.env.text import tokens, trigrams
from mhtable.registry import register


class Encoder:
    id: str = "encoder"  # part of the embedding-cache path
    dim: int = 0

    def encode(self, texts: list[str]) -> np.ndarray:
        """[len(texts), dim] float32, rows L2-normalised."""
        raise NotImplementedError


def _bucket(feature: str, dim: int) -> tuple[int, float]:
    h = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()  # stable across processes
    n = int.from_bytes(h, "little")
    return n % dim, 1.0 if (n >> 62) & 1 else -1.0


@register("encoder", "hash")
class HashEncoder(Encoder):
    def __init__(self, dim: int = 64):
        self.dim = dim
        self.id = f"hash{dim}"

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            feats = [f"w:{t}" for t in tokens(text)] + [f"g:{g}" for g in trigrams(text)]
            for f in feats or [f"raw:{text}"]:
                j, sign = _bucket(f, self.dim)
                out[i, j] += sign
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-8)
