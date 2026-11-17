"""
Embedding cache: text -> vector, a dict lookup at training time (DESIGN.md §6.5).

Two modes:
  frozen   `EmbCache(dim, vectors)` — built once on a GPU (P4 `build_cache`) and loaded here.
           A miss on a required text is an error, never a silent re-encode.
  live     `EmbCache.live(encoder)` — encodes on first use; for unit tests and smoke runs with
           the cheap `hash` encoder.
Optional texts (anchor values that are not in the cell index, e.g. numbers taken from the
question) are looked up with `get(text, default=None)`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mhtable.encoders.base import Encoder

_MISSING = object()


class EmbCache:
    def __init__(self, dim: int, vectors: dict[str, np.ndarray] | None = None, encoder: Encoder | None = None):
        self.dim = dim
        self.vectors: dict[str, np.ndarray] = dict(vectors or {})
        self.encoder = encoder

    @classmethod
    def live(cls, encoder: Encoder) -> "EmbCache":
        return cls(encoder.dim, encoder=encoder)

    def __contains__(self, text: str) -> bool:
        return text in self.vectors

    def __len__(self) -> int:
        return len(self.vectors)

    def add(self, texts: list[str], vectors: np.ndarray) -> None:
        for t, v in zip(texts, vectors):
            self.vectors[t] = np.asarray(v, dtype=np.float32)

    def get(self, text: str, default=_MISSING):
        v = self.vectors.get(text)
        if v is not None:
            return v
        if self.encoder is not None:
            v = self.encoder.encode([text])[0]
            self.vectors[text] = v
            return v
        if default is not _MISSING:
            return default
        raise KeyError(f"embedding cache miss for {text[:80]!r}: rebuild the cache (texts come from "
                       f"featurise.required_texts)")

    def save(self, path: str | Path) -> None:
        keys = list(self.vectors)
        np.savez_compressed(path, keys=np.array(keys, dtype=object),
                            vectors=np.stack([self.vectors[k] for k in keys]) if keys else np.zeros((0, self.dim)))

    @classmethod
    def load(cls, path: str | Path) -> "EmbCache":
        z = np.load(path, allow_pickle=True)
        vecs = z["vectors"].astype(np.float32)
        return cls(vecs.shape[1] if len(vecs) else 0, dict(zip(z["keys"].tolist(), vecs)))
