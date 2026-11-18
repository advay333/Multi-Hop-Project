"""
Embedding cache: text -> vector, a dict lookup at training time (DESIGN.md §6.5).

Two modes:
  frozen   `EmbCache.load(path)` — built once on a GPU (`python -m mhtable.encoders.build_cache`).
           A miss on a required text is an error, never a silent re-encode.
  live     `EmbCache.live(encoder)` — encodes on first use; for unit tests and smoke runs with
           the cheap `hash` encoder.
Optional texts (anchor values that are not in the cell index, e.g. numbers taken from the
question) are looked up with `get(text, default=None)`.

A loaded cache keeps one float16 matrix plus a text -> row dict (BIRD's cell values run into
the millions; one small array per text would cost several times the memory). `get` returns
float32. Files are `.npz` with `keys` (texts) and `vectors`; several files (e.g. one per family)
can be loaded together: `EmbCache.load([a, b])`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mhtable.encoders.base import Encoder

_MISSING = object()


class EmbCache:
    def __init__(self, dim: int, vectors: dict[str, np.ndarray] | None = None, encoder: Encoder | None = None):
        self.dim = dim
        self.vectors: dict[str, np.ndarray] = {k: np.asarray(v, dtype=np.float32) for k, v in (vectors or {}).items()}
        self.encoder = encoder
        self._rows: dict[str, int] = {}  # frozen block: text -> row of _matrix
        self._matrix = np.zeros((0, dim), dtype=np.float16)

    @classmethod
    def live(cls, encoder: Encoder) -> "EmbCache":
        return cls(encoder.dim, encoder=encoder)

    def __contains__(self, text: str) -> bool:
        return text in self.vectors or text in self._rows

    def __len__(self) -> int:
        return len(self.vectors) + len(self._rows)

    def texts(self) -> list[str]:
        return list(self._rows) + [t for t in self.vectors if t not in self._rows]

    def add(self, texts: list[str], vectors: np.ndarray) -> None:
        for t, v in zip(texts, vectors):
            self.vectors[t] = np.asarray(v, dtype=np.float32)

    def get(self, text: str, default=_MISSING):
        v = self.vectors.get(text)
        if v is not None:
            return v
        r = self._rows.get(text)
        if r is not None:
            return self._matrix[r].astype(np.float32)
        if self.encoder is not None:
            v = self.encoder.encode([text])[0]
            self.vectors[text] = v
            return v
        if default is not _MISSING:
            return default
        raise KeyError(f"embedding cache miss for {text[:80]!r}: rebuild the cache (texts come from "
                       f"featurise.required_texts)")

    def save(self, path: str | Path, dtype=np.float16, compress: bool = False) -> None:
        keys = self.texts()
        mat = (np.stack([self.get(k) for k in keys]) if keys else np.zeros((0, self.dim))).astype(dtype)
        (np.savez_compressed if compress else np.savez)(path, keys=np.array(keys, dtype=object), vectors=mat)

    @classmethod
    def load(cls, paths) -> "EmbCache":
        paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
        keys, mats = [], []
        for p in paths:
            z = np.load(p, allow_pickle=True)
            keys += z["keys"].tolist()
            mats.append(z["vectors"].astype(np.float16))
        dims = {m.shape[1] for m in mats if len(m)}
        if len(dims) > 1:
            raise ValueError(f"embedding files {paths} have different dimensions {sorted(dims)}")
        out = cls(dims.pop() if dims else 0)
        if mats:
            out._matrix = np.concatenate(mats) if len(mats) > 1 else mats[0]
            out._rows = {k: i for i, k in enumerate(keys)}  # a text in two files keeps the later row
        return out
