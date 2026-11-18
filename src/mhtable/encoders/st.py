"""
sentence-transformers encoder — bge-m3 by default (report E.1: frozen bge-m3, 1024-d).

    encoder:
      name: st
      model: BAAI/bge-m3
      batch_size: 256
      max_seq_length: 256     # questions and column names are short; long cells are cut
      dtype: float16          # GPU only; CPU always runs float32

Only `build_cache` builds this encoder (on a GPU); training then loads the frozen cache and
never imports sentence-transformers. The model fixes the dimension: a `dim` key inherited from
configs/base.yaml (meant for the `hash` encoder) is ignored.
"""

from __future__ import annotations

import numpy as np

from mhtable.encoders.base import Encoder
from mhtable.registry import register


@register("encoder", "st")
class STEncoder(Encoder):
    def __init__(self, model: str = "BAAI/bge-m3", batch_size: int = 256, max_seq_length: int = 256,
                 device: str | None = None, dtype: str = "float16", dim: int | None = None):
        import torch
        from sentence_transformers import SentenceTransformer

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        kw = {"dtype": getattr(torch, dtype)} if device != "cpu" and dtype else {}
        self.st = SentenceTransformer(model, device=device, model_kwargs=kw)
        self.st.max_seq_length = max_seq_length
        self.batch_size = batch_size
        dim_of = getattr(self.st, "get_embedding_dimension", None) or self.st.get_sentence_embedding_dimension
        self.dim = int(dim_of())
        self.id = model.rstrip("/").split("/")[-1].lower()  # "bge-m3": the cache folder name

    def encode(self, texts: list[str], progress: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        v = self.st.encode(texts, batch_size=self.batch_size, normalize_embeddings=True, convert_to_numpy=True,
                           show_progress_bar=progress)
        return np.asarray(v, dtype=np.float32)
