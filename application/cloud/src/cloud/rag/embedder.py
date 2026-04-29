"""Embedding model wrapper for the ChPL corpus.

Production uses `sdadas/mmlw-roberta-large` (1024-dim, ~700 MB,
Polish-tuned via knowledge distillation). Tests use `_FakeEmbedder`
which produces deterministic 1024-dim hash-based vectors so we never
download the model in CI.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Protocol

EMBEDDING_DIM = 1024


class Embedder(Protocol):
    """Minimum surface for the retriever."""

    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


@dataclass
class _RealEmbedder:
    """sentence-transformers wrapper around `sdadas/mmlw-roberta-large`."""

    model_id: str = "sdadas/mmlw-roberta-large"
    dim: int = EMBEDDING_DIM

    @property
    def name(self) -> str:
        return self.model_id

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        model = _get_model(self.model_id)
        vectors = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        model = _get_model(self.model_id)
        vector = model.encode([text], normalize_embeddings=True, convert_to_numpy=True)
        return vector[0].tolist()


@lru_cache(maxsize=1)
def _get_model(model_id: str):  # type: ignore[no-untyped-def]
    """Lazy sentence-transformers model load. Only called outside tests."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_id)


@dataclass
class _FakeEmbedder:
    """Deterministic hash-based embedder for tests.

    Produces 1024-dim vectors derived from SHA-256 of the input. Same
    input always yields the same vector. NOT semantic — adjacent words
    will give wildly different embeddings — but that's fine for tests
    that only check determinism + shape.
    """

    dim: int = EMBEDDING_DIM
    name: str = "fake-embedder"
    _cache: dict[str, list[float]] = field(default_factory=dict)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        if text in self._cache:
            return self._cache[text]
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        # Repeat digest bytes to fill `dim` floats; map [0,255] → [-1,1].
        floats: list[float] = []
        i = 0
        while len(floats) < self.dim:
            byte = digest[i % len(digest)]
            floats.append((byte - 128) / 128.0)
            i += 1
        self._cache[text] = floats
        return floats
