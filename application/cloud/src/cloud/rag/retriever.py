"""ChPL excerpt retriever — Qdrant in-process.

The corpus is small (~12 hand-curated Polish ChPL excerpts in
`application/cloud/data/chpl_excerpts/<atc>_<name>.md`); the vector
store exists for query-time top-k retrieval per drug ATC.

Why Qdrant in-process:

* DD-1 in plan: ChromaDB is broken on Python 3.13 + Apple Silicon.
* Qdrant supports `:memory:` location AND on-disk; we use disk for
  persistence between agent runs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from cloud.rag.embedder import EMBEDDING_DIM, Embedder

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChPLExcerpt:
    """One Polish ChPL excerpt indexed in the retriever."""

    atc_code: str
    drug_name: str
    text: str
    source: str = ""
    score: float | None = None  # populated only on retrieval results


class ChPLRetriever(Protocol):
    """Minimum retriever surface needed by the orchestrator."""

    def retrieve(self, query: str, *, k: int = 3) -> list[ChPLExcerpt]: ...

    def __len__(self) -> int: ...


# ---------------------------------------------------------------------------
# Real Qdrant-backed retriever.
# ---------------------------------------------------------------------------


@dataclass
class _QdrantRetriever:
    """In-process Qdrant retriever.

    Build via `build_qdrant_retriever(embedder, excerpts, path)`. The
    constructor itself is bare — index population happens in the factory
    so failures (e.g. missing qdrant-client) surface there.
    """

    embedder: Embedder
    collection: str = "chpl_excerpts"
    path: str | None = None  # None → in-memory
    _client: object | None = None
    _excerpts: list[ChPLExcerpt] = field(default_factory=list)

    def retrieve(self, query: str, *, k: int = 3) -> list[ChPLExcerpt]:
        if self._client is None:
            return []
        from qdrant_client import models as qmodels  # type: ignore[import-untyped]

        vector = self.embedder.embed_query(query)
        # qdrant-client 1.10+: `.search()` removed in favor of
        # `.query_points()` which returns a QueryResponse with `.points`.
        response = self._client.query_points(  # type: ignore[attr-defined]
            collection_name=self.collection,
            query=vector,
            limit=k,
            search_params=qmodels.SearchParams(exact=True),
        )
        hits = getattr(response, "points", response)
        out: list[ChPLExcerpt] = []
        for hit in hits:
            payload = hit.payload or {}
            out.append(
                ChPLExcerpt(
                    atc_code=payload.get("atc_code", ""),
                    drug_name=payload.get("drug_name", ""),
                    text=payload.get("text", ""),
                    source=payload.get("source", ""),
                    score=float(hit.score),
                )
            )
        return out

    def __len__(self) -> int:
        return len(self._excerpts)


def build_qdrant_retriever(
    *,
    embedder: Embedder,
    excerpts: list[ChPLExcerpt],
    path: Path | str | None = None,
    collection: str = "chpl_excerpts",
) -> _QdrantRetriever:
    """Build a Qdrant retriever, populating it with `excerpts`.

    `path=None` uses Qdrant's `:memory:` mode (no disk). For production
    pass a directory path; Qdrant handles its own SQLite-style storage.
    """
    from qdrant_client import QdrantClient
    from qdrant_client import models as qmodels

    client_path = str(path) if path is not None else ":memory:"
    client = QdrantClient(path=client_path)

    # Re-create the collection — the corpus is ~12 docs so this is cheap.
    client.recreate_collection(
        collection_name=collection,
        vectors_config=qmodels.VectorParams(
            size=embedder.dim,
            distance=qmodels.Distance.COSINE,
        ),
    )

    if excerpts:
        vectors = embedder.embed_documents([e.text for e in excerpts])
        points = [
            qmodels.PointStruct(
                id=idx,
                vector=vec,
                payload={
                    "atc_code": e.atc_code,
                    "drug_name": e.drug_name,
                    "text": e.text,
                    "source": e.source,
                },
            )
            for idx, (e, vec) in enumerate(zip(excerpts, vectors, strict=True))
        ]
        client.upsert(collection_name=collection, points=points)

    log.info(
        "qdrant retriever built: collection=%s docs=%d path=%s",
        collection,
        len(excerpts),
        client_path,
    )
    retriever = _QdrantRetriever(
        embedder=embedder, collection=collection, path=client_path
    )
    retriever._client = client
    retriever._excerpts = list(excerpts)
    return retriever


# ---------------------------------------------------------------------------
# Test fixture — no Qdrant, no model.
# ---------------------------------------------------------------------------


@dataclass
class _FakeRetriever:
    """Naive retriever for tests: substring + ATC-prefix match.

    Avoids both Qdrant dep and embedding round-trip in unit tests.
    """

    excerpts: list[ChPLExcerpt] = field(default_factory=list)

    def retrieve(self, query: str, *, k: int = 3) -> list[ChPLExcerpt]:
        if not self.excerpts:
            return []
        q = query.casefold()
        scored: list[tuple[float, ChPLExcerpt]] = []
        for e in self.excerpts:
            score = 0.0
            if e.atc_code and e.atc_code.casefold() in q:
                score += 1.0
            if e.drug_name and e.drug_name.casefold() in q:
                score += 0.5
            if e.text and any(tok in e.text.casefold() for tok in q.split()):
                score += 0.1
            if score > 0:
                scored.append((score, e))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [e for _, e in scored[:k]]

    def __len__(self) -> int:
        return len(self.excerpts)


__all__ = [
    "EMBEDDING_DIM",
    "ChPLExcerpt",
    "ChPLRetriever",
    "_FakeRetriever",
    "_QdrantRetriever",
    "build_qdrant_retriever",
]
