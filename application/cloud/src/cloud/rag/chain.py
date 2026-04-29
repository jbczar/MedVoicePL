"""Helpers wiring `Embedder` + `ChPLRetriever` for orchestrator use.

The orchestrator's `compose_suggestions` node calls `retrieve_for_drug`
per candidate to fetch relevant ChPL excerpts. Pure helper layer; the
LLM ↔ RAG fusion happens inside the LangGraph node.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from cloud.rag.embedder import Embedder, _RealEmbedder
from cloud.rag.retriever import (
    ChPLExcerpt,
    ChPLRetriever,
    _FakeRetriever,
    build_qdrant_retriever,
)


@dataclass
class ChPLChain:
    """Bundled embedder + retriever — what the orchestrator depends on."""

    embedder: Embedder
    retriever: ChPLRetriever

    def retrieve_for_drug(self, *, atc_code: str, drug_name: str, k: int = 3) -> list[ChPLExcerpt]:
        """Compose a query from ATC + drug name. ATC alone is too sparse
        (the embedder doesn't know that `B01AF02` means apixaban); drug
        name + ATC together gets to the right doc.
        """
        query = f"{drug_name} {atc_code}".strip()
        return self.retriever.retrieve(query, k=k)


def load_chpl_excerpts(corpus_dir: Path) -> list[ChPLExcerpt]:
    """Load `*.md` files from `corpus_dir` as `ChPLExcerpt` records.

    Each file is expected to have YAML frontmatter with `atc_code`,
    `drug_name`, optional `source`. The body is the excerpt text.
    Filenames follow `<atc>_<drug>.md` convention but the frontmatter
    is authoritative.
    """
    if not corpus_dir.is_dir():
        return []

    excerpts: list[ChPLExcerpt] = []
    for md_path in sorted(corpus_dir.glob("*.md")):
        if md_path.name.lower() == "readme.md":
            continue
        meta, body = _split_frontmatter(md_path.read_text(encoding="utf-8"))
        atc_code = str(meta.get("atc_code", "")).strip().upper()
        drug_name = str(meta.get("drug_name", "")).strip()
        source = str(meta.get("source", "")).strip()
        if not atc_code or not drug_name:
            continue
        excerpts.append(
            ChPLExcerpt(
                atc_code=atc_code,
                drug_name=drug_name,
                text=body.strip(),
                source=source,
            )
        )
    return excerpts


def _split_frontmatter(text: str) -> tuple[dict[str, object], str]:
    """Minimal `---`-delimited YAML frontmatter parser. Returns ({}, text)
    when no frontmatter is present.
    """
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", maxsplit=2)
    if len(parts) < 3:
        return {}, text
    meta = yaml.safe_load(parts[1]) or {}
    if not isinstance(meta, dict):
        meta = {}
    return meta, parts[2].lstrip("\n")


def build_chain(
    *,
    corpus_dir: Path,
    qdrant_path: Path | None = None,
    embedder: Embedder | None = None,
) -> ChPLChain:
    """Construct a production `ChPLChain` from disk.

    `embedder=None` → `_RealEmbedder` (sdadas/mmlw-roberta-large).
    Tests inject `_FakeEmbedder` + a pre-built `_FakeRetriever` and
    bypass this factory entirely.
    """
    embedder = embedder or _RealEmbedder()
    excerpts = load_chpl_excerpts(corpus_dir)
    retriever = build_qdrant_retriever(
        embedder=embedder, excerpts=excerpts, path=qdrant_path
    )
    return ChPLChain(embedder=embedder, retriever=retriever)


def build_fake_chain(excerpts: list[ChPLExcerpt]) -> ChPLChain:
    """Test convenience — `ChPLChain` backed by `_FakeRetriever`."""
    from cloud.rag.embedder import _FakeEmbedder

    return ChPLChain(embedder=_FakeEmbedder(), retriever=_FakeRetriever(excerpts=excerpts))


__all__ = [
    "ChPLChain",
    "build_chain",
    "build_fake_chain",
    "load_chpl_excerpts",
]
