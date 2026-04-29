"""Tests for `cloud.rag.{embedder,retriever,chain}` — fakes only.

Avoids `sentence-transformers` model download AND avoids needing
`qdrant-client` to be initialized (the in-process Qdrant build is
covered by the Step 4 smoke command in CLAUDE.md, not by unit tests).
"""

from __future__ import annotations

from pathlib import Path

from cloud.rag.chain import (
    ChPLChain,
    _split_frontmatter,
    build_fake_chain,
    load_chpl_excerpts,
)
from cloud.rag.embedder import EMBEDDING_DIM, _FakeEmbedder
from cloud.rag.retriever import ChPLExcerpt, _FakeRetriever

# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------


class TestFakeEmbedder:
    def test_dim_is_1024(self) -> None:
        emb = _FakeEmbedder()
        v = emb.embed_query("propranolol")
        assert len(v) == EMBEDDING_DIM
        assert emb.dim == 1024

    def test_deterministic(self) -> None:
        emb = _FakeEmbedder()
        a = emb.embed_query("propranolol")
        b = emb.embed_query("propranolol")
        assert a == b

    def test_different_inputs_differ(self) -> None:
        emb = _FakeEmbedder()
        a = emb.embed_query("propranolol")
        b = emb.embed_query("apixaban")
        assert a != b

    def test_embed_documents_shape(self) -> None:
        emb = _FakeEmbedder()
        out = emb.embed_documents(["a", "b", "c"])
        assert len(out) == 3
        assert all(len(v) == EMBEDDING_DIM for v in out)

    def test_floats_in_unit_range(self) -> None:
        emb = _FakeEmbedder()
        v = emb.embed_query("anything")
        assert all(-1.01 <= x <= 1.0 for x in v)


# ---------------------------------------------------------------------------
# Retriever (FakeRetriever)
# ---------------------------------------------------------------------------


def _excerpts() -> list[ChPLExcerpt]:
    return [
        ChPLExcerpt(
            atc_code="C07AA05",
            drug_name="propranolol",
            text="propranolol — beta-blocker, leczenie nadciśnienia, tachykardii.",
        ),
        ChPLExcerpt(
            atc_code="B01AF02",
            drug_name="apixaban",
            text="apixaban — DOAC, profilaktyka udaru w migotaniu przedsionków.",
        ),
        ChPLExcerpt(
            atc_code="A02BC01",
            drug_name="omeprazol",
            text="omeprazol — IPP, leczenie GERD, choroby wrzodowej.",
        ),
    ]


class TestFakeRetriever:
    def test_atc_match_wins(self) -> None:
        ret = _FakeRetriever(excerpts=_excerpts())
        results = ret.retrieve("c07aa05", k=1)
        assert results
        assert results[0].drug_name == "propranolol"

    def test_drug_name_match(self) -> None:
        ret = _FakeRetriever(excerpts=_excerpts())
        results = ret.retrieve("apixaban dose info", k=1)
        assert results
        assert results[0].atc_code == "B01AF02"

    def test_top_k_respected(self) -> None:
        ret = _FakeRetriever(excerpts=_excerpts())
        results = ret.retrieve("propranolol apixaban omeprazol", k=2)
        assert len(results) <= 2

    def test_empty_corpus(self) -> None:
        ret = _FakeRetriever(excerpts=[])
        assert ret.retrieve("anything") == []

    def test_no_match_returns_empty(self) -> None:
        ret = _FakeRetriever(excerpts=_excerpts())
        assert ret.retrieve("krasnoludek") == []


# ---------------------------------------------------------------------------
# ChPLChain — integration of embedder + retriever
# ---------------------------------------------------------------------------


class TestChPLChain:
    def test_retrieve_for_drug(self) -> None:
        chain = build_fake_chain(_excerpts())
        results = chain.retrieve_for_drug(atc_code="B01AF02", drug_name="apixaban", k=1)
        assert results
        assert results[0].atc_code == "B01AF02"

    def test_retrieve_for_drug_no_match(self) -> None:
        chain = build_fake_chain(_excerpts())
        results = chain.retrieve_for_drug(atc_code="X99XX99", drug_name="krasnoludek")
        assert results == []

    def test_chain_exposes_components(self) -> None:
        chain = build_fake_chain(_excerpts())
        assert isinstance(chain, ChPLChain)
        assert chain.embedder is not None
        assert chain.retriever is not None


# ---------------------------------------------------------------------------
# Frontmatter parsing
# ---------------------------------------------------------------------------


class TestFrontmatter:
    def test_basic_frontmatter(self) -> None:
        text = (
            "---\n"
            "atc_code: C07AA05\n"
            "drug_name: propranolol\n"
            "source: ChPL\n"
            "---\n"
            "Body text here.\n"
        )
        meta, body = _split_frontmatter(text)
        assert meta["atc_code"] == "C07AA05"
        assert meta["drug_name"] == "propranolol"
        assert body == "Body text here.\n"

    def test_no_frontmatter(self) -> None:
        meta, body = _split_frontmatter("Just body, no frontmatter.\n")
        assert meta == {}
        assert "Just body" in body


# ---------------------------------------------------------------------------
# Corpus loader
# ---------------------------------------------------------------------------


class TestLoadChPLExcerpts:
    def test_empty_dir(self, tmp_path: Path) -> None:
        assert load_chpl_excerpts(tmp_path) == []

    def test_missing_dir(self, tmp_path: Path) -> None:
        assert load_chpl_excerpts(tmp_path / "nonexistent") == []

    def test_loads_md_files(self, tmp_path: Path) -> None:
        (tmp_path / "C07AA05_propranolol.md").write_text(
            "---\n"
            "atc_code: C07AA05\n"
            "drug_name: Propranolol\n"
            "source: ChPL\n"
            "---\n"
            "Beta-blocker, leczenie nadciśnienia.\n",
            encoding="utf-8",
        )
        (tmp_path / "README.md").write_text("# Index", encoding="utf-8")
        excerpts = load_chpl_excerpts(tmp_path)
        assert len(excerpts) == 1
        assert excerpts[0].atc_code == "C07AA05"
        assert excerpts[0].drug_name == "Propranolol"
        assert "Beta-blocker" in excerpts[0].text

    def test_skips_files_without_atc(self, tmp_path: Path) -> None:
        (tmp_path / "incomplete.md").write_text(
            "---\nfoo: bar\n---\nbody\n", encoding="utf-8"
        )
        assert load_chpl_excerpts(tmp_path) == []
