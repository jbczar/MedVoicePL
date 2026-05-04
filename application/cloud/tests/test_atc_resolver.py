"""Tests for `cloud.nfz.atc_resolver` — fakes only, no real LLM."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from cloud.nfz.atc_resolver import (
    AtcLookupResult,
    _FakeAtcResolver,
    _GeminiResolver,
    _NullResolver,
    build_resolver,
)


class _StubLLM:
    """Returns a canned `AtcLookupResult` for every call."""

    name = "stub-llm"

    def __init__(self, atc: str = "C07AA05", display: str = "Propranolol") -> None:
        self.atc = atc
        self.display = display
        self.calls: list[str] = []

    def generate(self, prompt: str, *, response_schema: type[BaseModel], system_instruction=None):
        self.calls.append(prompt)
        assert response_schema is AtcLookupResult
        return AtcLookupResult(atc_code=self.atc, display_pl=self.display)


class TestNullResolver:
    def test_always_returns_none(self) -> None:
        r = _NullResolver()
        assert r.resolve("Propranolol") is None
        assert r.resolve("anything else") is None


class TestFakeResolver:
    def test_direct_hit(self) -> None:
        r = _FakeAtcResolver(canned={"Propranolol": "C07AA05"})
        assert r.resolve("Propranolol") == "C07AA05"
        assert r.calls == 1

    def test_case_insensitive_fallback(self) -> None:
        r = _FakeAtcResolver(canned={"Apixaban": "B01AF02"})
        assert r.resolve("APIXABAN") == "B01AF02"

    def test_miss_returns_none(self) -> None:
        r = _FakeAtcResolver(canned={"Propranolol": "C07AA05"})
        assert r.resolve("Krasnoludek") is None

    def test_negative_canned_returns_none(self) -> None:
        r = _FakeAtcResolver(canned={"Krasnoludek": None})
        assert r.resolve("Krasnoludek") is None


class TestGeminiResolverCache:
    def test_first_call_hits_llm(self, tmp_path: Path) -> None:
        llm = _StubLLM(atc="R03BB05", display="Aklidynium")
        r = _GeminiResolver(llm=llm, cache_path=tmp_path / "cache.json")
        assert r.resolve("Aclidinium") == "R03BB05"
        assert len(llm.calls) == 1

    def test_second_call_uses_cache(self, tmp_path: Path) -> None:
        llm = _StubLLM(atc="R03BB05")
        r = _GeminiResolver(llm=llm, cache_path=tmp_path / "cache.json")
        r.resolve("Aclidinium")
        r.resolve("Aclidinium")
        r.resolve("Aclidinium")
        assert len(llm.calls) == 1

    def test_negative_result_cached(self, tmp_path: Path) -> None:
        class _UnknownLLM:
            name = "unknown-llm"

            def __init__(self) -> None:
                self.calls = 0

            def generate(self, prompt, *, response_schema, system_instruction=None):
                self.calls += 1
                return AtcLookupResult(atc_code="UNKNOWN", display_pl="")

        llm = _UnknownLLM()
        r = _GeminiResolver(llm=llm, cache_path=tmp_path / "cache.json")
        assert r.resolve("Krasnoludek") is None
        assert r.resolve("Krasnoludek") is None
        assert llm.calls == 1  # second call hits cache

    def test_cache_persisted_to_disk(self, tmp_path: Path) -> None:
        cache_file = tmp_path / "cache.json"
        llm = _StubLLM(atc="R03BB05", display="Aklidynium")
        r = _GeminiResolver(llm=llm, cache_path=cache_file)
        r.resolve("Aclidinium")
        assert cache_file.is_file()
        data = json.loads(cache_file.read_text("utf-8"))
        assert "Aclidinium" in data
        assert data["Aclidinium"]["atc"] == "R03BB05"

    def test_cache_loaded_across_instances(self, tmp_path: Path) -> None:
        cache_file = tmp_path / "cache.json"
        llm_a = _StubLLM(atc="R03BB05")
        r_a = _GeminiResolver(llm=llm_a, cache_path=cache_file)
        r_a.resolve("Aclidinium")
        # New instance, fresh LLM that would crash if called.
        llm_b = _StubLLM(atc="WRONG999")
        r_b = _GeminiResolver(llm=llm_b, cache_path=cache_file)
        assert r_b.resolve("Aclidinium") == "R03BB05"
        assert llm_b.calls == []  # cache hit, no LLM call

    def test_case_insensitive_disk_cache(self, tmp_path: Path) -> None:
        cache_file = tmp_path / "cache.json"
        llm = _StubLLM(atc="R03BB05")
        r = _GeminiResolver(llm=llm, cache_path=cache_file)
        r.resolve("Aclidinium")
        assert r.resolve("aclidinium") == "R03BB05"
        assert len(llm.calls) == 1


class TestBuildResolver:
    def test_null_when_llm_missing(self, tmp_path: Path) -> None:
        r = build_resolver(llm=None, cache_path=tmp_path / "c.json")
        assert isinstance(r, _NullResolver)

    def test_gemini_when_llm_present(self, tmp_path: Path) -> None:
        r = build_resolver(llm=_StubLLM(), cache_path=tmp_path / "c.json")
        assert isinstance(r, _GeminiResolver)


class TestAtcLookupResultSchema:
    def test_valid_atc(self) -> None:
        result = AtcLookupResult(atc_code="C07AA05", display_pl="Propranolol")
        assert result.atc_code == "C07AA05"

    def test_unknown_accepted(self) -> None:
        result = AtcLookupResult(atc_code="UNKNOWN", display_pl="")
        assert result.atc_code == "UNKNOWN"

    def test_malformed_rejected(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            AtcLookupResult(atc_code="not-an-atc", display_pl="")
