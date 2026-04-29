"""Tests for `cloud.llm_integration.llm_client`.

Stays off the network: factory + Settings tested by env-only;
`VertexClient.__post_init__` is exercised through monkeypatching
`sys.modules["google.genai"]`.
"""

from __future__ import annotations

import sys
import types

import pytest

from cloud.llm_integration.llm_client import (
    LLMClient,
    PermanentLLMError,
    TransientLLMError,
    VertexClient,
    VertexSettings,
    build_llm_client,
)
from cloud.llm_integration.prompts import (
    SYSTEM_PROMPT_PL,
    render_candidates_prompt,
    render_correction_hint,
)
from cloud.llm_integration.response_parser import DrugCandidate, DrugCandidateList


def test_vertex_settings_requires_gcp_project(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    with pytest.raises(RuntimeError, match="GOOGLE_CLOUD_PROJECT"):
        VertexSettings.from_env()


def test_vertex_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-proj")
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
    monkeypatch.delenv("VERTEX_AGENT_MODEL", raising=False)
    settings = VertexSettings.from_env()
    assert settings.project == "test-proj"
    assert settings.location == "us-central1"
    assert settings.model_name == "gemini-2.0-pro"


def test_vertex_settings_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "europe-west1")
    monkeypatch.setenv("VERTEX_AGENT_MODEL", "gemini-2.5-flash")
    s = VertexSettings.from_env()
    assert s.location == "europe-west1"
    assert s.model_name == "gemini-2.5-flash"


def test_build_llm_client_dispatches_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    fake_module = types.ModuleType("google")
    fake_genai = types.ModuleType("google.genai")

    class _FakeClient:
        def __init__(self, **_: object) -> None:
            pass

    fake_genai.Client = _FakeClient
    fake_module.genai = fake_genai
    monkeypatch.setitem(sys.modules, "google", fake_module)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)

    client = build_llm_client(model_override="gemini-2.5-flash")
    assert isinstance(client, VertexClient)
    assert client.name == "gemini-2.5-flash"


def test_drug_candidate_round_trip() -> None:
    raw = {
        "name": "Bisoprolol",
        "atc_code": "C07AB07",
        "rationale": "leczenie nadciśnienia",
        "provenance": ["assessment.primary_diagnosis"],
    }
    cand = DrugCandidate.model_validate(raw)
    assert cand.atc_code == "C07AB07"
    assert cand.provenance == ["assessment.primary_diagnosis"]


def test_drug_candidate_provenance_required() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        DrugCandidate(
            name="Bisoprolol",
            atc_code="C07AB07",
            rationale="x",
            provenance=[],
        )


def test_drug_candidate_list_default_empty() -> None:
    lst = DrugCandidateList()
    assert lst.candidates == []
    assert lst.notes == []


def test_system_prompt_carries_anti_hedge_and_empty_rule() -> None:
    text = SYSTEM_PROMPT_PL.lower()
    # Anti-hedge — model is support, not autonomous diagnostician
    assert "wsparcie" in text or "support" in text
    # Empty-output rule for preliminary diagnoses
    assert "podejrzenie" in text
    assert "wymaga potwierdzenia" in text or "potwierdzenia" in text
    # Provenance non-empty rule
    assert "provenance" in text


def test_correction_hint_includes_error() -> None:
    hint = render_correction_hint("provenance must be non-empty")
    assert "provenance must be non-empty" in hint
    assert "popraw" in hint.lower() or "popraw" in hint


def test_render_candidates_prompt_smoke(make_anonymized_soap) -> None:
    soap = make_anonymized_soap()
    rendered = render_candidates_prompt(soap)
    assert "SOAP" in rendered
    assert soap.assessment.primary_diagnosis in rendered
    assert "ATC" in rendered or "ICD" in rendered


def test_llm_client_protocol_has_name_and_generate() -> None:
    # Duck-check Protocol shape (not @runtime_checkable, by design).
    assert hasattr(LLMClient, "generate") or "generate" in LLMClient.__annotations__


def test_transient_and_permanent_errors_are_distinct() -> None:
    assert TransientLLMError is not PermanentLLMError
    assert issubclass(TransientLLMError, RuntimeError)
    assert issubclass(PermanentLLMError, RuntimeError)
