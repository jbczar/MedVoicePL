"""Tests for `cloud.api.app` — FastAPI TestClient with stubbed agent.

No Vertex / Qdrant / sentence-transformers loading. The full
`CloudAgent` is replaced via `app.dependency_overrides[get_agent]`.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from cloud.api.app import _AGENT_HOLDER, app, get_agent
from cloud.llm_integration.response_parser import DrugCandidate, DrugCandidateList
from cloud.nfz.formulary_search import NFZEntry, _FakeNFZIndex
from cloud.orchestrator import CloudAgent
from cloud.rag.chain import build_fake_chain
from cloud.rag.retriever import ChPLExcerpt
from shared.schemas import ReimbursementLevel


class _FakeLLM:
    name = "fake-llm"

    def __init__(self, canned: DrugCandidateList) -> None:
        self.canned = canned

    def generate(self, prompt: str, *, response_schema: type, system_instruction: str | None = None):
        return self.canned


def _build_test_agent() -> CloudAgent:
    canned = DrugCandidateList(
        candidates=[
            DrugCandidate(
                name="Propranolol",
                atc_code="C07AA05",
                rationale="leczenie tachykardii",
                provenance=["assessment.primary_diagnosis"],
            )
        ]
    )
    return CloudAgent(
        llm=_FakeLLM(canned=canned),
        nfz=_FakeNFZIndex(
            entries=(
                NFZEntry(
                    atc_code="C07AA05",
                    substance="Propranolol",
                    reimbursement=ReimbursementLevel.standard,
                    patient_copay_pln=Decimal("3.20"),
                ),
            ),
            snapshot_date=date(2026, 4, 1),
        ),
        chain=build_fake_chain(
            [
                ChPLExcerpt(
                    atc_code="C07AA05",
                    drug_name="propranolol",
                    text="propranolol — beta-blocker.",
                )
            ]
        ),
        agent_version="3a.0-test",
    )


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CLOUD_AGENT_CACHE_ROOT", str(tmp_path))
    monkeypatch.setenv("AGENT_VERSION", "3a.0-test")
    _AGENT_HOLDER.clear()
    agent = _build_test_agent()
    app.dependency_overrides[get_agent] = lambda: agent
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    _AGENT_HOLDER.clear()


def _sample_payload(make_anonymized_soap: Any) -> dict[str, Any]:
    soap = make_anonymized_soap()
    return soap.model_dump(mode="json")


class TestHealth:
    def test_health_shape(self, client: TestClient) -> None:
        resp = client.get("/v1/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["agent_version"] == "3a.0-test"
        assert body["nfz_index_size"] == 1
        assert body["chpl_corpus_size"] == 1
        assert body["llm_loaded"] is True


class TestSuggest:
    def test_happy_path(self, client: TestClient, make_anonymized_soap: Any) -> None:
        resp = client.post("/v1/agent/suggest", json=_sample_payload(make_anonymized_soap))
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["drugs"]) == 1
        drug = body["drugs"][0]
        assert drug["atc_code"] == "C07AA05"
        assert drug["reimbursement"] == "70%"
        assert resp.headers.get("x-cached") == "false"

    def test_idempotent_second_call_is_cached(
        self, client: TestClient, make_anonymized_soap: Any
    ) -> None:
        payload = _sample_payload(make_anonymized_soap)
        first = client.post("/v1/agent/suggest", json=payload)
        assert first.status_code == 200
        second = client.post("/v1/agent/suggest", json=payload)
        assert second.status_code == 200
        assert second.headers.get("x-cached") == "true"
        assert second.json()["drugs"] == first.json()["drugs"]

    def test_force_bypasses_cache(
        self, client: TestClient, make_anonymized_soap: Any
    ) -> None:
        payload = _sample_payload(make_anonymized_soap)
        client.post("/v1/agent/suggest", json=payload)
        resp = client.post("/v1/agent/suggest?force=true", json=payload)
        assert resp.headers.get("x-cached") == "false"

    def test_invalid_body_returns_422(self, client: TestClient) -> None:
        resp = client.post("/v1/agent/suggest", json={"foo": "bar"})
        assert resp.status_code == 422

    def test_extra_fields_rejected(
        self, client: TestClient, make_anonymized_soap: Any
    ) -> None:
        payload = _sample_payload(make_anonymized_soap)
        payload["surplus_field"] = "should fail extra=forbid"
        resp = client.post("/v1/agent/suggest", json=payload)
        assert resp.status_code == 422


class TestErrorHandling:
    def test_agent_failure_returns_500(
        self, client: TestClient, make_anonymized_soap: Any
    ) -> None:
        class _BoomAgent:
            llm = None
            nfz = type("N", (), {"__len__": lambda self: 0})()
            chain = type("C", (), {"retriever": type("R", (), {"__len__": lambda self: 0})()})()

            def run(self, _soap):
                raise RuntimeError("boom")

        app.dependency_overrides[get_agent] = lambda: _BoomAgent()
        try:
            resp = client.post("/v1/agent/suggest", json=_sample_payload(make_anonymized_soap))
            assert resp.status_code == 500
            assert "boom" in resp.text
        finally:
            app.dependency_overrides[get_agent] = lambda: _build_test_agent()
