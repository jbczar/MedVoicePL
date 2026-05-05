"""Tests for POST /v1/visits/{vid}/finalize (Step 5)."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from gateway.api.app import build_app
from gateway.clients.cloud import _FakeCloudClient
from gateway.clients.on_premise import _FakeOnPremiseClient
from gateway.config import GatewayConfig
from gateway.orchestrator import VisitOrchestrator
from gateway.security import hash_password
from gateway.storage import build_session_store, build_visit_store
from shared.schemas import AgentSuggestion, AnonymizedSOAPNote

FIXTURES = Path(__file__).parent / "fixtures"


def _load_soap() -> AnonymizedSOAPNote:
    return AnonymizedSOAPNote.model_validate_json(
        (FIXTURES / "anonymized_soap_endo_gra.json").read_text()
    )


def _load_suggestion() -> AgentSuggestion:
    return AgentSuggestion.model_validate_json(
        (FIXTURES / "agent_suggestion_endo_gra.json").read_text()
    )


@pytest.fixture
def authed_client_with_orchestrator(tmp_path: Path):
    """Build a fully-wired app with fakes — useful for finalize tests."""
    cfg = GatewayConfig(
        host="127.0.0.1",
        port=8080,
        doctor_username="demo_doctor",
        doctor_password_hash=hash_password("demo_password"),
        session_secret="x" * 32,
        session_ttl_hours=8,
        on_premise_api_url="http://127.0.0.1:8001",
        cloud_api_url="http://127.0.0.1:8002",
        audit_retention_days=730,
        cors_origins=("http://localhost:3000",),
        rate_limit_enabled=False,
        rate_limit_per_min=60,
        log_level="info",
        output_dir=tmp_path / "output",
        audit_dir=tmp_path / "audit",
        on_premise_audit_dir=tmp_path / "on_premise_audit",
    )
    sessions = build_session_store(cfg.output_dir / "sessions")
    visits = build_visit_store(cfg.output_dir / "visits")
    on_prem = _FakeOnPremiseClient()
    cloud = _FakeCloudClient(response=_load_suggestion())
    orchestrator = VisitOrchestrator(
        on_premise=on_prem,
        cloud=cloud,
        visit_store=visits,
    )
    app = build_app(
        config=cfg,
        session_store=sessions,
        visit_store=visits,
        orchestrator=orchestrator,
    )
    client = TestClient(app)
    client.post(
        "/v1/auth/login",
        json={"username": "demo_doctor", "password": "demo_password"},
    )
    return client, on_prem, cloud, visits


def _wav_bytes() -> bytes:
    import struct

    header = b"RIFF"
    data_size = 1024 * 2
    chunk_size = 36 + data_size
    header += struct.pack("<I", chunk_size)
    header += b"WAVE"
    header += b"fmt "
    header += struct.pack("<IHHIIHH", 16, 1, 1, 16000, 32000, 2, 16)
    header += b"data"
    header += struct.pack("<I", data_size)
    return header + b"\x00" * data_size


def _setup_uploaded_visit(client: TestClient) -> str:
    visit = client.post(
        "/v1/visits",
        json={"specialization": "endokrynologia", "visit_date": "2026-05-05"},
    ).json()
    client.post(
        f"/v1/visits/{visit['visit_id']}/audio",
        files={"file": ("audio.wav", io.BytesIO(_wav_bytes()), "audio/wav")},
    )
    return visit["visit_id"]


def test_finalize_happy_path(authed_client_with_orchestrator) -> None:
    client, on_prem, _cloud, _visits = authed_client_with_orchestrator
    vid = _setup_uploaded_visit(client)
    transcript_id = vid[:8]
    on_prem.responses = {transcript_id: _load_soap()}
    resp = client.post(f"/v1/visits/{vid}/finalize")
    assert resp.status_code == 200
    body = resp.json()
    assert body["visit"]["status"] == "completed"
    assert body["suggestion"]["drugs"]


def test_finalize_returns_502_on_on_prem_error(authed_client_with_orchestrator) -> None:
    client, on_prem, _cloud, _visits = authed_client_with_orchestrator
    vid = _setup_uploaded_visit(client)
    on_prem.raise_on = {vid[:8]}
    resp = client.post(f"/v1/visits/{vid}/finalize")
    assert resp.status_code == 502


def test_finalize_rejects_draft_visit(authed_client_with_orchestrator) -> None:
    client, _on_prem, _cloud, _visits = authed_client_with_orchestrator
    visit = client.post("/v1/visits", json={"specialization": "endokrynologia"}).json()
    resp = client.post(f"/v1/visits/{visit['visit_id']}/finalize")
    assert resp.status_code == 409


def test_finalize_unknown_visit_404(authed_client_with_orchestrator) -> None:
    from uuid import uuid4

    client, _on_prem, _cloud, _visits = authed_client_with_orchestrator
    resp = client.post(f"/v1/visits/{uuid4()}/finalize")
    assert resp.status_code == 404


def test_finalize_orchestrator_unconfigured_returns_503(client: TestClient) -> None:
    # Use the default test client (no orchestrator wired).
    login = client.post(
        "/v1/auth/login",
        json={"username": "demo_doctor", "password": "demo_password"},
    )
    assert login.status_code == 204
    visit = client.post(
        "/v1/visits", json={"specialization": "endokrynologia"}
    ).json()
    resp = client.post(f"/v1/visits/{visit['visit_id']}/finalize")
    assert resp.status_code == 503


def test_finalize_idempotent_replay(authed_client_with_orchestrator) -> None:
    client, on_prem, cloud, _visits = authed_client_with_orchestrator
    vid = _setup_uploaded_visit(client)
    on_prem.responses = {vid[:8]: _load_soap()}

    a = client.post(f"/v1/visits/{vid}/finalize")
    b = client.post(f"/v1/visits/{vid}/finalize")
    assert a.status_code == 200
    assert b.status_code == 200
    # Backend called once.
    assert len(on_prem.calls) == 1
    assert len(cloud.calls) == 1
