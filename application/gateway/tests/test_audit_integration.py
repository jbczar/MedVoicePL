"""End-to-end audit wiring: login + visit_create + visit_complete."""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from gateway.api.app import build_app
from gateway.audit import GatewayAuditLogger
from gateway.clients.cloud import _FakeCloudClient
from gateway.clients.on_premise import _FakeOnPremiseClient
from gateway.config import GatewayConfig
from gateway.orchestrator import VisitOrchestrator
from gateway.security import hash_password
from gateway.storage import build_session_store, build_visit_store
from shared.schemas import AgentSuggestion, AnonymizedSOAPNote

FIXTURES = Path(__file__).parent / "fixtures"


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


@pytest.fixture
def audited_client(tmp_path: Path):
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

    soap = AnonymizedSOAPNote.model_validate_json(
        (FIXTURES / "anonymized_soap_endo_gra.json").read_text()
    )
    sugg = AgentSuggestion.model_validate_json(
        (FIXTURES / "agent_suggestion_endo_gra.json").read_text()
    )
    on_prem = _FakeOnPremiseClient()
    cloud = _FakeCloudClient(response=sugg)
    audit = GatewayAuditLogger(audit_dir=cfg.audit_dir, retention_days=730)
    orch = VisitOrchestrator(
        on_premise=on_prem,
        cloud=cloud,
        visit_store=visits,
        audit=audit,
    )
    app = build_app(
        config=cfg,
        session_store=sessions,
        visit_store=visits,
        orchestrator=orch,
    )
    app.state.audit = audit
    client = TestClient(app)
    return client, audit, on_prem, soap, cfg


def test_login_success_writes_audit(audited_client) -> None:
    client, _audit, _on_prem, _soap, cfg = audited_client
    resp = client.post(
        "/v1/auth/login",
        json={"username": "demo_doctor", "password": "demo_password"},
    )
    assert resp.status_code == 204
    today = datetime.now(UTC).date().isoformat()
    log_file = cfg.audit_dir / today / "_login" / "logins.jsonl"
    assert log_file.is_file()
    line = log_file.read_text().strip().splitlines()[-1]
    record = json.loads(line)
    assert record["success"] is True
    assert record["username"] == "demo_doctor"


def test_login_failure_writes_audit(audited_client) -> None:
    client, _audit, _on_prem, _soap, cfg = audited_client
    resp = client.post(
        "/v1/auth/login",
        json={"username": "demo_doctor", "password": "wrong"},
    )
    assert resp.status_code == 401
    today = datetime.now(UTC).date().isoformat()
    log_file = cfg.audit_dir / today / "_login" / "logins.jsonl"
    line = json.loads(log_file.read_text().strip().splitlines()[-1])
    assert line["success"] is False


def test_visit_create_writes_audit(audited_client) -> None:
    client, _audit, _on_prem, _soap, cfg = audited_client
    client.post(
        "/v1/auth/login",
        json={"username": "demo_doctor", "password": "demo_password"},
    )
    visit = client.post(
        "/v1/visits", json={"specialization": "endokrynologia"}
    ).json()
    files = list(cfg.audit_dir.rglob("visit_create.json"))
    assert len(files) == 1
    record = json.loads(files[0].read_text())
    assert record["visit_id"] == visit["visit_id"]
    assert record["specialization"] == "endokrynologia"


def test_visit_complete_writes_audit(audited_client) -> None:
    client, _audit, on_prem, soap, cfg = audited_client
    client.post(
        "/v1/auth/login",
        json={"username": "demo_doctor", "password": "demo_password"},
    )
    visit = client.post(
        "/v1/visits",
        json={"specialization": "endokrynologia", "visit_date": "2026-05-06"},
    ).json()
    client.post(
        f"/v1/visits/{visit['visit_id']}/audio",
        files={"file": ("audio.wav", io.BytesIO(_wav_bytes()), "audio/wav")},
    )
    import hashlib

    on_prem.responses = {hashlib.sha256(_wav_bytes()).hexdigest()[:8]: soap}
    resp = client.post(f"/v1/visits/{visit['visit_id']}/finalize")
    assert resp.status_code == 200
    completes = list(cfg.audit_dir.rglob("visit_complete.json"))
    assert len(completes) == 1
    record = json.loads(completes[0].read_text())
    assert record["visit_id"] == visit["visit_id"]
    assert record["drug_count"] >= 1
