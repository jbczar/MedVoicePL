"""Tests for /v1/health (Step 7)."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from gateway.api.app import build_app
from gateway.clients.cloud import _FakeCloudClient
from gateway.clients.on_premise import _FakeOnPremiseClient
from gateway.config import GatewayConfig
from gateway.security import hash_password


def _build_cfg(tmp_path: Path) -> GatewayConfig:
    return GatewayConfig(
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


def test_health_ok_when_both_reachable(tmp_path: Path) -> None:
    cfg = _build_cfg(tmp_path)
    app = build_app(
        config=cfg,
        on_premise_client=_FakeOnPremiseClient(healthy=True),
        cloud_client=_FakeCloudClient(healthy=True),
    )
    resp = TestClient(app).get("/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["on_premise_reachable"] is True
    assert body["cloud_reachable"] is True


def test_health_degraded_when_on_premise_down(tmp_path: Path) -> None:
    cfg = _build_cfg(tmp_path)
    app = build_app(
        config=cfg,
        on_premise_client=_FakeOnPremiseClient(healthy=False),
        cloud_client=_FakeCloudClient(healthy=True),
    )
    resp = TestClient(app).get("/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["on_premise_reachable"] is False


def test_health_degraded_when_clients_unconfigured(tmp_path: Path) -> None:
    app = build_app(config=_build_cfg(tmp_path))
    resp = TestClient(app).get("/v1/health")
    assert resp.json()["status"] == "degraded"


def test_health_no_auth_required(tmp_path: Path) -> None:
    """Health probe must work without a session cookie."""
    cfg = _build_cfg(tmp_path)
    app = build_app(
        config=cfg,
        on_premise_client=_FakeOnPremiseClient(healthy=True),
        cloud_client=_FakeCloudClient(healthy=True),
    )
    client = TestClient(app)
    client.cookies.clear()
    assert client.get("/v1/health").status_code == 200
