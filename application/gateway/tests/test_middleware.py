"""Tests for CORS + rate limit middleware (Step 7)."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from gateway.api.app import build_app
from gateway.clients.cloud import _FakeCloudClient
from gateway.clients.on_premise import _FakeOnPremiseClient
from gateway.config import GatewayConfig
from gateway.security import hash_password


def _build_cfg(tmp_path: Path, **overrides) -> GatewayConfig:
    base = {
        "host": "127.0.0.1",
        "port": 8080,
        "doctor_username": "demo_doctor",
        "doctor_password_hash": hash_password("demo_password"),
        "session_secret": "x" * 32,
        "session_ttl_hours": 8,
        "on_premise_api_url": "http://127.0.0.1:8001",
        "cloud_api_url": "http://127.0.0.1:8002",
        "audit_retention_days": 730,
        "cors_origins": ("http://localhost:3000",),
        "rate_limit_enabled": False,
        "rate_limit_per_min": 60,
        "log_level": "info",
        "output_dir": tmp_path / "output",
        "audit_dir": tmp_path / "audit",
        "on_premise_audit_dir": tmp_path / "on_premise_audit",
    }
    base.update(overrides)
    return GatewayConfig(**base)


def test_cors_allows_whitelisted_origin(tmp_path: Path) -> None:
    cfg = _build_cfg(tmp_path)
    app = build_app(
        config=cfg,
        on_premise_client=_FakeOnPremiseClient(),
        cloud_client=_FakeCloudClient(),
    )
    client = TestClient(app)
    resp = client.options(
        "/v1/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_cors_rejects_disallowed_origin(tmp_path: Path) -> None:
    cfg = _build_cfg(tmp_path)
    app = build_app(
        config=cfg,
        on_premise_client=_FakeOnPremiseClient(),
        cloud_client=_FakeCloudClient(),
    )
    client = TestClient(app)
    resp = client.options(
        "/v1/health",
        headers={
            "Origin": "http://malicious.example",
            "Access-Control-Request-Method": "GET",
        },
    )
    # Either no CORS header or 400; we just assert no allow-origin echo.
    assert resp.headers.get("access-control-allow-origin") != "http://malicious.example"


def test_rate_limit_triggers_429_after_threshold(tmp_path: Path) -> None:
    cfg = _build_cfg(tmp_path, rate_limit_enabled=True, rate_limit_per_min=3)
    app = build_app(
        config=cfg,
        on_premise_client=_FakeOnPremiseClient(),
        cloud_client=_FakeCloudClient(),
    )
    client = TestClient(app)
    statuses = [client.get("/v1/health").status_code for _ in range(6)]
    # First 3 succeed; remaining hit 429.
    assert statuses[:3] == [200, 200, 200]
    assert 429 in statuses[3:]


def test_rate_limit_disabled_when_flag_off(tmp_path: Path) -> None:
    cfg = _build_cfg(tmp_path, rate_limit_enabled=False, rate_limit_per_min=3)
    app = build_app(
        config=cfg,
        on_premise_client=_FakeOnPremiseClient(),
        cloud_client=_FakeCloudClient(),
    )
    client = TestClient(app)
    statuses = [client.get("/v1/health").status_code for _ in range(10)]
    assert all(s == 200 for s in statuses)
