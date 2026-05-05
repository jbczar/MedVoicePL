"""Shared fixtures for gateway tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from gateway.api.app import build_app
from gateway.config import GatewayConfig
from gateway.security import hash_password
from gateway.storage import build_session_store, build_visit_store

DEMO_USERNAME = "demo_doctor"
DEMO_PASSWORD = "demo_password"


@pytest.fixture
def gateway_config(tmp_path: Path) -> GatewayConfig:
    return GatewayConfig(
        host="127.0.0.1",
        port=8080,
        doctor_username=DEMO_USERNAME,
        doctor_password_hash=hash_password(DEMO_PASSWORD),
        session_secret="test-secret-32-bytes-minimum-okay",
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


@pytest.fixture
def session_store(gateway_config: GatewayConfig):
    return build_session_store(gateway_config.output_dir / "sessions")


@pytest.fixture
def visit_store(gateway_config: GatewayConfig):
    return build_visit_store(gateway_config.output_dir / "visits")


@pytest.fixture
def app(gateway_config: GatewayConfig, session_store, visit_store):
    return build_app(
        config=gateway_config,
        session_store=session_store,
        visit_store=visit_store,
    )


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def authed_client(client: TestClient) -> TestClient:
    response = client.post(
        "/v1/auth/login",
        json={"username": DEMO_USERNAME, "password": DEMO_PASSWORD},
    )
    assert response.status_code == 204
    return client
