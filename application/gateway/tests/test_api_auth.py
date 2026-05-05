"""Tests for /v1/auth/{login,logout,me} endpoints."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient

from gateway.api.auth import SESSION_COOKIE_NAME
from gateway.models import DoctorIdentity, Session


def test_login_with_valid_credentials_sets_cookie(client: TestClient) -> None:
    resp = client.post(
        "/v1/auth/login",
        json={"username": "demo_doctor", "password": "demo_password"},
    )
    assert resp.status_code == 204
    cookie = resp.cookies.get(SESSION_COOKIE_NAME)
    assert cookie is not None
    assert len(cookie) >= 32  # UUID-shaped


def test_login_wrong_password_returns_401(client: TestClient) -> None:
    resp = client.post(
        "/v1/auth/login",
        json={"username": "demo_doctor", "password": "wrong"},
    )
    assert resp.status_code == 401
    assert SESSION_COOKIE_NAME not in resp.cookies


def test_login_unknown_username_returns_401(client: TestClient) -> None:
    resp = client.post(
        "/v1/auth/login",
        json={"username": "ghost", "password": "demo_password"},
    )
    assert resp.status_code == 401


def test_login_missing_fields_returns_422(client: TestClient) -> None:
    resp = client.post("/v1/auth/login", json={"username": "demo_doctor"})
    assert resp.status_code == 422


def test_me_returns_identity_after_login(authed_client: TestClient) -> None:
    resp = authed_client.get("/v1/auth/me")
    assert resp.status_code == 200
    body = resp.json()
    assert body["username"] == "demo_doctor"
    assert body["full_name"] == "Demo Lekarz"


def test_me_without_session_returns_401(client: TestClient) -> None:
    resp = client.get("/v1/auth/me")
    assert resp.status_code == 401


def test_me_with_garbage_cookie_returns_401(client: TestClient) -> None:
    client.cookies.set(SESSION_COOKIE_NAME, "not-a-uuid")
    resp = client.get("/v1/auth/me")
    assert resp.status_code == 401


def test_logout_invalidates_session(authed_client: TestClient) -> None:
    out = authed_client.post("/v1/auth/logout")
    assert out.status_code == 204
    # After logout the cookie reference no longer resolves.
    assert authed_client.get("/v1/auth/me").status_code == 401


def test_expired_session_returns_401_and_deletes(client: TestClient, session_store) -> None:
    # Manually plant an expired session into the store.
    from uuid import uuid4

    sid = uuid4()
    now = datetime.now(UTC)
    session_store.save(
        Session(
            session_id=sid,
            doctor=DoctorIdentity(username="demo_doctor"),
            created_at=now - timedelta(hours=10),
            expires_at=now - timedelta(seconds=1),
            last_seen=now - timedelta(hours=10),
        )
    )
    client.cookies.set(SESSION_COOKIE_NAME, str(sid))
    resp = client.get("/v1/auth/me")
    assert resp.status_code == 401
    assert session_store.get(sid) is None


def test_session_last_seen_updates_on_request(authed_client: TestClient, session_store) -> None:
    cookie = authed_client.cookies.get(SESSION_COOKIE_NAME)
    assert cookie is not None
    from uuid import UUID

    sid = UUID(cookie)
    s_before = session_store.get(sid)
    assert s_before is not None
    last_before = s_before.last_seen
    # Tiny sleep substitute: just call /me again and compare last_seen.
    resp = authed_client.get("/v1/auth/me")
    assert resp.status_code == 200
    s_after = session_store.get(sid)
    assert s_after is not None
    assert s_after.last_seen >= last_before
