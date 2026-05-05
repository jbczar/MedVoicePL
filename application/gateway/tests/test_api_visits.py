"""Tests for /v1/visits CRUD + audio upload (Step 3, no finalize yet)."""

from __future__ import annotations

import hashlib
import io
from datetime import date

import pytest
from fastapi.testclient import TestClient


def test_create_visit_returns_draft(authed_client: TestClient) -> None:
    resp = authed_client.post(
        "/v1/visits",
        json={"specialization": "kardiologia", "visit_date": "2026-05-05"},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "draft"
    assert "visit_id" in body


def test_create_visit_without_session_401(client: TestClient) -> None:
    resp = client.post("/v1/visits", json={})
    assert resp.status_code == 401


def test_create_visit_extra_field_rejected(authed_client: TestClient) -> None:
    resp = authed_client.post("/v1/visits", json={"patient_name": "leak"})
    assert resp.status_code == 422


def test_get_visit_returns_summary(authed_client: TestClient) -> None:
    created = authed_client.post(
        "/v1/visits",
        json={"specialization": "kardiologia"},
    ).json()
    resp = authed_client.get(f"/v1/visits/{created['visit_id']}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["visit_id"] == created["visit_id"]
    assert body["status"] == "draft"
    assert "session_id" not in body  # PII-safe summary


def test_get_visit_not_found(authed_client: TestClient) -> None:
    from uuid import uuid4

    resp = authed_client.get(f"/v1/visits/{uuid4()}")
    assert resp.status_code == 404


def test_get_visit_cross_session_isolation(client: TestClient) -> None:
    # Login as doctor, create visit, logout. Login again to get fresh
    # session, then attempt to read previous visit_id.
    a = client.post(
        "/v1/auth/login",
        json={"username": "demo_doctor", "password": "demo_password"},
    )
    assert a.status_code == 204
    visit = client.post("/v1/visits", json={}).json()
    client.post("/v1/auth/logout")
    client.cookies.clear()

    # Fresh login = fresh session.
    client.post(
        "/v1/auth/login",
        json={"username": "demo_doctor", "password": "demo_password"},
    )
    resp = client.get(f"/v1/visits/{visit['visit_id']}")
    # Same response as not_found — don't reveal existence.
    assert resp.status_code == 404


@pytest.fixture
def small_wav_bytes() -> bytes:
    # 44-byte WAV header + 1024 silence samples = 1068 bytes.
    import struct

    header = b"RIFF"
    data_size = 1024 * 2  # 1024 samples × 2 bytes (16-bit mono)
    chunk_size = 36 + data_size
    header += struct.pack("<I", chunk_size)
    header += b"WAVE"
    header += b"fmt "
    header += struct.pack("<IHHIIHH", 16, 1, 1, 16000, 32000, 2, 16)
    header += b"data"
    header += struct.pack("<I", data_size)
    return header + b"\x00" * data_size


def test_upload_audio_happy_path(authed_client: TestClient, small_wav_bytes: bytes) -> None:
    visit = authed_client.post("/v1/visits", json={}).json()
    resp = authed_client.post(
        f"/v1/visits/{visit['visit_id']}/audio",
        files={"file": ("audio.wav", io.BytesIO(small_wav_bytes), "audio/wav")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "audio_uploaded"
    assert body["audio_sha256"] == hashlib.sha256(small_wav_bytes).hexdigest()


def test_upload_audio_rejects_non_wav_mime(
    authed_client: TestClient, small_wav_bytes: bytes
) -> None:
    visit = authed_client.post("/v1/visits", json={}).json()
    resp = authed_client.post(
        f"/v1/visits/{visit['visit_id']}/audio",
        files={"file": ("audio.mp3", io.BytesIO(small_wav_bytes), "audio/mpeg")},
    )
    assert resp.status_code == 415


def test_upload_audio_idempotent_second_upload(
    authed_client: TestClient, small_wav_bytes: bytes
) -> None:
    visit = authed_client.post("/v1/visits", json={}).json()
    first = authed_client.post(
        f"/v1/visits/{visit['visit_id']}/audio",
        files={"file": ("audio.wav", io.BytesIO(small_wav_bytes), "audio/wav")},
    )
    assert first.status_code == 200
    second = authed_client.post(
        f"/v1/visits/{visit['visit_id']}/audio",
        files={"file": ("audio.wav", io.BytesIO(small_wav_bytes), "audio/wav")},
    )
    assert second.status_code == 200
    assert second.json()["audio_sha256"] == first.json()["audio_sha256"]


def test_upload_audio_too_large_returns_413(authed_client: TestClient) -> None:
    visit = authed_client.post("/v1/visits", json={}).json()
    huge = b"\x00" * (51 * 1024 * 1024)
    resp = authed_client.post(
        f"/v1/visits/{visit['visit_id']}/audio",
        files={"file": ("audio.wav", io.BytesIO(huge), "audio/wav")},
    )
    assert resp.status_code == 413


def test_upload_audio_visit_not_found(
    authed_client: TestClient, small_wav_bytes: bytes
) -> None:
    from uuid import uuid4

    resp = authed_client.post(
        f"/v1/visits/{uuid4()}/audio",
        files={"file": ("audio.wav", io.BytesIO(small_wav_bytes), "audio/wav")},
    )
    assert resp.status_code == 404


def test_visit_state_transitions_draft_to_uploaded(
    authed_client: TestClient, small_wav_bytes: bytes
) -> None:
    visit = authed_client.post("/v1/visits", json={}).json()
    assert authed_client.get(f"/v1/visits/{visit['visit_id']}").json()["status"] == "draft"
    authed_client.post(
        f"/v1/visits/{visit['visit_id']}/audio",
        files={"file": ("audio.wav", io.BytesIO(small_wav_bytes), "audio/wav")},
    )
    assert (
        authed_client.get(f"/v1/visits/{visit['visit_id']}").json()["status"]
        == "audio_uploaded"
    )


def test_create_visit_with_invalid_specialization(authed_client: TestClient) -> None:
    resp = authed_client.post("/v1/visits", json={"specialization": "ghosts"})
    assert resp.status_code == 422


def test_create_visit_visit_date_round_trip(authed_client: TestClient) -> None:
    created = authed_client.post(
        "/v1/visits", json={"visit_date": "2026-05-05"}
    ).json()
    summary = authed_client.get(f"/v1/visits/{created['visit_id']}").json()
    assert summary["visit_date"] == str(date(2026, 5, 5))
