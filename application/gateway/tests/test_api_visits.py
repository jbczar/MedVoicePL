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


# --- SOAP + transcript endpoints (Phase 3c.1 frontend enrichment) ---


def _make_completed_visit(
    authed_client: TestClient, visit_store, *, with_soap: bool = True
) -> dict:
    """Helper: create a visit, force-set status=completed, drop a soap.json."""
    import json
    from uuid import UUID

    from shared.schemas import VisitStatus

    created = authed_client.post(
        "/v1/visits", json={"specialization": "kardiologia"}
    ).json()
    vid = UUID(created["visit_id"])
    visit = visit_store.get(vid)
    visit.status = VisitStatus.completed
    visit.transcript_id = "deadbeef"
    visit_store.save(visit)

    if with_soap:
        soap_path = visit_store.visit_dir(vid) / "soap.json"
        soap_path.parent.mkdir(parents=True, exist_ok=True)
        soap_path.write_text(
            json.dumps(
                {
                    "subjective": {
                        "chief_complaint": "ból w klatce piersiowej",
                        "history_of_present_illness": "od dwóch dni",
                        "review_of_systems": [],
                        "medical_history": [],
                        "medications_current": [],
                        "allergies": [],
                        "provenance": [1, 2],
                    },
                    "objective": {
                        "physical_exam": ["BP 140/90"],
                        "vitals": [],
                        "labs_mentioned": [],
                        "provenance": [],
                    },
                    "assessment": {
                        "primary_diagnosis": "nadciśnienie tętnicze",
                        "icd10_code": "I10",
                        "differential": [],
                        "clinical_reasoning": "podwyższone wartości RR",
                        "provenance": [],
                    },
                    "plan": {
                        "medications_prescribed": [],
                        "diagnostic_tests": [],
                        "referrals": [],
                        "follow_up": None,
                        "patient_education": [],
                        "provenance": [],
                    },
                    "red_flags": [],
                    "red_flags_provenance": [],
                    "metadata": {"anonymizer_version": "2c.0"},
                }
            ),
            encoding="utf-8",
        )
    return created


def test_get_soap_returns_cached(authed_client: TestClient, visit_store) -> None:
    created = _make_completed_visit(authed_client, visit_store)
    resp = authed_client.get(f"/v1/visits/{created['visit_id']}/soap")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["assessment"]["primary_diagnosis"] == "nadciśnienie tętnicze"
    assert body["assessment"]["icd10_code"] == "I10"


def test_get_soap_409_when_not_completed(authed_client: TestClient) -> None:
    created = authed_client.post("/v1/visits", json={}).json()
    resp = authed_client.get(f"/v1/visits/{created['visit_id']}/soap")
    assert resp.status_code == 409
    assert "soap_unavailable_in_status" in resp.json()["detail"]


def test_get_soap_404_when_file_missing(
    authed_client: TestClient, visit_store
) -> None:
    created = _make_completed_visit(authed_client, visit_store, with_soap=False)
    resp = authed_client.get(f"/v1/visits/{created['visit_id']}/soap")
    assert resp.status_code == 404


def test_get_transcript_503_without_on_premise_client(
    authed_client: TestClient, visit_store
) -> None:
    # No on_premise_client wired in the conftest app — endpoint should 503.
    created = _make_completed_visit(authed_client, visit_store)
    resp = authed_client.get(f"/v1/visits/{created['visit_id']}/transcript")
    assert resp.status_code == 503
    assert resp.json()["detail"] == "on_premise_client_not_configured"


def test_get_transcript_with_fake_client(
    gateway_config, session_store, visit_store
) -> None:
    """Build app WITH a _FakeOnPremiseClient and verify proxy works."""
    from gateway.api.app import build_app
    from gateway.clients.on_premise import _FakeOnPremiseClient

    fake = _FakeOnPremiseClient(
        canned_transcripts={
            "deadbeef": {
                "scenario_id": "deadbeef",
                "audio_path": "/tmp/x.wav",
                "audio_sha256": "0" * 64,
                "duration_s": 12.3,
                "segments": [
                    {
                        "speaker": "doctor",
                        "start_s": 0.0,
                        "end_s": 5.0,
                        "text": "Co Pana boli?",
                    },
                    {
                        "speaker": "patient",
                        "start_s": 5.0,
                        "end_s": 10.0,
                        "text": "Boli mnie w klatce",
                    },
                ],
                "metadata": {"engine": "mlx", "model": "whisper-large-v3"},
            }
        },
    )
    app = build_app(
        config=gateway_config,
        session_store=session_store,
        visit_store=visit_store,
        on_premise_client=fake,
    )
    client = TestClient(app)
    client.post(
        "/v1/auth/login",
        json={"username": "demo_doctor", "password": "demo_password"},
    )
    created = _make_completed_visit(client, visit_store)
    resp = client.get(f"/v1/visits/{created['visit_id']}/transcript")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert len(body["segments"]) == 2
    assert body["segments"][0]["speaker"] == "doctor"
    assert fake.get_transcript_calls == ["deadbeef"]
