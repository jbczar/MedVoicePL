"""Tests for the on-premise FastAPI app.

Uses `TestClient` plus FastAPI dependency overrides to inject:

* a stub `StructuredExtractor` that returns canned SOAP JSON
  (no mlx-lm load, no Bielik download),
* a stub `RecognizerRegistry` with canned `RecognizerResult` lists
  (no spaCy load).

Env vars (`ON_PREMISE_TRANSCRIPTS_ROOT`, `ON_PREMISE_ANONYMIZED_ROOT`,
`AUDIT_ROOT`) are pointed at `tmp_path` via `monkeypatch.setenv` so
each test runs in isolation.

Coverage:
  - ``GET /v1/health`` returns expected shape.
  - ``POST`` with valid transcript returns ``200`` + valid AnonymizedSOAPNote.
  - second ``POST`` returns ``X-Cached: true``.
  - ``POST?force=true`` bypasses cache.
  - ``404`` on missing transcript_id.
  - ``500`` when trust gate trips on a deliberately bad anonymization.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from presidio_analyzer import RecognizerResult

from on_premise.anonymization.presidio_agent import PresidioAnonymizer
from on_premise.api.app import (
    create_app,
    get_anonymizer,
    get_extractor,
)

# ---------------------------------------------------------------------------
# Stubs — same idiom as Phase 2b's _StubExtractor.
# ---------------------------------------------------------------------------


class _StubExtractor:
    """Returns canned SOAPNoteSubmission JSON. No model download."""

    name = "mlx-lm:stub"
    device = "mps"
    structured_mode = "grammar"

    def __init__(self, response: str) -> None:
        self._response = response

    def extract(self, prompt: str, *, schema: Mapping[str, Any]) -> str:
        return self._response


class _FakeRegistry:
    """Returns canned RecognizerResult lists for matching texts."""

    def __init__(self, canned: dict[str, list[RecognizerResult]]) -> None:
        self.canned = canned

    def analyze(self, text: str, language: str = "pl") -> list[RecognizerResult]:
        for trigger, results in self.canned.items():
            if trigger in text:
                offset = text.index(trigger)
                return [
                    RecognizerResult(
                        entity_type=r.entity_type,
                        start=r.start + offset,
                        end=r.end + offset,
                        score=r.score,
                    )
                    for r in results
                ]
        return []


def _canned_soap_submission() -> str:
    """A minimally valid SOAPNoteSubmission JSON the stub returns."""
    return json.dumps(
        {
            "subjective": {
                "chief_complaint": "Pacjent Krzysztof Nowakowski zgłasza ból głowy.",
                "history_of_present_illness": "Bóle od dwóch dni.",
                "review_of_systems": [],
                "medical_history": [],
                "medications_current": [],
                "allergies": [],
                "provenance": [1, 2],
            },
            "objective": {
                "physical_exam": [],
                "vitals": [],
                "labs_mentioned": [],
                "provenance": [],
            },
            "assessment": {
                "primary_diagnosis": "napięciowy ból głowy",
                "icd10_code": "R51",
                "differential": [],
                "clinical_reasoning": "",
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
        }
    )


def _write_fake_transcript(transcripts_root: Path, transcript_id: str) -> Path:
    """Write a minimal valid Transcript JSON for the test."""
    bundle_dir = transcripts_root / transcript_id
    bundle_dir.mkdir(parents=True, exist_ok=True)
    transcript = {
        "scenario_id": transcript_id,
        "audio_path": "/tmp/x.wav",
        "audio_sha256": "b" * 64,
        "duration_s": 5.0,
        "language": "pl",
        "segments": [
            {
                "speaker": "doctor",
                "text": "Dzień dobry.",
                "start_s": 0.0,
                "end_s": 1.0,
                "words": [],
            },
            {
                "speaker": "patient",
                "text": "Boli mnie głowa.",
                "start_s": 1.1,
                "end_s": 4.5,
                "words": [],
            },
        ],
        "metadata": {
            "stt_backend": "mlx",
            "stt_model": "mlx-whisper:large-v3",
            "stt_device": "mps",
            "stt_compute_type": None,
            "diarization_model": "pyannote/speaker-diarization-3.1",
            "diarization_device": "cpu",
            "transcribed_at": datetime(2026, 4, 24, 12, 0, 0).astimezone().isoformat(),
        },
    }
    path = bundle_dir / "transcript.json"
    path.write_text(json.dumps(transcript), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Common test fixture: env vars + dependency overrides.
# ---------------------------------------------------------------------------


@pytest.fixture
def app_with_stubs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Return (TestClient, transcripts_root) with all deps stubbed.

    The trick: env vars must be set BEFORE create_app() runs because
    the dep functions read them at call time. monkeypatch handles the
    teardown automatically.
    """
    transcripts_root = tmp_path / "transcripts"
    transcripts_root.mkdir()
    anonymized_root = tmp_path / "anonymized"
    audit_root = tmp_path / "audit"
    monkeypatch.setenv("ON_PREMISE_TRANSCRIPTS_ROOT", str(transcripts_root))
    monkeypatch.setenv("ON_PREMISE_ANONYMIZED_ROOT", str(anonymized_root))
    monkeypatch.setenv("AUDIT_ROOT", str(audit_root))
    monkeypatch.setenv("ANONYMIZER_VERSION", "2c.0-test")
    monkeypatch.setenv("DATASET_SCENARIOS_ROOT", str(tmp_path / "no-scenarios"))

    app = create_app()

    # Stub the extractor (no mlx-lm).
    app.dependency_overrides[get_extractor] = lambda: _StubExtractor(_canned_soap_submission())

    # Stub the anonymizer (no spaCy).
    registry = _FakeRegistry(
        canned={
            "Pacjent Krzysztof Nowakowski": [
                RecognizerResult(entity_type="PERSON", start=8, end=28, score=1.0),
            ],
        }
    )
    app.dependency_overrides[get_anonymizer] = lambda: PresidioAnonymizer(registry=registry)

    client = TestClient(app)
    return client, transcripts_root, audit_root, anonymized_root


# ---------------------------------------------------------------------------
# Health endpoint.
# ---------------------------------------------------------------------------


def test_health_returns_expected_shape(app_with_stubs) -> None:
    client, _, _, _ = app_with_stubs
    response = client.get("/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    # `analyzer_loaded` may be False because TestClient skips startup
    # in test mode + we don't load real spaCy. Either is fine.
    assert isinstance(body["analyzer_loaded"], bool)
    assert body["anonymizer_version"] == "2c.0-test"


# ---------------------------------------------------------------------------
# Anonymize endpoint — happy path.
# ---------------------------------------------------------------------------


def test_anonymize_visit_returns_valid_payload(app_with_stubs) -> None:
    client, transcripts_root, _, _ = app_with_stubs
    _write_fake_transcript(transcripts_root, "scen-01")
    response = client.post(
        "/v1/visits/scen-01/anonymized-soap",
        params={
            "session_id": str(uuid4()),
            "specialization": "medycyna_rodzinna",
            "visit_date": "2026-04-26",
        },
    )
    assert response.status_code == 200, response.text
    assert response.headers.get("X-Cached") == "false"
    body = response.json()
    # Anonymizer redacted "Krzysztof Nowakowski" → <PERSON_1>.
    assert "Krzysztof Nowakowski" not in json.dumps(body)
    assert "<PERSON_1>" in body["subjective"]["chief_complaint"]
    # Metadata block is the gateway-supplied bits + counts.
    assert body["metadata"]["specialization"] == "medycyna_rodzinna"
    assert body["metadata"]["visit_date"] == "2026-04-26"
    assert body["metadata"]["anonymizer_version"] == "2c.0-test"


def test_anonymize_writes_audit_and_token_map(app_with_stubs) -> None:
    client, transcripts_root, audit_root, anonymized_root = app_with_stubs
    _write_fake_transcript(transcripts_root, "scen-02")
    sid = str(uuid4())
    response = client.post(
        "/v1/visits/scen-02/anonymized-soap",
        params={
            "session_id": sid,
            "specialization": "medycyna_rodzinna",
            "visit_date": "2026-04-26",
        },
    )
    assert response.status_code == 200, response.text

    # Anonymized output persisted.
    assert (anonymized_root / "scen-02" / "anonymized_soap.json").is_file()

    # Audit + token map written under audit/<date>/<session_id>/.
    session_dirs = list(audit_root.glob(f"*/{sid}"))
    assert len(session_dirs) == 1, f"audit dir not found: {list(audit_root.rglob('*'))}"
    sd = session_dirs[0]
    assert (sd / "audit.json").is_file()
    assert (sd / "trust_gate.json").is_file()
    assert (sd / "token_map.json").is_file()

    # Token map has the redacted name.
    token_map = json.loads((sd / "token_map.json").read_text())
    assert token_map.get("<PERSON_1>") == "Krzysztof Nowakowski"


# ---------------------------------------------------------------------------
# Idempotency.
# ---------------------------------------------------------------------------


def test_second_post_returns_cached(app_with_stubs) -> None:
    client, transcripts_root, _, _ = app_with_stubs
    _write_fake_transcript(transcripts_root, "scen-cache")
    params = {
        "session_id": str(uuid4()),
        "specialization": "medycyna_rodzinna",
        "visit_date": "2026-04-26",
    }
    r1 = client.post("/v1/visits/scen-cache/anonymized-soap", params=params)
    assert r1.status_code == 200
    assert r1.headers.get("X-Cached") == "false"

    # Second call with same transcript_id (different session_id is fine).
    params["session_id"] = str(uuid4())
    r2 = client.post("/v1/visits/scen-cache/anonymized-soap", params=params)
    assert r2.status_code == 200
    assert r2.headers.get("X-Cached") == "true"
    # Cached payload is the same content.
    assert r1.json() == r2.json()


def test_force_true_bypasses_cache(app_with_stubs) -> None:
    client, transcripts_root, _, _ = app_with_stubs
    _write_fake_transcript(transcripts_root, "scen-force")
    params = {
        "session_id": str(uuid4()),
        "specialization": "medycyna_rodzinna",
        "visit_date": "2026-04-26",
    }
    client.post("/v1/visits/scen-force/anonymized-soap", params=params)
    params["force"] = "true"
    r2 = client.post("/v1/visits/scen-force/anonymized-soap", params=params)
    assert r2.status_code == 200
    assert r2.headers.get("X-Cached") == "false"


# ---------------------------------------------------------------------------
# Errors.
# ---------------------------------------------------------------------------


def test_404_on_missing_transcript(app_with_stubs) -> None:
    client, _, _, _ = app_with_stubs
    response = client.post(
        "/v1/visits/nonexistent/anonymized-soap",
        params={
            "session_id": str(uuid4()),
            "specialization": "medycyna_rodzinna",
            "visit_date": "2026-04-26",
        },
    )
    assert response.status_code == 404


def test_500_when_trust_gate_violated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If the anonymizer's `_FakeRegistry` returns no spans, the
    PII slips through unredacted and the trust gate trips.
    """
    transcripts_root = tmp_path / "transcripts"
    transcripts_root.mkdir()
    monkeypatch.setenv("ON_PREMISE_TRANSCRIPTS_ROOT", str(transcripts_root))
    monkeypatch.setenv("ON_PREMISE_ANONYMIZED_ROOT", str(tmp_path / "anonymized"))
    monkeypatch.setenv("AUDIT_ROOT", str(tmp_path / "audit"))
    monkeypatch.setenv("ANONYMIZER_VERSION", "2c.0-test")
    monkeypatch.setenv("DATASET_SCENARIOS_ROOT", str(tmp_path / "no-scenarios"))

    app = create_app()
    # Canned SOAP contains a phone-shaped string — 11 digits in
    # `subjective.chief_complaint`. Empty registry → no redaction →
    # trust gate trips on regex check.
    bad_soap = json.dumps(
        {
            "subjective": {
                "chief_complaint": "Pacjent z telefonem 501-234-567 zgłasza.",
                "history_of_present_illness": "ok",
                "review_of_systems": [],
                "medical_history": [],
                "medications_current": [],
                "allergies": [],
                "provenance": [],
            },
            "objective": {
                "physical_exam": [],
                "vitals": [],
                "labs_mentioned": [],
                "provenance": [],
            },
            "assessment": {
                "primary_diagnosis": "dx",
                "icd10_code": None,
                "differential": [],
                "clinical_reasoning": "",
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
        }
    )
    app.dependency_overrides[get_extractor] = lambda: _StubExtractor(bad_soap)
    app.dependency_overrides[get_anonymizer] = lambda: PresidioAnonymizer(
        registry=_FakeRegistry(canned={})  # detects nothing
    )

    client = TestClient(app)
    _write_fake_transcript(transcripts_root, "scen-bad")
    sid = str(uuid4())
    response = client.post(
        "/v1/visits/scen-bad/anonymized-soap",
        params={
            "session_id": sid,
            "specialization": "medycyna_rodzinna",
            "visit_date": "2026-04-26",
        },
    )
    assert response.status_code == 500
    body = response.json()
    detail = body["detail"]
    if isinstance(detail, dict):
        assert detail["detail"] == "trust gate violation"
        assert any("phone" in v.get("detail", "") for v in detail["violations"])
    else:
        # FastAPI may serialize structured detail as a string in some configs.
        assert "trust gate" in str(detail)
