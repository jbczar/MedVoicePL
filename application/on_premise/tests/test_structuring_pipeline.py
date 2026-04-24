"""Tests for `on_premise.structuring.pipeline.transcript_to_soap`.

No model downloads — a `_StubExtractor` returns canned JSON strings.
Covers:
  - happy path (valid JSON validates on first try)
  - validation retry (first call returns bad JSON, second call valid)
  - PII hallucination scrub drops PESEL/phone not in transcript
  - metadata determinism across identical transcripts
  - extractor failure surfaces as SOAPExtractionError
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from typing import Any

import pytest

from on_premise.structuring.pipeline import (
    SOAPExtractionError,
    transcript_to_soap,
)
from on_premise.stt.schema import (
    Transcript,
    TranscriptionMetadata,
    TranscriptSegment,
)


class _StubExtractor:
    """Returns `responses[i]` on call `i`, cycling through a list."""

    name = "mlx-lm:stub"
    device = "mps"
    structured_mode = "grammar"

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self.calls = 0

    def extract(self, prompt: str, *, schema: Mapping[str, Any]) -> str:
        idx = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        return self._responses[idx]


def _transcript(text: str = "Dzień dobry. Boli mnie głowa.") -> Transcript:
    return Transcript(
        scenario_id="test-01",
        audio_path="/tmp/x.wav",
        audio_sha256="b" * 64,
        duration_s=5.0,
        language="pl",
        segments=[
            TranscriptSegment(speaker="doctor", text="Dzień dobry.", start_s=0.0, end_s=1.0),
            TranscriptSegment(speaker="patient", text=text, start_s=1.1, end_s=4.5),
        ],
        metadata=TranscriptionMetadata(
            stt_backend="mlx",
            stt_model="mlx-whisper:large-v3",
            stt_device="mps",
            stt_compute_type=None,
            diarization_model="pyannote/speaker-diarization-3.1",
            diarization_device="cpu",
            transcribed_at=datetime(2026, 4, 24, 12, 0, 0).astimezone(),
        ),
    )


def _valid_soap_json(*, cc: str = "ból głowy", icd10: str | None = "R51") -> str:
    """Build a minimal-but-valid SOAPNote JSON for the stub to return.

    Note: `metadata` is included because the schema requires it, but the
    pipeline overwrites it — we still need a syntactically valid block.
    """
    return json.dumps(
        {
            "subjective": {
                "chief_complaint": cc,
                "history_of_present_illness": "od rana.",
                "review_of_systems": [],
                "medical_history": [],
                "medications_current": [],
                "allergies": [],
                "provenance": [1],
            },
            "objective": {
                "physical_exam": [],
                "vitals": [],
                "labs_mentioned": [],
                "provenance": [],
            },
            "assessment": {
                "primary_diagnosis": "napięciowy ból głowy",
                "icd10_code": icd10,
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
            "metadata": {
                "source_transcript_sha256": "c" * 64,
                "source_audio_sha256": "b" * 64,
                "scenario_id": "test-01",
                "slm_backend": "mlx-lm",
                "slm_model": "stub",
                "slm_device": "cpu",
                "slm_structured_mode": "grammar",
                "generated_at": "2026-04-24T12:00:00+00:00",
                "generation_duration_s": 0.0,
            },
        }
    )


def test_happy_path_validates_first_try() -> None:
    extractor = _StubExtractor([_valid_soap_json()])
    soap = transcript_to_soap(_transcript(), extractor=extractor)
    assert extractor.calls == 1
    assert soap.subjective.chief_complaint == "ból głowy"
    assert soap.assessment.icd10_code == "R51"
    # Metadata gets rebuilt by the pipeline.
    assert soap.metadata.slm_backend == "mlx-lm"
    assert soap.metadata.slm_model == "stub"
    assert soap.metadata.source_audio_sha256 == "b" * 64


def test_retry_on_bad_json_then_succeeds() -> None:
    # First response: broken (empty string -> JSON parse fail).
    # Second response: valid.
    extractor = _StubExtractor(["not json at all {", _valid_soap_json()])
    soap = transcript_to_soap(_transcript(), extractor=extractor, max_validation_retries=2)
    assert extractor.calls == 2
    assert soap.subjective.chief_complaint == "ból głowy"


def test_retry_exhausted_raises() -> None:
    extractor = _StubExtractor(["garbage" * 4])
    with pytest.raises(SOAPExtractionError):
        transcript_to_soap(_transcript(), extractor=extractor, max_validation_retries=1)
    # 1 initial + 1 retry = 2 calls
    assert extractor.calls == 2


def test_pii_scrub_drops_hallucinated_pesel() -> None:
    # Transcript does NOT contain any PESEL.
    # SOAP output is tampered to include a fake 11-digit number in HPI.
    fake_pesel = "90010112345"
    tampered = _valid_soap_json().replace(
        "od rana.",
        f"od rana, pacjent podał PESEL {fake_pesel}.",
    )
    extractor = _StubExtractor([tampered])
    soap = transcript_to_soap(_transcript(), extractor=extractor)
    # The PESEL-shaped substring should be stripped (replaced with "").
    assert fake_pesel not in soap.subjective.history_of_present_illness


def test_pii_scrub_keeps_pesel_that_is_in_transcript() -> None:
    real_pesel = "85050512345"
    transcript = _transcript(text=f"Mój PESEL to {real_pesel}, doktorze.")
    tampered = _valid_soap_json().replace(
        "od rana.",
        f"pacjent podał PESEL {real_pesel}.",
    )
    extractor = _StubExtractor([tampered])
    soap = transcript_to_soap(transcript, extractor=extractor)
    # The real PESEL is in the source transcript -> NOT scrubbed.
    assert real_pesel in soap.subjective.history_of_present_illness


def test_source_transcript_sha256_deterministic() -> None:
    t1 = _transcript()
    t2 = _transcript()
    e1 = _StubExtractor([_valid_soap_json()])
    e2 = _StubExtractor([_valid_soap_json()])
    soap1 = transcript_to_soap(t1, extractor=e1)
    soap2 = transcript_to_soap(t2, extractor=e2)
    assert soap1.metadata.source_transcript_sha256 == soap2.metadata.source_transcript_sha256


def test_extractor_exception_surfaces_as_soapextractionerror() -> None:
    class _Boom:
        name = "mlx-lm:boom"
        device = "mps"
        structured_mode = "grammar"

        def extract(self, prompt: str, *, schema: Mapping[str, Any]) -> str:
            raise RuntimeError("out of memory")

    with pytest.raises(SOAPExtractionError, match="extractor failed"):
        transcript_to_soap(_transcript(), extractor=_Boom())
