"""Tests for `on_premise.stt.schema` — the trust-boundary contract."""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from on_premise.stt.schema import (
    Transcript,
    TranscriptionMetadata,
    TranscriptSegment,
    WordTiming,
)


def _metadata() -> TranscriptionMetadata:
    return TranscriptionMetadata(
        stt_backend="mlx",
        stt_model="mlx-whisper:large-v3",
        stt_device="mps",
        stt_compute_type=None,
        diarization_model="pyannote/speaker-diarization-3.1",
        diarization_device="mps",
        transcribed_at=datetime(2026, 4, 22, 14, 0, 0),
    )


def test_transcript_roundtrip_json() -> None:
    t = Transcript(
        scenario_id="endo-acromegaly-01",
        audio_path="dataset_pipeline/output/audio/endo-acr/dialogue.wav",
        audio_sha256="0" * 64,
        duration_s=185.5,
        segments=[
            TranscriptSegment(
                start_s=0.0,
                end_s=3.2,
                speaker="doctor",
                text="Dzień dobry, proszę usiąść.",
                words=[WordTiming(text="Dzień", start_s=0.0, end_s=0.4)],
                diarization_confidence=None,
            ),
        ],
        metadata=_metadata(),
    )
    dumped = t.model_dump_json()
    assert Transcript.model_validate_json(dumped) == t


def test_transcript_text_joins_segments_in_order() -> None:
    t = Transcript(
        audio_path="/tmp/foo.wav",
        audio_sha256="a" * 64,
        duration_s=5.0,
        segments=[
            TranscriptSegment(start_s=0.0, end_s=1.0, speaker="doctor", text="Witam."),
            TranscriptSegment(start_s=1.2, end_s=2.0, speaker="patient", text="Dzień dobry."),
        ],
        metadata=_metadata(),
    )
    assert t.text == "Witam. Dzień dobry."


def test_transcript_text_skips_empty_segments() -> None:
    t = Transcript(
        audio_path="/tmp/foo.wav",
        audio_sha256="a" * 64,
        duration_s=5.0,
        segments=[
            TranscriptSegment(start_s=0.0, end_s=1.0, speaker="doctor", text=""),
            TranscriptSegment(start_s=1.2, end_s=2.0, speaker="patient", text="Hej."),
        ],
        metadata=_metadata(),
    )
    assert t.text == "Hej."


def test_extra_fields_rejected_on_transcript() -> None:
    with pytest.raises(ValidationError):
        Transcript(
            audio_path="/tmp/foo.wav",
            audio_sha256="a" * 64,
            duration_s=1.0,
            segments=[],
            metadata=_metadata(),
            something_extra="nope",  # type: ignore[call-arg]
        )


def test_invalid_sha_rejected() -> None:
    with pytest.raises(ValidationError):
        Transcript(
            audio_path="/tmp/foo.wav",
            audio_sha256="not-hex",
            duration_s=1.0,
            segments=[],
            metadata=_metadata(),
        )


def test_speaker_literal_enforced() -> None:
    with pytest.raises(ValidationError):
        TranscriptSegment(
            start_s=0.0,
            end_s=1.0,
            speaker="nurse",  # type: ignore[arg-type]
            text="...",
        )
