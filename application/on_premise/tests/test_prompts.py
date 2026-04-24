"""Prompt rendering tests — no SLM required."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, ValidationError

from on_premise.structuring.prompts import (
    SYSTEM_PROMPT_PL,
    render_correction_hint,
    render_user_prompt,
)
from on_premise.stt.schema import (
    Transcript,
    TranscriptionMetadata,
    TranscriptSegment,
)


def _transcript_with(*segments: tuple[str, str, float, float]) -> Transcript:
    segs = [
        TranscriptSegment(speaker=spk, text=txt, start_s=a, end_s=b)  # type: ignore[arg-type]
        for txt, spk, a, b in segments
    ]
    return Transcript(
        scenario_id="test-01",
        audio_path="/tmp/x.wav",
        audio_sha256="a" * 64,
        duration_s=10.0,
        language="pl",
        segments=segs,
        metadata=TranscriptionMetadata(
            stt_backend="mlx",
            stt_model="mlx-whisper:large-v3",
            stt_device="mps",
            stt_compute_type=None,
            diarization_model="pyannote/speaker-diarization-3.1",
            diarization_device="cpu",
            transcribed_at=datetime.now().astimezone(),
        ),
    )


def test_render_user_prompt_numbers_segments() -> None:
    transcript = _transcript_with(
        ("Dzień dobry.", "doctor", 0.0, 1.0),
        ("Boli mnie gardło.", "patient", 1.1, 2.5),
        ("Od kiedy?", "doctor", 2.6, 3.3),
    )
    rendered = render_user_prompt(transcript, include_few_shot=False)
    assert "[0] doctor: Dzień dobry." in rendered
    assert "[1] patient: Boli mnie gardło." in rendered
    assert "[2] doctor: Od kiedy?" in rendered
    # Instruction to return JSON is appended.
    assert "JSON" in rendered


def test_render_user_prompt_includes_few_shot_by_default() -> None:
    transcript = _transcript_with(("Hej.", "doctor", 0.0, 0.5))
    with_few = render_user_prompt(transcript)
    without = render_user_prompt(transcript, include_few_shot=False)
    assert len(with_few) > len(without)
    assert "Przykład" in with_few
    assert "Przykład" not in without


def test_render_user_prompt_handles_empty_transcript() -> None:
    transcript = _transcript_with()
    rendered = render_user_prompt(transcript, include_few_shot=False)
    assert "(transkrypt pusty)" in rendered


def test_render_user_prompt_skips_empty_segments() -> None:
    # Empty `text` shouldn't produce a `[N] speaker: ` line.
    transcript = _transcript_with(
        ("", "doctor", 0.0, 0.5),
        ("Real content.", "patient", 0.6, 1.0),
    )
    rendered = render_user_prompt(transcript, include_few_shot=False)
    assert "[0] doctor:" not in rendered  # index 0 was empty — skipped
    assert "[1] patient: Real content." in rendered


def test_render_correction_hint_formats_pydantic_errors() -> None:
    class _ToyModel(BaseModel):
        chief_complaint: str = Field(min_length=3)
        icd10_code: str = Field(pattern=r"^[A-Z]\d{2}$")

    try:
        _ToyModel(chief_complaint="", icd10_code="bad")
        raise AssertionError("expected ValidationError")
    except ValidationError as exc:
        hint = render_correction_hint(exc)

    assert "chief_complaint" in hint
    assert "icd10_code" in hint
    assert "JSON" in hint
    assert "Popraw" in hint or "popraw" in hint.lower()


def test_system_prompt_has_anti_hedge_framing() -> None:
    # Regression guard: Bielik (both v2.3 and v3.0) over-refuses clinical
    # prompts unless the task is explicitly reframed as "reformat, don't diagnose".
    assert "Nie stawiasz" in SYSTEM_PROMPT_PL or "reformat" in SYSTEM_PROMPT_PL.lower()
    assert "SOAP" in SYSTEM_PROMPT_PL
    assert "JSON" in SYSTEM_PROMPT_PL
