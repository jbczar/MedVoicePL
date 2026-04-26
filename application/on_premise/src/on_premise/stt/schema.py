"""Typed contract for transcripts produced by the on-premise STT pipeline.

**TRUST BOUNDARY NOTICE**
This module is importable only from within `application/on_premise/`. The
`Transcript` / `TranscriptSegment` types describe *un-anonymized* speech
content (raw Polish medical dialogue, may contain PESEL, phone numbers,
names). Do NOT re-export any symbol from this module via
`application/shared/schemas/`. Phase 2c adds the real `trust_gate.py`
enforcement; `tests/test_trust_boundary_imports.py` is the preview guard
keeping us honest in the meantime.

The shape mirrors `dataset_pipeline.audio_schema.ScenarioAudioBundle`
(Pydantic v2, `extra="forbid"`, SHA anchors) so JSON artifacts round-trip
cleanly and are debuggable by eye.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class WordTiming(BaseModel):
    """One word emitted by Whisper with its time anchors."""

    model_config = ConfigDict(extra="forbid")

    text: str
    start_s: float = Field(ge=0.0)
    end_s: float = Field(ge=0.0)
    probability: float | None = None  # faster-whisper exposes it; mlx-whisper may not


class TranscriptSegment(BaseModel):
    """One contiguous stretch of speech attributed to a single speaker."""

    model_config = ConfigDict(extra="forbid")

    start_s: float = Field(ge=0.0)
    end_s: float = Field(ge=0.0)
    speaker: Literal["doctor", "patient"]
    text: str
    words: list[WordTiming] = Field(default_factory=list)
    # pyannote returns a per-segment score; MLX path without diarization fills `None`.
    diarization_confidence: float | None = None


class TranscriptionMetadata(BaseModel):
    """Who/what/when produced this transcript. Stable across reruns on same HW."""

    model_config = ConfigDict(extra="forbid")

    stt_backend: Literal["mlx", "faster-whisper"]
    stt_model: str
    stt_device: str
    stt_compute_type: str | None  # None for MLX (no quantization knob)
    diarization_model: str
    diarization_device: str
    transcribed_at: datetime


class Transcript(BaseModel):
    """Full transcript of one audio file, segmented by speaker."""

    model_config = ConfigDict(extra="forbid")

    # `scenario_id` is set when the source is a synthetic bundle; production
    # recordings (future Phase 3) will pass `None` and use `audio_path` alone.
    scenario_id: str | None = None
    audio_path: str  # absolute or repo-relative, forward slashes
    audio_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    duration_s: float = Field(ge=0.0)
    language: str = "pl"
    segments: list[TranscriptSegment] = Field(default_factory=list)
    metadata: TranscriptionMetadata

    @property
    def text(self) -> str:
        """Join segment texts in chronological order. Used by WER scoring."""
        return " ".join(seg.text for seg in self.segments if seg.text)
