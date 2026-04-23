"""Tests for `on_premise.stt.diarization`.

No pyannote load: the pipeline factory only constructs when a valid HF token
is present, so we test the missing-token path and the Protocol contract with
a fake. Real pyannote inference is exercised by the end-to-end CLI smoke test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from on_premise.stt.diarization import (
    Diarizer,
    RawDiarizationSegment,
    build_diarizer,
)


class _FakeDiarizer:
    """Satisfies the Diarizer Protocol without loading pyannote."""

    name = "fake/diarizer"
    device = "cpu"

    def __init__(self, segments: list[RawDiarizationSegment] | None = None) -> None:
        self._segs = segments or []

    def diarize(self, wav_path: Path) -> list[RawDiarizationSegment]:
        return list(self._segs)


def test_fake_diarizer_satisfies_protocol() -> None:
    d: Diarizer = _FakeDiarizer(
        [RawDiarizationSegment(start_s=0.0, end_s=1.5, speaker_label="SPEAKER_00")]
    )
    result = d.diarize(Path("/dev/null"))
    assert len(result) == 1
    assert result[0].speaker_label == "SPEAKER_00"


def test_build_diarizer_without_token_raises() -> None:
    with pytest.raises(RuntimeError, match="HUGGINGFACE_TOKEN is required"):
        build_diarizer(config=None, env={"HUGGINGFACE_TOKEN": ""})


def test_build_diarizer_token_stripped() -> None:
    # Even whitespace-only token should be treated as missing.
    with pytest.raises(RuntimeError, match="HUGGINGFACE_TOKEN is required"):
        build_diarizer(config=None, env={"HUGGINGFACE_TOKEN": "   "})
