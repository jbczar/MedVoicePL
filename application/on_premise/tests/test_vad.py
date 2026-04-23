"""Tests for `on_premise.stt.vad.detect_silence_segments`."""

from __future__ import annotations

import numpy as np
import pytest

from on_premise.stt.vad import detect_silence_segments


def test_all_silence_reports_one_span() -> None:
    sr = 16000
    audio = np.zeros(sr, dtype=np.float32)  # 1 s of silence
    spans = detect_silence_segments(audio, sr, min_silence_ms=100)
    assert len(spans) == 1
    assert spans[0][0] == 0.0
    assert spans[0][1] == pytest.approx(1.0, abs=0.02)


def test_silence_then_tone_then_silence() -> None:
    sr = 16000
    # 500 ms silence + 500 ms tone at -6 dBFS + 500 ms silence
    silence = np.zeros(int(sr * 0.5), dtype=np.float32)
    t = np.arange(int(sr * 0.5)) / sr
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    audio = np.concatenate([silence, tone, silence])

    spans = detect_silence_segments(audio, sr, min_silence_ms=300)
    # Two silent spans: one at the head, one at the tail.
    assert len(spans) == 2
    head, tail = spans
    assert head[0] == 0.0
    assert head[1] == pytest.approx(0.5, abs=0.05)
    assert tail[0] == pytest.approx(1.0, abs=0.05)


def test_short_silences_below_threshold_ignored() -> None:
    sr = 16000
    # 100 ms silence — below the 500 ms default min_silence_ms.
    audio = np.zeros(int(sr * 0.1), dtype=np.float32)
    assert detect_silence_segments(audio, sr, min_silence_ms=500) == []


def test_stereo_audio_rejected() -> None:
    sr = 16000
    audio = np.zeros((sr, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="expected mono"):
        detect_silence_segments(audio, sr)


def test_invalid_sample_rate_rejected() -> None:
    audio = np.zeros(100, dtype=np.float32)
    with pytest.raises(ValueError, match="sample_rate"):
        detect_silence_segments(audio, 0)
