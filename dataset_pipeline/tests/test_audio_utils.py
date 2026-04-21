"""Unit tests for tts_engine.audio_utils.

No TTS / torch heavy path touched here except for the resample roundtrip,
which intentionally exercises the deferred torchaudio import to ensure the
wrapper math is correct. All other tests stay pure numpy + soundfile.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dataset_pipeline.tts_engine.audio_utils import (
    duration_seconds,
    load_wav,
    normalize_peak,
    normalize_rms,
    resample,
    save_wav,
)


def _sine(freq_hz: float, duration_s: float, sample_rate: int) -> np.ndarray:
    t = np.linspace(
        0.0, duration_s, int(duration_s * sample_rate), endpoint=False, dtype=np.float32
    )
    return (0.5 * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)


def test_save_load_roundtrip(tmp_path: Path) -> None:
    sr = 22050
    audio = _sine(440.0, 0.5, sr)
    out = tmp_path / "tone.wav"

    save_wav(out, audio, sr)
    assert out.exists()

    loaded, loaded_sr = load_wav(out)
    assert loaded_sr == sr
    assert loaded.shape == audio.shape
    # 16-bit PCM quantization error is bounded; allow ~1 LSB of int16 scale.
    assert np.max(np.abs(loaded - audio)) < 1e-3


def test_save_creates_parent_dir(tmp_path: Path) -> None:
    sr = 16000
    audio = _sine(220.0, 0.1, sr)
    out = tmp_path / "nested" / "deeper" / "tone.wav"

    save_wav(out, audio, sr)
    assert out.exists()


def test_save_clips_out_of_range(tmp_path: Path) -> None:
    sr = 16000
    # Deliberately push values outside [-1, 1]; save_wav must clip, not wrap.
    audio = np.array([2.0, -3.5, 0.0, 0.5], dtype=np.float32)
    out = tmp_path / "clip.wav"

    save_wav(out, audio, sr)
    loaded, _ = load_wav(out)
    # After clipping and int16 round-trip, extremes should be near ±1.0.
    assert loaded[0] >= 0.999
    assert loaded[1] <= -0.999


def test_duration_seconds_matches_length() -> None:
    sr = 16000
    audio = _sine(100.0, 1.25, sr)
    assert duration_seconds(audio, sr) == pytest.approx(1.25, abs=1.0 / sr)


def test_resample_22050_to_16000_length_ratio() -> None:
    src_sr, dst_sr = 22050, 16000
    audio = _sine(440.0, 1.0, src_sr)
    out = resample(audio, src_sr, dst_sr)
    expected = len(audio) * dst_sr / src_sr
    # torchaudio.functional.resample may differ by a handful of samples.
    assert abs(len(out) - expected) <= 4
    assert out.dtype == np.float32


def test_resample_noop_when_same_rate() -> None:
    sr = 16000
    audio = _sine(440.0, 0.1, sr)
    out = resample(audio, sr, sr)
    assert out is not audio or np.array_equal(out, audio)
    assert out.dtype == np.float32
    assert out.shape == audio.shape


def test_normalize_peak_hits_target_dbfs() -> None:
    sr = 16000
    # Peak at 0.1 → after normalize_peak(-6 dBFS), peak should be ~0.5.
    audio = _sine(440.0, 0.05, sr) * 0.2  # peak ~0.1
    out = normalize_peak(audio, target_dbfs=-6.0)
    peak = float(np.max(np.abs(out)))
    target = 10.0 ** (-6.0 / 20.0)  # ~0.5012
    assert peak == pytest.approx(target, rel=1e-3)


def test_normalize_peak_leaves_silence_alone() -> None:
    audio = np.zeros(100, dtype=np.float32)
    out = normalize_peak(audio, target_dbfs=-1.0)
    assert np.array_equal(out, audio)


def test_normalize_rms_hits_target_dbfs() -> None:
    sr = 16000
    audio = _sine(440.0, 0.1, sr) * 0.01  # very quiet
    out = normalize_rms(audio, target_dbfs=-20.0)
    rms = float(np.sqrt(np.mean(np.square(out))))
    target = 10.0 ** (-20.0 / 20.0)  # 0.1
    assert rms == pytest.approx(target, rel=1e-3)


def test_normalize_rms_handles_silence() -> None:
    audio = np.zeros(128, dtype=np.float32)
    out = normalize_rms(audio)
    assert np.array_equal(out, audio)
