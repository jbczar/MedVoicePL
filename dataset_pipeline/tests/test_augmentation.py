"""Tests for `postprocessing.augmentation` (Phase 1b.1 pedalboard-backed FX).

Pedalboard is pre-compiled (no numba/llvmlite), so it's safe to import in
unit tests without dragging the full DSP stack in.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dataset_pipeline.postprocessing.augmentation import (
    AugmentConfig,
    ReverbConfig,
    RoomSize,
    load_augment_config,
    load_reverb_config,
    normalize,
    pitch_shift,
    reverb,
    tempo,
)

SR = 22050


def _sine(duration_s: float = 1.0, freq: float = 220.0) -> np.ndarray:
    t = np.linspace(0, duration_s, int(duration_s * SR), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_tempo_factor_one_is_identity() -> None:
    x = _sine()
    out = tempo(x, SR, factor=1.0)
    assert np.array_equal(out, x.astype(np.float32))


def test_tempo_faster_shortens_output() -> None:
    x = _sine(duration_s=2.0)
    faster = tempo(x, SR, factor=1.2)
    # pedalboard time-stretch is length-accurate within a small rounding.
    expected = len(x) / 1.2
    assert abs(len(faster) - expected) / expected < 0.05


def test_tempo_slower_lengthens_output() -> None:
    x = _sine(duration_s=1.0)
    slower = tempo(x, SR, factor=0.8)
    expected = len(x) / 0.8
    assert abs(len(slower) - expected) / expected < 0.05


def test_pitch_shift_zero_is_identity() -> None:
    x = _sine()
    out = pitch_shift(x, SR, semitones=0.0)
    assert np.array_equal(out, x.astype(np.float32))


def test_pitch_shift_preserves_length() -> None:
    x = _sine(duration_s=1.0)
    out = pitch_shift(x, SR, semitones=2.0)
    # PitchShift is length-preserving (phase-vocoder variant in pedalboard).
    assert abs(len(out) - len(x)) < SR // 50  # ≤20 ms slack


def test_pitch_shift_changes_spectrum() -> None:
    x = _sine(freq=220.0, duration_s=1.0)
    out = pitch_shift(x, SR, semitones=7.0)  # up a fifth
    # Crude check: peak bin of FFT should move up.
    def peak_freq(sig: np.ndarray) -> float:
        spec = np.abs(np.fft.rfft(sig))
        return float(np.fft.rfftfreq(len(sig), 1 / SR)[np.argmax(spec)])

    assert peak_freq(out) > peak_freq(x) + 20.0


def test_reverb_changes_waveform_length_preserved() -> None:
    x = _sine(duration_s=1.0)
    out = reverb(x, SR, room_size=0.5)
    assert abs(len(out) - len(x)) < SR // 50
    assert not np.allclose(out, x)


def test_normalize_pushes_peak_to_target() -> None:
    x = (0.1 * _sine()).astype(np.float32)
    out = normalize(x, target_dbfs=-1.0)
    peak = float(np.max(np.abs(out)))
    # -1 dBFS ≈ 0.891
    assert abs(peak - 10 ** (-1.0 / 20.0)) < 1e-3


def test_load_augment_config_defaults(tmp_path: Path) -> None:
    cfg_path = tmp_path / "post.yaml"
    cfg_path.write_text("augment: {}\n", encoding="utf-8")
    cfg = load_augment_config(cfg_path)
    assert isinstance(cfg, AugmentConfig)
    assert cfg.enabled is False
    assert cfg.pitch_semitones_min == pytest.approx(-1.0)
    assert cfg.tempo_factor_max == pytest.approx(1.05)


def test_load_reverb_config_parses_rooms(tmp_path: Path) -> None:
    cfg_path = tmp_path / "post.yaml"
    cfg_path.write_text(
        """
reverb:
  enabled: true
  damping: 0.4
  wet_level: 0.3
  dry_level: 0.7
  room_sizes:
    - { name: tiny, size: 0.1, weight: 0.5 }
    - { name: huge, size: 0.9, weight: 0.5 }
""".strip(),
        encoding="utf-8",
    )
    cfg = load_reverb_config(cfg_path)
    assert isinstance(cfg, ReverbConfig)
    assert cfg.enabled is True
    assert cfg.damping == pytest.approx(0.4)
    assert len(cfg.room_sizes) == 2
    assert cfg.room_sizes[0] == RoomSize(name="tiny", size=0.1, weight=0.5)
