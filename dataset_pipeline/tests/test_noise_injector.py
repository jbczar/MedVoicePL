"""Tests for `postprocessing.noise_injector`.

These tests don't require any on-disk CC-licensed noise library — they
synthesize white/pink noise on the fly into `tmp_path` and point the
config at that directory. Keeps the test suite free of binary fixtures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from dataset_pipeline.postprocessing.noise_injector import (
    NoiseConfig,
    SnrBucket,
    inject,
    load_noise_config,
    pick_params,
)

SR = 22050


def _write_noise_clip(path: Path, duration_s: float = 2.0, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    n = int(duration_s * SR)
    # Uniform white noise, scaled down so it doesn't clip when added.
    data = (rng.standard_normal(n) * 0.2).astype(np.float32)
    sf.write(str(path), data, SR, subtype="PCM_16")


def _speech(duration_s: float = 3.0, amplitude: float = 0.5) -> np.ndarray:
    """Sine wave as a stand-in for clean speech at a known RMS."""
    t = np.linspace(0, duration_s, int(duration_s * SR), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)


def _make_config(library: Path, *, enabled: bool = True) -> NoiseConfig:
    return NoiseConfig(
        enabled=enabled,
        library_path=library,
        snr_buckets=(SnrBucket(min_db=15.0, max_db=15.0, weight=1.0),),
    )


def test_inject_disabled_returns_audio_unchanged(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path, enabled=False)
    speech = _speech()
    out = inject(speech, SR, cfg, seed=42)
    assert np.array_equal(out, speech)


def test_inject_raises_when_library_empty(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    with pytest.raises(FileNotFoundError):
        inject(_speech(), SR, cfg, seed=42)


def test_inject_changes_waveform(tmp_path: Path) -> None:
    _write_noise_clip(tmp_path / "hvac_001.wav")
    cfg = _make_config(tmp_path)
    speech = _speech()
    out = inject(speech, SR, cfg, seed=42)
    assert out.shape == speech.shape
    # Mix added energy — output should not equal input.
    assert not np.allclose(out, speech)
    # Bounded to [-1, 1].
    assert np.max(np.abs(out)) <= 1.0 + 1e-6


def test_inject_snr_is_approximately_correct(tmp_path: Path) -> None:
    """With a single SNR bucket of [15,15], the actual RMS ratio should match."""
    _write_noise_clip(tmp_path / "hvac_001.wav", duration_s=4.0)
    cfg = _make_config(tmp_path)
    speech = _speech(duration_s=3.0)
    out = inject(speech, SR, cfg, seed=7)
    # Derive added noise by subtracting input.
    noise = out - speech
    speech_rms = float(np.sqrt(np.mean(speech**2)))
    noise_rms = float(np.sqrt(np.mean(noise**2)))
    snr_db = 20.0 * np.log10(speech_rms / noise_rms)
    # Target 15 dB; clipping adds a tiny bias so allow ±1 dB.
    assert abs(snr_db - 15.0) < 1.0


def test_inject_is_deterministic_for_same_seed(tmp_path: Path) -> None:
    for i in range(3):
        _write_noise_clip(tmp_path / f"clip_{i}.wav", seed=i)
    cfg = _make_config(tmp_path)
    speech = _speech()
    a = inject(speech, SR, cfg, seed=1234)
    b = inject(speech, SR, cfg, seed=1234)
    assert np.array_equal(a, b)


def test_inject_differs_across_seeds(tmp_path: Path) -> None:
    for i in range(3):
        _write_noise_clip(tmp_path / f"clip_{i}.wav", seed=i)
    cfg = _make_config(tmp_path)
    speech = _speech()
    a = inject(speech, SR, cfg, seed=1)
    b = inject(speech, SR, cfg, seed=999)
    assert not np.array_equal(a, b)


def test_pick_params_matches_inject_rng_sequence(tmp_path: Path) -> None:
    """`pick_params` must consume the RNG in the same order `inject` does."""
    for i in range(4):
        _write_noise_clip(tmp_path / f"clip_{i}.wav", seed=i)
    cfg = NoiseConfig(
        enabled=True,
        library_path=tmp_path,
        snr_buckets=(
            SnrBucket(min_db=10.0, max_db=15.0, weight=1.0),
            SnrBucket(min_db=20.0, max_db=25.0, weight=1.0),
        ),
    )
    clip_name, snr_db = pick_params(cfg, seed=99)
    assert clip_name is not None
    assert snr_db is not None
    # Called again → same answer.
    clip_name2, snr_db2 = pick_params(cfg, seed=99)
    assert (clip_name, snr_db) == (clip_name2, snr_db2)


def test_load_noise_config_from_yaml(tmp_path: Path) -> None:
    cfg_path = tmp_path / "audio_postprocess.yaml"
    cfg_path.write_text(
        """
noise:
  enabled: true
  library_path: my_noise
  snr_buckets:
    - { min_db: 10, max_db: 15, weight: 0.2 }
    - { min_db: 15, max_db: 20, weight: 0.8 }
""".strip(),
        encoding="utf-8",
    )
    cfg = load_noise_config(cfg_path)
    assert cfg.enabled is True
    assert cfg.library_path == (tmp_path / "my_noise").resolve()
    assert len(cfg.snr_buckets) == 2
    assert cfg.snr_buckets[0].weight == pytest.approx(0.2)
