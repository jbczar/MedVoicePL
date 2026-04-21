"""Unit tests for postprocessing.dialog_mixer.

Feeds synthetic sine-wave turns through the mixer and asserts:
  - output length matches the crossfade + pause math,
  - seeded runs are byte-identical,
  - different seeds produce different outputs,
  - edge cases (empty list, single turn) are handled,
  - yaml loader surfaces the expected defaults.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dataset_pipeline.postprocessing.dialog_mixer import (
    MixerConfig,
    _equal_power_fade,
    load_mixer_config,
    mix_turns,
)


def _sine(freq_hz: float, duration_s: float, sample_rate: int) -> np.ndarray:
    n = int(duration_s * sample_rate)
    t = np.linspace(0.0, duration_s, n, endpoint=False, dtype=np.float32)
    return (0.5 * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)


SR = 22050
CONFIG = MixerConfig(crossfade_ms=200, pause_ms_min=300, pause_ms_max=1500)


def test_empty_input_returns_empty() -> None:
    out = mix_turns([], sample_rate=SR, config=CONFIG, seed=0)
    assert out.shape == (0,)
    assert out.dtype == np.float32


def test_single_turn_returned_as_is() -> None:
    turn = _sine(440.0, 0.5, SR)
    out = mix_turns([turn], sample_rate=SR, config=CONFIG, seed=0)
    assert out.shape == turn.shape
    assert np.array_equal(out, turn)


def test_length_math_matches_formula() -> None:
    """Output length == sum(turns) + sum(pauses) - crossfade*(N-1)."""
    turns = [_sine(220.0, 0.8, SR), _sine(330.0, 0.6, SR), _sine(440.0, 0.4, SR)]
    out = mix_turns(turns, sample_rate=SR, config=CONFIG, seed=123)

    import random as _random

    rng = _random.Random(123)
    pauses_ms = [rng.randint(CONFIG.pause_ms_min, CONFIG.pause_ms_max) for _ in range(2)]
    pause_samples = sum(int(p * SR / 1000) for p in pauses_ms)
    crossfade_samples = int(CONFIG.crossfade_ms * SR / 1000)
    expected = (
        sum(len(t) for t in turns)
        + pause_samples
        - crossfade_samples * (len(turns) - 1)
    )
    assert len(out) == expected
    assert out.dtype == np.float32


def test_determinism_with_same_seed() -> None:
    turns = [_sine(220.0, 0.4, SR), _sine(330.0, 0.4, SR), _sine(440.0, 0.4, SR)]
    a = mix_turns(turns, sample_rate=SR, config=CONFIG, seed=42)
    b = mix_turns(turns, sample_rate=SR, config=CONFIG, seed=42)
    assert np.array_equal(a, b)


def test_different_seeds_produce_different_pauses() -> None:
    turns = [_sine(220.0, 0.4, SR), _sine(330.0, 0.4, SR)]
    a = mix_turns(turns, sample_rate=SR, config=CONFIG, seed=1)
    b = mix_turns(turns, sample_rate=SR, config=CONFIG, seed=999)
    # Lengths should differ because random pauses differ (the expected-value
    # spread of the uniform [300, 1500] ms draw is huge relative to one sample).
    assert len(a) != len(b)


def test_equal_power_curve_invariant() -> None:
    """The core property: fade_out[i]**2 + fade_in[i]**2 == 1 pointwise.

    This is what makes the crossfade "equal-power": for two *uncorrelated*
    signals of equal energy, the expected output power is conserved through
    the overlap. (Correlated signals can interfere constructively or
    destructively — that's signal correlation, not a mixer property.)
    """
    fo, fi = _equal_power_fade(2048)
    assert np.allclose(fo**2 + fi**2, 1.0, atol=1e-5)
    # Endpoints: fade_out starts at 1 & ends at 0; fade_in is the mirror.
    assert fo[0] == pytest.approx(1.0, abs=1e-5)
    assert fo[-1] == pytest.approx(0.0, abs=1e-5)
    assert fi[0] == pytest.approx(0.0, abs=1e-5)
    assert fi[-1] == pytest.approx(1.0, abs=1e-5)


def test_equal_power_fade_zero_length_returns_empty() -> None:
    fo, fi = _equal_power_fade(0)
    assert fo.shape == (0,)
    assert fi.shape == (0,)


def test_crossfade_is_click_free_for_same_level_signals() -> None:
    """With two identical-sign constant signals, the output is smooth.

    That's a lower bar than "constant power" but catches the one bug users
    actually hear: a discontinuity / click where two turns meet.
    """
    cf_ms = 200
    config = MixerConfig(crossfade_ms=cf_ms, pause_ms_min=0, pause_ms_max=0)
    level = 0.5
    a = np.full(int(0.5 * SR), level, dtype=np.float32)
    b = np.full(int(0.5 * SR), level, dtype=np.float32)
    out = mix_turns([a, b], sample_rate=SR, config=config, seed=0)

    # Largest sample-to-sample jump must be small — no discontinuity.
    assert float(np.max(np.abs(np.diff(out)))) < 1e-2


def test_output_dtype_is_float32() -> None:
    turns = [_sine(440.0, 0.2, SR), _sine(880.0, 0.2, SR)]
    out = mix_turns(turns, sample_rate=SR, config=CONFIG, seed=7)
    assert out.dtype == np.float32


def test_load_mixer_config_reads_yaml(tmp_path: Path) -> None:
    cfg_path = tmp_path / "postprocess.yaml"
    cfg_path.write_text(
        """
crossfade:
  duration_ms: 150
  pause_between_turns_ms:
    min: 400
    max: 1200
""".strip(),
        encoding="utf-8",
    )
    cfg = load_mixer_config(cfg_path)
    assert cfg.crossfade_ms == 150
    assert cfg.pause_ms_min == 400
    assert cfg.pause_ms_max == 1200


def test_load_mixer_config_applies_defaults(tmp_path: Path) -> None:
    cfg_path = tmp_path / "empty.yaml"
    cfg_path.write_text("other: key\n", encoding="utf-8")
    cfg = load_mixer_config(cfg_path)
    assert cfg.crossfade_ms == 200
    assert cfg.pause_ms_min == 300
    assert cfg.pause_ms_max == 1500
