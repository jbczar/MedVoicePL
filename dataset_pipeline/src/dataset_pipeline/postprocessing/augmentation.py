"""Audio augmentation for Phase 1b.1: tempo, pitch shift, reverb.

Implementations back onto `pedalboard` (Spotify, MIT-licensed) so we stay
off `librosa` / `numba` / `llvmlite` — that whole stack is slow to install
and fragile on Apple Silicon.

All functions are pure: no RNG inside, no I/O. The caller (pipeline) seeds
and picks parameters, then hands them in. This keeps determinism auditing
simple — if you want reproducibility, seed upstream.

`normalize()` was the only thing shipped in Phase 1b; kept unchanged for
back-compat.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import yaml

from dataset_pipeline.tts_engine.audio_utils import normalize_peak

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np


@dataclass(frozen=True)
class AugmentConfig:
    """Per-turn augmentation — applied before mixing."""

    enabled: bool
    pitch_semitones_min: float
    pitch_semitones_max: float
    tempo_factor_min: float
    tempo_factor_max: float


@dataclass(frozen=True)
class RoomSize:
    name: str
    size: float
    weight: float


@dataclass(frozen=True)
class ReverbConfig:
    """Dialogue-level reverb — applied once after mixing."""

    enabled: bool
    room_sizes: tuple[RoomSize, ...]
    damping: float
    wet_level: float
    dry_level: float


def load_augment_config(config_path: Path) -> AugmentConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    aug = raw.get("augment") or {}
    pitch = aug.get("pitch_semitones") or {}
    tempo_range = aug.get("tempo_factor") or {}
    return AugmentConfig(
        enabled=bool(aug.get("enabled", False)),
        pitch_semitones_min=float(pitch.get("min", -1.0)),
        pitch_semitones_max=float(pitch.get("max", 1.0)),
        tempo_factor_min=float(tempo_range.get("min", 0.95)),
        tempo_factor_max=float(tempo_range.get("max", 1.05)),
    )


def load_reverb_config(config_path: Path) -> ReverbConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    rv = raw.get("reverb") or {}
    rooms_raw = rv.get("room_sizes") or []
    rooms = tuple(
        RoomSize(
            name=str(r.get("name", f"room_{i}")),
            size=float(r["size"]),
            weight=float(r.get("weight", 1.0)),
        )
        for i, r in enumerate(rooms_raw)
    ) or (RoomSize(name="medium", size=0.35, weight=1.0),)
    return ReverbConfig(
        enabled=bool(rv.get("enabled", False)),
        room_sizes=rooms,
        damping=float(rv.get("damping", 0.5)),
        wet_level=float(rv.get("wet_level", 0.25)),
        dry_level=float(rv.get("dry_level", 0.75)),
    )


def normalize(audio: np.ndarray, target_dbfs: float = -1.0) -> np.ndarray:
    """Peak-normalize mono float32 audio. See `audio_utils.normalize_peak`."""
    return normalize_peak(audio, target_dbfs=target_dbfs)  # type: ignore[no-any-return]


def _apply_pedalboard(audio: np.ndarray, sample_rate: int, effect) -> np.ndarray:  # type: ignore[no-untyped-def]
    """Run a single pedalboard effect on a 1-D float32 array.

    pedalboard expects (channels, samples) float32. We guarantee mono in →
    mono out.
    """
    import numpy as np
    from pedalboard import Pedalboard

    # Accept 1-D (preferred) or (1, N).
    flat = np.ascontiguousarray(audio, dtype=np.float32).reshape(-1)
    board = Pedalboard([effect])
    out = board(flat, sample_rate)
    # pedalboard returns 1-D for 1-D input in recent versions; older versions
    # return (1, N). Normalize to 1-D.
    if out.ndim == 2:
        out = out[0]
    return out.astype(np.float32, copy=False)  # type: ignore[no-any-return]


def tempo(audio: np.ndarray, sample_rate: int, factor: float) -> np.ndarray:
    """Change speed WITHOUT changing pitch.

    `factor > 1.0` → faster (shorter output). `factor < 1.0` → slower.
    Typical Phase 1b.1 range: 0.95-1.05 (+/-5%).
    """
    if factor == 1.0:
        import numpy as np

        return audio.astype(np.float32, copy=False)  # type: ignore[no-any-return]

    # pedalboard exposes time-stretch as a standalone function (faster path
    # than going through the Pedalboard chain).
    import numpy as np
    from pedalboard import time_stretch

    flat = np.ascontiguousarray(audio, dtype=np.float32).reshape(-1)
    out = time_stretch(flat, sample_rate, stretch_factor=factor, pitch_shift_in_semitones=0.0)
    if out.ndim == 2:
        out = out[0]
    return out.astype(np.float32, copy=False)  # type: ignore[no-any-return]


def pitch_shift(audio: np.ndarray, sample_rate: int, semitones: float) -> np.ndarray:
    """Shift pitch WITHOUT changing speed. `semitones` is a float (±1 typical)."""
    if semitones == 0.0:
        import numpy as np

        return audio.astype(np.float32, copy=False)  # type: ignore[no-any-return]
    from pedalboard import PitchShift

    return _apply_pedalboard(audio, sample_rate, PitchShift(semitones=semitones))


def reverb(
    audio: np.ndarray,
    sample_rate: int,
    room_size: float,
    *,
    damping: float = 0.5,
    wet_level: float = 0.25,
    dry_level: float = 0.75,
) -> np.ndarray:
    """Add reverb with a room-size parameter in [0, 1].

    Defaults for `damping`/`wet_level`/`dry_level` match the `reverb:` block
    in `audio_postprocess.yaml`. Callers from the pipeline pass the config
    values through; direct callers (tests) can rely on the defaults.
    """
    from pedalboard import Reverb

    return _apply_pedalboard(
        audio,
        sample_rate,
        Reverb(
            room_size=float(room_size),
            damping=float(damping),
            wet_level=float(wet_level),
            dry_level=float(dry_level),
        ),
    )
