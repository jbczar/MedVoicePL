"""Background noise injection for Phase 1b.1.

Mixes a randomly-selected clip from the noise library into clean speech at a
deterministically-sampled SNR. Clips come from `dataset_pipeline/noise_library/`
(populated by `scripts/fetch_noise_library.py` — CC0/CC-BY clinical ambience:
keyboard clicks, HVAC hum, paper rustle, distant chatter, etc.).

Algorithm per call:
  1. Seed a local RNG from the caller-supplied `seed` (stable per scenario).
  2. Pick an SNR bucket (stratified by weight), then a dB value within it.
  3. Pick a clip from the library (uniform).
  4. Loop or trim the clip to match the speech duration.
  5. Scale the noise so that 20*log10(RMS_speech / RMS_noise) == target SNR.
  6. Return speech + noise, clipped to [-1, 1].

The signature accepts `audio` + `sample_rate` + `config` + `seed`. `seed` is
required (no default) — Phase 1b.1 is explicitly about reproducibility, and
making callers think about the seed prevents accidental non-determinism.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import yaml

from dataset_pipeline.tts_engine.audio_utils import load_wav

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SnrBucket:
    min_db: float
    max_db: float
    weight: float


@dataclass(frozen=True)
class NoiseConfig:
    enabled: bool
    library_path: Path  # absolute or relative to the config file's directory
    snr_buckets: tuple[SnrBucket, ...]


def load_noise_config(config_path: Path) -> NoiseConfig:
    """Parse the `noise:` block from `configs/audio_postprocess.yaml`.

    Resolves `library_path` relative to the config file's directory (so the
    yaml can stay the same whether it's loaded from repo root or a CLI run).
    """
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    noise = raw.get("noise") or {}
    buckets_raw = noise.get("snr_buckets") or []
    buckets = tuple(
        SnrBucket(
            min_db=float(b["min_db"]),
            max_db=float(b["max_db"]),
            weight=float(b.get("weight", 1.0)),
        )
        for b in buckets_raw
    )
    if not buckets:
        # Safe default so tests / smoke runs without the yaml still work.
        buckets = (SnrBucket(min_db=15.0, max_db=20.0, weight=1.0),)

    lib = noise.get("library_path") or "noise_library"
    lib_path = (config_path.parent / lib).resolve() if not str(lib).startswith("/") else __import__("pathlib").Path(lib)

    return NoiseConfig(
        enabled=bool(noise.get("enabled", False)),
        library_path=lib_path,
        snr_buckets=buckets,
    )


def _list_noise_clips(library_path: Path) -> list[Path]:
    """Return sorted WAVs in the noise library (sorted = deterministic order)."""
    if not library_path.exists():
        return []
    return sorted(p for p in library_path.iterdir() if p.suffix.lower() == ".wav")


def _rms(x: np.ndarray) -> float:
    """Root-mean-square of a float array. Returns 0 for silent input."""
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def _pick_bucket(rng: random.Random, buckets: tuple[SnrBucket, ...]) -> SnrBucket:
    weights = [b.weight for b in buckets]
    return rng.choices(buckets, weights=weights, k=1)[0]


def _fit_length(noise: np.ndarray, target_len: int) -> np.ndarray:
    """Loop or trim `noise` to exactly `target_len` samples."""
    if noise.size == 0:
        return np.zeros(target_len, dtype=np.float32)
    if noise.size >= target_len:
        return noise[:target_len].astype(np.float32, copy=False)
    # Loop the clip. `np.tile` + slice is fine — clips are tens of seconds,
    # target is at most a few minutes, so the allocation is modest.
    reps = (target_len + noise.size - 1) // noise.size
    return np.tile(noise, reps)[:target_len].astype(np.float32, copy=False)


def inject(
    audio: np.ndarray,
    sample_rate: int,
    config: NoiseConfig,
    *,
    seed: int,
) -> np.ndarray:
    """Mix ambient noise into `audio` at a seeded SNR from `config.snr_buckets`.

    Returns a mono float32 array the same length as `audio`, clipped to [-1,1].
    If `config.enabled` is False, returns `audio` unchanged.

    Raises:
      FileNotFoundError — noise library missing or empty.
    """
    if not config.enabled:
        return audio.astype(np.float32, copy=False)  # type: ignore[no-any-return]
    if audio.size == 0:
        return audio.astype(np.float32, copy=False)  # type: ignore[no-any-return]

    rng = random.Random(seed)
    clips = _list_noise_clips(config.library_path)
    if not clips:
        raise FileNotFoundError(
            f"noise library empty: {config.library_path}; "
            f"run dataset_pipeline/scripts/fetch_noise_library.py"
        )

    bucket = _pick_bucket(rng, config.snr_buckets)
    target_snr_db = rng.uniform(bucket.min_db, bucket.max_db)
    clip_path = rng.choice(clips)

    noise_raw, clip_sr = load_wav(clip_path)
    if clip_sr != sample_rate:
        # Caller is responsible for this never happening in production — the
        # fetch script writes at 22050. But resample defensively so a
        # mismatched library doesn't crash the whole pipeline run.
        from dataset_pipeline.tts_engine.audio_utils import resample

        noise_raw = resample(noise_raw, clip_sr, sample_rate)

    noise = _fit_length(noise_raw, audio.size)

    speech_rms = _rms(audio)
    noise_rms = _rms(noise)
    if speech_rms == 0.0 or noise_rms == 0.0:
        # Degenerate case: silent speech or silent noise clip. No-op rather
        # than divide-by-zero.
        return audio.astype(np.float32, copy=False)  # type: ignore[no-any-return]

    # gain * noise_rms == speech_rms * 10^(-snr/20)
    target_noise_rms = speech_rms * (10.0 ** (-target_snr_db / 20.0))
    gain = target_noise_rms / noise_rms
    mixed = audio.astype(np.float32, copy=False) + (noise * gain).astype(
        np.float32, copy=False
    )

    log.debug(
        "noise inject: clip=%s snr_db=%.1f bucket=[%.0f,%.0f] gain=%.4f",
        clip_path.name,
        target_snr_db,
        bucket.min_db,
        bucket.max_db,
        gain,
    )

    return np.clip(mixed, -1.0, 1.0).astype(np.float32, copy=False)  # type: ignore[no-any-return]


def pick_params(
    config: NoiseConfig, *, seed: int
) -> tuple[str | None, float | None]:
    """Return (clip_name, snr_db) that `inject()` *would* pick with this seed.

    Used by the pipeline to populate manifest metadata without having to
    refactor `inject()` to return both audio and metadata. Both values share
    the same RNG sequence as `inject()` so they stay in sync.

    Returns (None, None) if noise is disabled.
    """
    if not config.enabled:
        return None, None
    rng = random.Random(seed)
    clips = _list_noise_clips(config.library_path)
    if not clips:
        return None, None
    bucket = _pick_bucket(rng, config.snr_buckets)
    snr_db = rng.uniform(bucket.min_db, bucket.max_db)
    clip_path = rng.choice(clips)
    return clip_path.name, snr_db
