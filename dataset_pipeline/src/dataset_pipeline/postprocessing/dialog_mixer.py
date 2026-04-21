"""Doctor-patient dialogue crossfading and pause insertion.

Mixes N per-turn arrays into a single dialogue track by:
  1. inserting a silent pause (randomized within a configured range) between
     each consecutive pair of turns,
  2. overlapping the previous turn's tail with the next turn's head using an
     equal-power (sqrt) crossfade.

Params come from `configs/audio_postprocess.yaml`
(`crossfade.duration_ms`, `crossfade.pause_between_turns_ms`).

The random pause is seeded, so a given `(scenario_id, turn_count)` pair
produces byte-identical output across reruns — that lets the manifest SHA-256
stay stable for dialogue.wav.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import yaml

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class MixerConfig:
    crossfade_ms: int
    pause_ms_min: int
    pause_ms_max: int


def load_mixer_config(config_path: Path) -> MixerConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    crossfade = raw.get("crossfade") or {}
    pause_range = crossfade.get("pause_between_turns_ms") or {}
    return MixerConfig(
        crossfade_ms=int(crossfade.get("duration_ms", 200)),
        pause_ms_min=int(pause_range.get("min", 300)),
        pause_ms_max=int(pause_range.get("max", 1500)),
    )


def _equal_power_fade(length: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (fade_out, fade_in) equal-power curves of `length` samples."""
    if length <= 0:
        return np.ones(0, dtype=np.float32), np.ones(0, dtype=np.float32)
    t = np.linspace(0.0, 1.0, num=length, endpoint=True, dtype=np.float32)
    fade_out = np.sqrt(1.0 - t, dtype=np.float32)
    fade_in = np.sqrt(t, dtype=np.float32)
    return fade_out, fade_in


def mix_turns(
    turns: list[np.ndarray],
    *,
    sample_rate: int,
    config: MixerConfig,
    seed: int | None = None,
) -> np.ndarray:
    """Concatenate per-turn arrays with inter-turn pauses and crossfades.

    Returns a mono float32 array at `sample_rate`. The output length equals:
        sum(len(t) for t in turns)
      + sum(pause_samples between turns)
      - crossfade_samples * (N - 1)

    where each pause is drawn from [pause_ms_min, pause_ms_max] (uniform).
    """
    if not turns:
        return np.zeros(0, dtype=np.float32)
    if len(turns) == 1:
        return turns[0].astype(np.float32, copy=False)

    rng = random.Random(seed)
    crossfade_samples = max(0, int(config.crossfade_ms * sample_rate / 1000))
    fade_out, fade_in = _equal_power_fade(crossfade_samples)

    # Start with the first turn.
    out = turns[0].astype(np.float32, copy=True)
    for current in turns[1:]:
        current = current.astype(np.float32, copy=False)
        pause_ms = rng.randint(config.pause_ms_min, config.pause_ms_max)
        pause_samples = int(pause_ms * sample_rate / 1000)
        silence = np.zeros(pause_samples, dtype=np.float32)

        # Add silence, then prepare to crossfade into `current`.
        # The crossfade overlaps the end of (out + silence) with the start of
        # `current`. If either segment is shorter than the fade, we clip the
        # fade region to min length so we still produce sensible audio.
        cf = min(crossfade_samples, len(current), len(out) + len(silence))
        if cf <= 0:
            out = np.concatenate([out, silence, current])
            continue

        # Apply fade-out to the tail of (out + silence). Since silence is 0,
        # applying fade-out to its tail does nothing; but the logical tail for
        # the crossfade includes whichever samples sit in the last `cf` slots
        # of (out + silence).
        combined = np.concatenate([out, silence])
        head = combined[:-cf]
        tail = combined[-cf:]
        fade_out_local, fade_in_local = _equal_power_fade(cf) if cf != crossfade_samples else (fade_out, fade_in)
        mixed = tail * fade_out_local + current[:cf] * fade_in_local
        rest = current[cf:]
        out = np.concatenate([head, mixed, rest]).astype(np.float32, copy=False)

    return out
