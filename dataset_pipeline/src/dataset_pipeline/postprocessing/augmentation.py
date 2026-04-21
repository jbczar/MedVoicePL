"""Audio augmentation: tempo change, pitch shift, reverb.

Phase 1b ships a single utility (`normalize`, peak-normalize to `target_dbfs`)
so the final dialogue.wav has a predictable loudness. Tempo / pitch / reverb
are deferred to Phase 1b.1 together with noise injection — they benefit from
being tuned against the same STT-evaluation loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dataset_pipeline.tts_engine.audio_utils import normalize_peak

if TYPE_CHECKING:
    import numpy as np


def normalize(audio: np.ndarray, target_dbfs: float = -1.0) -> np.ndarray:
    """Peak-normalize mono float32 audio. See `audio_utils.normalize_peak`."""
    return normalize_peak(audio, target_dbfs=target_dbfs)  # type: ignore[no-any-return]


def tempo(audio: np.ndarray, factor: float) -> np.ndarray:  # pragma: no cover
    """Deferred to Phase 1b.1."""
    del audio, factor
    raise NotImplementedError(
        "tempo augmentation is deferred to Phase 1b.1; see CLAUDE.md"
    )


def pitch_shift(audio: np.ndarray, semitones: float) -> np.ndarray:  # pragma: no cover
    """Deferred to Phase 1b.1."""
    del audio, semitones
    raise NotImplementedError(
        "pitch_shift augmentation is deferred to Phase 1b.1; see CLAUDE.md"
    )


def reverb(audio: np.ndarray, room_size: float) -> np.ndarray:  # pragma: no cover
    """Deferred to Phase 1b.1."""
    del audio, room_size
    raise NotImplementedError(
        "reverb augmentation is deferred to Phase 1b.1; see CLAUDE.md"
    )
