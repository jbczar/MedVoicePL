"""Audio format conversion, resampling, and normalization utilities.

Thin wrappers over `soundfile` (WAV I/O) and `torchaudio.functional` (resample)
so downstream callers do not import torch directly and can be unit-tested with
synthetic numpy arrays.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import soundfile as sf

if TYPE_CHECKING:
    from pathlib import Path


def save_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    """Write a mono float32 array to a 16-bit PCM WAV file.

    Values are clipped to [-1.0, 1.0] before writing. The parent directory is
    created if it does not exist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio.astype(np.float32, copy=False), -1.0, 1.0)
    sf.write(str(path), clipped, sample_rate, subtype="PCM_16")


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read a WAV file into a mono float32 array + its sample rate."""
    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data.astype(np.float32, copy=False), int(sample_rate)


def resample(audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    """Resample a mono float32 array from `from_sr` to `to_sr` using torchaudio.

    Import is deferred so modules that only need save/load do not pay the torch
    import cost and so tests that never touch resampling stay independent of
    torch availability.
    """
    if from_sr == to_sr:
        return audio.astype(np.float32, copy=False)  # type: ignore[no-any-return]

    import torch
    from torchaudio.functional import resample as ta_resample

    tensor = torch.from_numpy(audio.astype(np.float32, copy=False)).unsqueeze(0)
    out = ta_resample(tensor, from_sr, to_sr).squeeze(0).cpu().numpy()
    return out.astype(np.float32, copy=False)  # type: ignore[no-any-return]


def normalize_peak(audio: np.ndarray, target_dbfs: float = -1.0) -> np.ndarray:
    """Scale `audio` so its peak magnitude matches `target_dbfs` dBFS.

    Returns the input unchanged when it is silent. dBFS reference is 1.0 (full
    scale for float32 PCM).
    """
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak == 0.0:
        return audio.astype(np.float32, copy=False)  # type: ignore[no-any-return]
    target_linear = 10.0 ** (target_dbfs / 20.0)
    gain = target_linear / peak
    return (audio * gain).astype(np.float32, copy=False)  # type: ignore[no-any-return]


def normalize_rms(audio: np.ndarray, target_dbfs: float = -20.0) -> np.ndarray:
    """Scale `audio` so its RMS level matches `target_dbfs` dBFS.

    Not used by the clean pipeline yet — exposed for Phase 1b.1 when noise
    injection needs a consistent speech loudness before mixing.
    """
    if audio.size == 0:
        return audio.astype(np.float32, copy=False)  # type: ignore[no-any-return]
    rms = float(math.sqrt(np.mean(np.square(audio))))
    if rms == 0.0:
        return audio.astype(np.float32, copy=False)  # type: ignore[no-any-return]
    target_linear = 10.0 ** (target_dbfs / 20.0)
    gain = target_linear / rms
    return (audio * gain).astype(np.float32, copy=False)  # type: ignore[no-any-return]


def duration_seconds(audio: np.ndarray, sample_rate: int) -> float:
    """Return duration in seconds of a mono audio array."""
    return float(audio.shape[-1]) / float(sample_rate)
