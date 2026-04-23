"""Thin silence detector used only for the `--trim-silence` CLI flag.

Whisper and pyannote both do their own VAD internally; this module does not
gate their input. It exists so the CLI can chop leading/trailing silence from
long recordings before passing them through the pipeline, shaving a bit of
diarization cost. RMS-threshold, 15 lines of numpy, no model weights.
"""

from __future__ import annotations

import numpy as np


def detect_silence_segments(
    audio: np.ndarray,
    sample_rate: int,
    *,
    min_silence_ms: int = 500,
    rms_threshold: float = 0.005,
    frame_ms: int = 20,
) -> list[tuple[float, float]]:
    """Return `(start_s, end_s)` ranges where the signal is below `rms_threshold`.

    A silence must last at least `min_silence_ms` to be reported — prevents
    chopping natural pauses between words. `audio` is mono float32 in [-1, 1].
    """
    if audio.ndim != 1:
        raise ValueError(f"expected mono audio, got shape {audio.shape}")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")

    frame_len = max(1, int(sample_rate * frame_ms / 1000))
    min_silence_frames = max(1, int(min_silence_ms / frame_ms))

    # Pad so the last partial frame still gets scored.
    n_frames = (len(audio) + frame_len - 1) // frame_len
    padded = np.pad(audio, (0, n_frames * frame_len - len(audio)))
    frames = padded.reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    is_silent = rms < rms_threshold

    # Group consecutive silent frames into runs.
    out: list[tuple[float, float]] = []
    run_start: int | None = None
    for i, quiet in enumerate(is_silent):
        if quiet and run_start is None:
            run_start = i
        elif not quiet and run_start is not None:
            if i - run_start >= min_silence_frames:
                out.append((run_start * frame_ms / 1000, i * frame_ms / 1000))
            run_start = None
    if run_start is not None and len(is_silent) - run_start >= min_silence_frames:
        out.append((run_start * frame_ms / 1000, len(is_silent) * frame_ms / 1000))
    return out
