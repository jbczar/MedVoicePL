"""Background noise injection (clinic ambient, corridor, medical equipment).

Deferred to **Phase 1b.1**. The Phase 1b pipeline ships clean audio so TTS
quality can be validated independently of noise parameters. `inject()` is a
no-op pass-through so downstream code can be wired up as if noise were
already in place — swapping in the real implementation later is additive.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def inject(audio: np.ndarray, sample_rate: int, config: Any | None = None) -> np.ndarray:
    """Return `audio` unchanged. Real impl lands in Phase 1b.1.

    The signature is frozen so callers (e.g., `pipeline.run_audio`) can add
    the call site now and swap in the noise implementation without code
    changes on the caller side.
    """
    del sample_rate, config  # unused until Phase 1b.1
    return audio.astype(np.float32, copy=False)
