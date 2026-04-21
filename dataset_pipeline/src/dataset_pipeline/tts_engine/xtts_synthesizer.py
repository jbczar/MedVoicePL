"""XTTSv2 wrapper for Polish speech synthesis.

Uses the community `coqui-tts` fork (original Coqui TTS company shut down in
early 2024). The model is ~2 GB and is cached under `TTS_MODEL_CACHE` (outside
the repo) so it is downloaded once per machine.

Device selection follows `TTS_DEVICE`:
  - `auto` (default): MPS on Apple Silicon when available, otherwise CPU.
  - `mps`, `cuda`, `cpu`: explicit override.

Model load is **lazy** — the first call to `synthesize()` instantiates the
underlying `TTS` object. This keeps unit tests free of the 2 GB download and
lets callers instantiate the class for validation/configuration without paying
the model cost upfront.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:
    from dataset_pipeline.tts_engine.voice_manager import VoiceProfile


log = logging.getLogger(__name__)

XTTS_V2_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
DEFAULT_SAMPLE_RATE = 22050


class Synthesizer(Protocol):
    """Minimal interface `pipeline.run_audio` depends on.

    Keeps tests able to swap in a `_FakeSynthesizer` without importing torch.
    """

    @property
    def sample_rate(self) -> int: ...

    @property
    def model_name(self) -> str: ...

    @property
    def device(self) -> str: ...

    def synthesize(self, text: str, profile: VoiceProfile) -> np.ndarray: ...


def _clean_text_for_tts(text: str) -> str:
    """Light Polish-safe cleanup before synthesis.

    XTTSv2 prosody is noticeably worse on text without sentence-ending
    punctuation and on runs of whitespace / markdown artifacts that sometimes
    leak through from LLM output.
    """
    # Strip light markdown (*, _, `, backticks) but keep letters/digits/diacritics.
    cleaned = re.sub(r"[*_`]+", "", text)
    # Collapse any whitespace run (incl. newlines) to a single space.
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return cleaned
    if cleaned[-1] not in ".?!":
        cleaned += "."
    return cleaned


def _resolve_device(explicit: str | None) -> str:
    """Pick a torch device string given `TTS_DEVICE` / explicit arg."""
    import torch

    if explicit is None or explicit == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if explicit not in {"cpu", "mps", "cuda"}:
        raise ValueError(
            f"TTS_DEVICE={explicit!r} not supported; use auto|cpu|mps|cuda"
        )
    return explicit


class XTTSSynthesizer:
    """Lazy-loaded XTTSv2 synthesizer (coqui-tts fork)."""

    def __init__(
        self,
        *,
        device: str | None = None,
        model_cache: Path | None = None,
        model_name: str = XTTS_V2_MODEL_NAME,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ) -> None:
        self._explicit_device = device or os.environ.get("TTS_DEVICE", "auto")
        self._model_cache = model_cache or Path(
            os.path.expanduser(
                os.environ.get("TTS_MODEL_CACHE", "~/.cache/medvoicepl/xtts")
            )
        )
        self._model_name = model_name
        self._sample_rate = sample_rate
        self._tts: Any | None = None
        self._resolved_device: str | None = None

    # -- Public interface -------------------------------------------------

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def device(self) -> str:
        if self._resolved_device is None:
            self._resolved_device = _resolve_device(self._explicit_device)
        return self._resolved_device

    def synthesize(self, text: str, profile: VoiceProfile) -> np.ndarray:
        """Synthesize a single turn of Polish speech.

        Returns a mono float32 array at `self.sample_rate`.
        """
        cleaned = _clean_text_for_tts(text)
        if not cleaned:
            raise ValueError("cannot synthesize empty text")
        if not profile.reference_audio.exists():
            raise FileNotFoundError(
                f"reference audio missing: {profile.reference_audio}; "
                f"run dataset_pipeline/scripts/fetch_reference_voices.py"
            )

        tts = self._ensure_loaded()
        log.debug(
            "xtts synthesize: profile=%s device=%s text=%r",
            profile.name,
            self.device,
            cleaned[:60],
        )
        audio = tts.tts(
            text=cleaned,
            speaker_wav=str(profile.reference_audio),
            language=profile.language,
            speed=profile.speed,
        )
        return np.asarray(audio, dtype=np.float32)

    # -- Internals --------------------------------------------------------

    def _ensure_loaded(self) -> Any:
        if self._tts is not None:
            return self._tts

        if os.environ.get("TTS_ACCEPT_COQUI_LICENSE") != "1":
            # coqui-tts asks interactively otherwise; opt in explicitly.
            os.environ["COQUI_TOS_AGREED"] = "1"
        else:
            os.environ["COQUI_TOS_AGREED"] = "1"

        self._model_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TTS_HOME", str(self._model_cache))

        from TTS.api import TTS  # heavy: torch + numpy + transformers

        log.info(
            "loading XTTSv2 model=%s device=%s cache=%s",
            self._model_name,
            self.device,
            self._model_cache,
        )
        tts = TTS(self._model_name, progress_bar=False).to(self.device)
        self._tts = tts
        return tts
