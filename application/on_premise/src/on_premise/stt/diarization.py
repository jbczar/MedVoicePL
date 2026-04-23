"""Speaker diarization for medical visits (2 speakers: doctor + patient).

Primary backend: `pyannote.audio 3.1` (model `pyannote/speaker-diarization-3.1`).
Gated on Hugging Face: needs `HUGGINGFACE_TOKEN` env + one-time license accept
on the model's HF page. `scripts/check_stt_models.py` diagnoses this cleanly.

Returns *opaque* speaker labels (`SPEAKER_00`, `SPEAKER_01`). Mapping to
`{doctor, patient}` is a downstream concern handled by `pipeline.py` so the
diarization module stays a pure speech-activity module with no domain
knowledge baked in.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RawDiarizationSegment:
    """One contiguous stretch of speech attributed to one opaque speaker."""

    start_s: float
    end_s: float
    speaker_label: str     # e.g. "SPEAKER_00", "SPEAKER_01"
    confidence: float | None = None


class Diarizer(Protocol):
    """Minimal contract every diarization backend satisfies."""

    name: str           # e.g. "pyannote/speaker-diarization-3.1"
    device: str         # "cpu" | "cuda" | "mps"

    def diarize(self, wav_path: Path) -> list[RawDiarizationSegment]: ...


# ---------------------------------------------------------------------------
# pyannote backend
# ---------------------------------------------------------------------------


class PyannoteDiarizer:
    """pyannote.audio 3.1 pipeline, pinned to 2 speakers."""

    DEFAULT_MODEL = "pyannote/speaker-diarization-3.1"

    def __init__(
        self,
        *,
        hf_token: str,
        model_id: str = DEFAULT_MODEL,
        device: str = "auto",
        num_speakers: int = 2,
    ) -> None:
        # Check HF token BEFORE importing pyannote so the error message the
        # user actually needs to act on surfaces first — and so our unit
        # tests can assert the missing-token path without pulling in
        # pyannote.audio (which carries the torch/torchaudio heavy deps).
        if not hf_token:
            raise RuntimeError(
                "HUGGINGFACE_TOKEN is required for pyannote/speaker-diarization-3.1. "
                "Accept the license on ALL THREE gated repos:\n"
                "  - https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                "  - https://huggingface.co/pyannote/segmentation-3.0\n"
                "  - https://huggingface.co/pyannote/speaker-diarization-community-1\n"
                "then export HUGGINGFACE_TOKEN=<your-read-token>."
            )

        try:
            from pyannote.audio import Pipeline  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "pyannote.audio is not installed. "
                "Run: uv pip install pyannote.audio"
            ) from exc

        resolved_device = _resolve_device(device)
        log.info("loading %s on %s", model_id, resolved_device)
        # pyannote.audio 3.x renamed `use_auth_token` -> `token` (alignment
        # with huggingface_hub). Try the new name first, fall back so we
        # stay compatible with both 2.x and 3.x installs.
        try:
            pipeline = Pipeline.from_pretrained(model_id, token=hf_token)
        except TypeError:  # pragma: no cover — pyannote 2.x compat path
            pipeline = Pipeline.from_pretrained(  # type: ignore[call-arg]
                model_id, use_auth_token=hf_token
            )
        if pipeline is None:
            raise RuntimeError(
                f"pyannote returned a None pipeline for {model_id!r}. "
                "Most likely: license not accepted or HF token lacks read scope."
            )
        # `pipeline.to(torch.device(...))` for GPU/MPS.
        try:
            import torch  # type: ignore[import-not-found]

            pipeline = pipeline.to(torch.device(resolved_device))
        except Exception as exc:  # pragma: no cover — fall back to CPU silently
            log.warning("pyannote .to(%s) failed (%s); staying on CPU", resolved_device, exc)
            resolved_device = "cpu"

        self._pipeline = pipeline
        self._num_speakers = num_speakers
        self.name = model_id
        self.device = resolved_device

    def diarize(self, wav_path: Path) -> list[RawDiarizationSegment]:
        result = self._pipeline(
            str(wav_path),
            num_speakers=self._num_speakers,
        )
        # pyannote 2.x returned an `Annotation` directly; 3.x wraps it in a
        # `DiarizeOutput` dataclass with the Annotation on `.speaker_diarization`
        # (exposed by the wrapper so downstream can also reach embeddings).
        # Duck-type across both versions without a hard version pin.
        annotation = result if hasattr(result, "itertracks") else getattr(
            result, "speaker_diarization", None
        )
        if annotation is None or not hasattr(annotation, "itertracks"):
            raise RuntimeError(
                f"pyannote pipeline returned unexpected shape: {type(result).__name__}. "
                f"Attributes: {dir(result)!r}"
            )
        segments: list[RawDiarizationSegment] = []
        # `annotation.itertracks(yield_label=True)` -> (Segment, track_id, label)
        for segment, _track, label in annotation.itertracks(yield_label=True):
            segments.append(
                RawDiarizationSegment(
                    start_s=float(segment.start),
                    end_s=float(segment.end),
                    speaker_label=str(label),
                    confidence=None,  # pyannote 3.1 doesn't expose per-seg score cleanly
                )
            )
        segments.sort(key=lambda s: s.start_s)
        return segments


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_diarizer(
    config: Mapping[str, object] | None = None,
    env: Mapping[str, str] | None = None,
) -> Diarizer:
    """Construct a Diarizer from env (no config file needed in 2a)."""
    env = env if env is not None else os.environ
    token = env.get("HUGGINGFACE_TOKEN", "").strip()
    device = env.get("PYANNOTE_DEVICE", "auto")
    model_id = str(
        (config or {}).get("model_id") or PyannoteDiarizer.DEFAULT_MODEL
    ) if isinstance(config, Mapping) else PyannoteDiarizer.DEFAULT_MODEL
    return PyannoteDiarizer(
        hf_token=token,
        model_id=model_id,
        device=device,
        num_speakers=2,
    )


def _resolve_device(device: str) -> str:
    """`auto` -> best available torch device; pass-through otherwise."""
    if device != "auto":
        return device
    try:
        import torch  # type: ignore[import-not-found]

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:  # pragma: no cover
        pass
    return "cpu"
