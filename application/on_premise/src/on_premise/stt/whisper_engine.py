"""Whisper STT backends behind a common `Transcriber` Protocol.

Two implementations ship in parallel:

* `MLXWhisperTranscriber` — `mlx-whisper` on Apple Silicon Metal GPU. Primary
  on the dev box (~5-10x faster than faster-whisper CPU int8 on an M-series).
* `FasterWhisperTranscriber` — `faster-whisper` (ctranslate2) for CUDA and
  portable CPU. Fallback for Linux deploys + CI.

Both return a `RawTranscription` with word-level timings so
`stt.pipeline.transcribe_and_diarize()` can merge against pyannote segments.

The Protocol + factory are imported by `research_and_testing/scripts/
eval_stt_on_dataset.py` so WER benchmarks and on-premise deploys share one
code path — measuring what we ship.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RawWord:
    """Word-level timing from Whisper, backend-agnostic."""

    text: str
    start_s: float
    end_s: float
    probability: float | None = None


@dataclass(frozen=True)
class RawTranscription:
    """Whisper output, pre-diarization."""

    text: str                 # concatenated segment texts (legacy WER path)
    words: list[RawWord] = field(default_factory=list)
    duration_s: float = 0.0
    language: str = "pl"


class Transcriber(Protocol):
    """Minimal contract every STT backend satisfies."""

    name: str       # "mlx-whisper:large-v3" | "faster-whisper:large-v3"
    device: str     # "mps" | "cpu" | "cuda"

    def transcribe(self, wav_path: Path) -> RawTranscription: ...


# ---------------------------------------------------------------------------
# faster-whisper backend (portable)
# ---------------------------------------------------------------------------


class FasterWhisperTranscriber:
    """ctranslate2-based Whisper. int8 on CPU, int8_float16 on CUDA."""

    def __init__(self, model_size: str, device: str, compute_type: str) -> None:
        try:
            from faster_whisper import WhisperModel  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover — install-time guard
            raise RuntimeError(
                "faster-whisper is not installed. "
                "Run: uv pip install faster-whisper"
            ) from exc

        resolved = device if device != "auto" else ("cuda" if _cuda_available() else "cpu")
        log.info("loading faster-whisper %s on %s (%s)", model_size, resolved, compute_type)
        self._model = WhisperModel(model_size, device=resolved, compute_type=compute_type)
        self.name = f"faster-whisper:{model_size}"
        self.device = resolved
        self.compute_type = compute_type

    def transcribe(self, wav_path: Path) -> RawTranscription:
        # `condition_on_previous_text=False`: same anti-hallucination
        # rationale as the MLX path — whisper-large-v3 loops on very
        # clean TTS audio when the decoder's prev-segment prompt is
        # enabled. Keeping the flag symmetric across backends means
        # switching WHISPER_BACKEND at deploy time doesn't perturb
        # WER baselines measured on the other backend.
        segments, info = self._model.transcribe(
            str(wav_path),
            language="pl",
            vad_filter=True,
            beam_size=5,
            word_timestamps=True,
            condition_on_previous_text=False,
        )
        texts: list[str] = []
        words: list[RawWord] = []
        for seg in segments:
            texts.append(seg.text.strip())
            for w in seg.words or []:
                words.append(
                    RawWord(
                        text=w.word.strip(),
                        start_s=float(w.start),
                        end_s=float(w.end),
                        probability=float(w.probability) if w.probability is not None else None,
                    )
                )
        return RawTranscription(
            text=" ".join(t for t in texts if t),
            words=words,
            duration_s=float(info.duration),
            language=info.language or "pl",
        )


# ---------------------------------------------------------------------------
# MLX backend (Apple Silicon)
# ---------------------------------------------------------------------------


class MLXWhisperTranscriber:
    """mlx-whisper on Metal. First call downloads weights from HF."""

    def __init__(self, model_repo: str) -> None:
        try:
            import mlx_whisper  # type: ignore[import-not-found]  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "mlx-whisper is not installed. "
                "Run: uv pip install mlx-whisper  (Apple Silicon only)"
            ) from exc

        self._model_repo = model_repo
        self.name = f"mlx-whisper:{model_repo}"
        # Labeled "mps" so the metadata string matches torch convention.
        self.device = "mps"
        log.info("using mlx-whisper %s on Metal GPU", model_repo)

    def transcribe(self, wav_path: Path) -> RawTranscription:
        import mlx_whisper  # type: ignore[import-not-found]

        # `condition_on_previous_text=False` disables the decoder's
        # prev-segment prompt. With studio-quality TTS audio (our
        # synthetic clean dialogues) large-v3 otherwise falls into
        # repetition loops — emitting "Dziękuję za obejrzenie" or
        # repeating a Polish phrase for minutes — which torched the
        # `neuro-ct` clean run at 72.7% WER. Noise breaks the loop
        # by perturbing the acoustic embedding, which is why the
        # same scenario's noisy WER sat at 6.7%. Turning the prompt
        # off removes the sampling bias without changing model
        # weights or per-segment quality.
        result = mlx_whisper.transcribe(
            str(wav_path),
            path_or_hf_repo=self._model_repo,
            language="pl",
            word_timestamps=True,
            condition_on_previous_text=False,
        )
        text = str(result.get("text") or "").strip()
        language = str(result.get("language") or "pl")
        words: list[RawWord] = []
        duration = 0.0
        for seg in result.get("segments") or []:
            duration = max(duration, float(seg.get("end") or 0.0))
            for w in seg.get("words") or []:
                words.append(
                    RawWord(
                        text=str(w.get("word") or "").strip(),
                        start_s=float(w.get("start") or 0.0),
                        end_s=float(w.get("end") or 0.0),
                        probability=(
                            float(w["probability"])
                            if w.get("probability") is not None
                            else None
                        ),
                    )
                )
        # If segments were absent but text present, we still return something sane.
        return RawTranscription(
            text=text,
            words=words,
            duration_s=duration,
            language=language,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_transcriber(
    config: Mapping[str, object],
    env: Mapping[str, str] | None = None,
) -> Transcriber:
    """Construct a Transcriber from config dict + env.

    `config` is the parsed `configs/whisper.yaml` with env-var templates
    already expanded by `load_whisper_config`. `env` defaults to `os.environ`
    and lets tests pass a sandboxed mapping.
    """
    env = env if env is not None else os.environ
    backend = env.get("WHISPER_BACKEND", "mlx").strip().lower()

    if backend == "mlx":
        repo = env.get("WHISPER_MLX_REPO", "mlx-community/whisper-large-v3-mlx")
        return MLXWhisperTranscriber(model_repo=repo)
    if backend == "faster-whisper":
        model_node = config.get("model") if isinstance(config, Mapping) else None
        # YAML file uses `model.size`; env overrides if set.
        size_from_config: object = (
            model_node.get("size") if isinstance(model_node, Mapping) else None
        )
        size = env.get("WHISPER_MODEL_SIZE") or size_from_config or "large-v3"
        device = env.get("WHISPER_DEVICE", "auto")
        compute_type = env.get("WHISPER_COMPUTE_TYPE", "int8")
        return FasterWhisperTranscriber(
            model_size=str(size),
            device=str(device),
            compute_type=str(compute_type),
        )
    raise ValueError(
        f"unknown WHISPER_BACKEND: {backend!r} (expected: mlx | faster-whisper)"
    )


def load_whisper_config(config_path: Path) -> dict[str, object]:
    """Load `configs/whisper.yaml` and expand `${VAR}` templates against env.

    The existing `whisper.yaml` uses `${WHISPER_MODEL_SIZE}`-style templating;
    `yaml.safe_load` returns them as literal strings. This helper expands them
    so `build_transcriber(config, env)` can treat its config as already-resolved.
    """
    import yaml  # local import — keeps module lightweight for tests

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    expanded = _expand_env_templates(raw)
    if not isinstance(expanded, dict):
        raise ValueError(f"whisper config must be a mapping, got {type(expanded)}")
    return expanded


def _expand_env_templates(obj: object) -> object:
    """Recursively replace `${VAR}` strings with `os.environ[VAR]` (or `VAR` if unset)."""
    if isinstance(obj, dict):
        return {k: _expand_env_templates(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_templates(v) for v in obj]
    if isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        var = obj[2:-1]
        return os.environ.get(var, obj)
    return obj


def _cuda_available() -> bool:
    try:
        import torch  # type: ignore[import-not-found]

        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover
        return False
