"""Local SLM backends behind a common `StructuredExtractor` Protocol.

Two implementations ship in parallel (mirroring the STT dual-backend
posture in `on_premise.stt.whisper_engine`):

* `MLXLMExtractor` — `mlx-lm` on Apple Silicon Metal GPU. Primary on the
  dev box. Loads the first-party SpeakLeash MLX 4-bit quant of
  Bielik-11B-v3.0-Instruct from HF once per process.
* `LlamaCppExtractor` — `llama-cpp-python` with a GGUF Bielik. CUDA on
  Linux; CPU-only on any platform. Installed via the `[portable]` extra
  because it compiles from source (~3-5 min on an M5, needs
  `CMAKE_ARGS="-DLLAMA_METAL=on"` for Metal offload, and most dev
  happens on MLX).

Both return raw JSON strings — the pipeline parses with Pydantic and
handles validation retry. When `structured_mode="grammar"`, a
`lm-format-enforcer` logits processor clamps decoding to the JSON
schema, so the string that comes back is syntactically valid JSON with
exactly the expected keys. `"raw"` mode disables the logits processor
(useful for A/B'ing whether grammar is pulling its weight) and relies on
the retry loop alone.

The Protocol + factory are imported by `on_premise.structuring.pipeline`
and by `research_and_testing/scripts/eval_slm_on_transcripts.py` — same
code path for benchmarks and deploys.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error types (mirror Phase 1a's llm_client.py shape)
# ---------------------------------------------------------------------------


class TransientSLMError(RuntimeError):
    """Raised for retriable failures — OOM, tokenizer reload flake."""


class PermanentSLMError(RuntimeError):
    """Raised for failures retry won't fix — model not found, chat
    template missing, grammar compile fail, unsupported backend.
    """


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class StructuredExtractor(Protocol):
    """Minimal contract every local-SLM backend satisfies.

    `extract()` returns a raw JSON string. The pipeline owns
    Pydantic validation and correction-hint retry — backends only
    need to constrain decoding (when grammar mode is on) and run
    the generation loop.
    """

    name: str               # "mlx-lm:Bielik-11B-v3.0-Instruct-MLX-4bit" | "llama-cpp:Bielik-Q4_K_M"
    device: str             # "mps" | "cpu" | "cuda"
    structured_mode: str    # "grammar" | "raw"

    def extract(self, prompt: str, *, schema: Mapping[str, Any]) -> str: ...


# ---------------------------------------------------------------------------
# MLX-LM backend (Apple Silicon)
# ---------------------------------------------------------------------------


class MLXLMExtractor:
    """`mlx-lm` on Metal. First call downloads weights from HF.

    Grammar mode uses `lm-format-enforcer`'s MLX logits processor
    which clamps token sampling to whatever the current JSON-schema
    parser will accept. Raw mode runs with the default sampler and
    relies on the pipeline's validation retry.
    """

    def __init__(
        self,
        model_repo: str,
        *,
        max_new_tokens: int = 2048,
        temperature: float = 0.1,
        structured_mode: str = "grammar",
    ) -> None:
        if structured_mode not in {"grammar", "raw"}:
            raise PermanentSLMError(
                f"structured_mode must be grammar|raw, got {structured_mode!r}"
            )
        try:
            import mlx_lm  # type: ignore[import-not-found]  # noqa: F401
        except ImportError as exc:  # pragma: no cover — install-time guard
            raise PermanentSLMError(
                "mlx-lm is not installed. "
                "Run: uv pip install mlx-lm  (Apple Silicon only)"
            ) from exc

        self._model_repo = model_repo
        self._max_new_tokens = int(max_new_tokens)
        self._temperature = float(temperature)
        self.structured_mode = structured_mode
        self.name = f"mlx-lm:{model_repo}"
        self.device = "mps"
        self._model: Any = None
        self._tokenizer: Any = None
        log.info("using mlx-lm %s on Metal GPU (mode=%s)", model_repo, structured_mode)

    def _load(self) -> None:
        """Lazy load — heavy, runs once per process."""
        if self._model is not None:
            return
        from mlx_lm import load  # type: ignore[import-not-found]

        try:
            # mlx-lm's `load()` returns `(model, tokenizer)` in 0.19-0.20 and
            # `(model, tokenizer, config)` in >=0.21. Accept either shape.
            loaded = load(self._model_repo)
            self._model, self._tokenizer = loaded[0], loaded[1]
        except Exception as exc:
            raise PermanentSLMError(
                f"failed to load mlx-lm model {self._model_repo!r}: {exc}"
            ) from exc

        # Chat-template check: Bielik's community MLX repacks sometimes
        # land without a tokenizer_config.chat_template, which silently
        # produces gibberish because mlx-lm's `generate` falls back to
        # raw concat. Fail fast with an actionable pointer.
        if getattr(self._tokenizer, "chat_template", None) is None:
            raise PermanentSLMError(
                f"tokenizer for {self._model_repo!r} has no chat_template. "
                "Pick a different MLX repo (search HF for 'Bielik MLX 4bit') "
                "or switch to llama-cpp with a GGUF that embeds the template."
            )

    def extract(self, prompt: str, *, schema: Mapping[str, Any]) -> str:
        self._load()
        from mlx_lm import generate  # type: ignore[import-not-found]

        messages = [{"role": "user", "content": prompt}]
        try:
            templated = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as exc:
            raise PermanentSLMError(f"chat-template render failed: {exc}") from exc

        logits_processors = None
        if self.structured_mode == "grammar":
            logits_processors = _build_mlx_logits_processors(self._tokenizer, schema)

        try:
            out = generate(
                self._model,
                self._tokenizer,
                prompt=templated,
                max_tokens=self._max_new_tokens,
                # mlx-lm's `generate` handles temperature/logits via
                # sampler kwargs in recent versions; older versions expose
                # `temp=` instead. Try new-style first.
                **_mlx_sampler_kwargs(
                    temperature=self._temperature,
                    logits_processors=logits_processors,
                ),
            )
        except RuntimeError as exc:
            # Metal OOM / tokenizer flake — transient, retry at pipeline level.
            raise TransientSLMError(str(exc)) from exc
        if not isinstance(out, str):
            # Some mlx-lm versions return (str, metadata); normalize.
            try:
                out = out[0] if isinstance(out, tuple) else str(out)
            except Exception as exc:
                raise PermanentSLMError(f"unexpected mlx-lm return type: {type(out)}") from exc
        return out


def _mlx_sampler_kwargs(
    *, temperature: float, logits_processors: list[Any] | None
) -> dict[str, Any]:
    """Bridge across mlx-lm API versions (temp= vs sampler=)."""
    kwargs: dict[str, Any] = {"verbose": False}
    try:
        from mlx_lm.sample_utils import make_sampler
        kwargs["sampler"] = make_sampler(temp=temperature)
    except ImportError:
        # fallback for very old mlx-lm versions
        kwargs["temp"] = temperature

    if logits_processors:
        kwargs["logits_processors"] = logits_processors
    return kwargs


def _build_mlx_logits_processors(tokenizer: Any, schema: Mapping[str, Any]) -> list[Any]:
    """Compile the JSON schema into an mlx-lm logits processor via lm-format-enforcer."""
    try:
        from lmformatenforcer import JsonSchemaParser  # type: ignore[import-not-found]

        from on_premise.structuring.mlx_logits_processor import (
            build_mlx_logits_processor,
        )
    except ImportError as exc:  # pragma: no cover
        raise PermanentSLMError(
            "lm-format-enforcer is not installed. Run: uv pip install lm-format-enforcer"
        ) from exc

    try:
        parser = JsonSchemaParser(dict(schema))
        processor = build_mlx_logits_processor(tokenizer, parser)
    except Exception as exc:
        raise PermanentSLMError(f"failed to compile JSON schema into MLX grammar: {exc}") from exc
    return [processor]


# ---------------------------------------------------------------------------
# llama.cpp backend (portable — CUDA / CPU / Metal via GGUF)
# ---------------------------------------------------------------------------


class LlamaCppExtractor:
    """`llama-cpp-python` with a GGUF Bielik. Portable fallback."""

    def __init__(
        self,
        gguf_path: Path,
        *,
        device: str = "auto",
        n_ctx: int = 4096,
        max_new_tokens: int = 2048,
        temperature: float = 0.1,
        structured_mode: str = "grammar",
    ) -> None:
        if structured_mode not in {"grammar", "raw"}:
            raise PermanentSLMError(
                f"structured_mode must be grammar|raw, got {structured_mode!r}"
            )
        if not gguf_path or not Path(gguf_path).exists():
            raise PermanentSLMError(
                f"SLM_GGUF_PATH is empty or does not exist: {gguf_path!r}"
            )
        try:
            from llama_cpp import Llama  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover — install-time guard
            raise PermanentSLMError(
                "llama-cpp-python is not installed. "
                "Run: uv sync --extra portable  "
                "(compiles from source; on Apple Silicon set "
                'CMAKE_ARGS="-DLLAMA_METAL=on" for GPU offload)'
            ) from exc

        resolved = device if device != "auto" else ("cuda" if _cuda_available() else "cpu")
        n_gpu_layers = -1 if resolved == "cuda" else 0
        try:
            self._llm = Llama(
                model_path=str(gguf_path),
                n_ctx=int(n_ctx),
                n_gpu_layers=n_gpu_layers,
                verbose=False,
            )
        except Exception as exc:
            raise PermanentSLMError(
                f"failed to load GGUF {gguf_path}: {exc}"
            ) from exc

        self._max_new_tokens = int(max_new_tokens)
        self._temperature = float(temperature)
        self.structured_mode = structured_mode
        self.name = f"llama-cpp:{Path(gguf_path).name}"
        self.device = resolved
        log.info("using llama-cpp %s on %s (mode=%s)", gguf_path, resolved, structured_mode)

    def extract(self, prompt: str, *, schema: Mapping[str, Any]) -> str:
        kwargs: dict[str, Any] = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self._max_new_tokens,
            "temperature": self._temperature,
        }
        if self.structured_mode == "grammar":
            kwargs["logits_processor"] = _build_llama_cpp_logits_processor(
                self._llm, schema
            )

        try:
            response = self._llm.create_chat_completion(**kwargs)
        except RuntimeError as exc:
            raise TransientSLMError(str(exc)) from exc

        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise PermanentSLMError(
                f"unexpected llama-cpp response shape: {response!r}"
            ) from exc
        return str(content)


def _build_llama_cpp_logits_processor(llm: Any, schema: Mapping[str, Any]) -> Any:
    """Compile JSON schema into a llama-cpp logits processor via lm-format-enforcer."""
    try:
        from lmformatenforcer import JsonSchemaParser  # type: ignore[import-not-found]
        from lmformatenforcer.integrations.llamacpp import (  # type: ignore[import-not-found]
            build_llamacpp_logits_processor,
        )
    except ImportError as exc:  # pragma: no cover
        raise PermanentSLMError(
            "lm-format-enforcer is not installed. Run: uv pip install lm-format-enforcer"
        ) from exc
    try:
        parser = JsonSchemaParser(dict(schema))
        return build_llamacpp_logits_processor(llm, parser)
    except Exception as exc:
        raise PermanentSLMError(
            f"failed to compile JSON schema into llama-cpp grammar: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Factory + config loader
# ---------------------------------------------------------------------------


def build_extractor(
    config: Mapping[str, Any],
    env: Mapping[str, str] | None = None,
) -> StructuredExtractor:
    """Construct a StructuredExtractor from config dict + env.

    `config` is the parsed `configs/slm.yaml` with env-var templates
    already expanded by `load_slm_config`. `env` defaults to `os.environ`
    and lets tests pass a sandboxed mapping.
    """
    env = env if env is not None else os.environ
    backend = (env.get("SLM_BACKEND") or str(config.get("backend") or "mlx-lm")).strip().lower()

    model_cfg = config.get("model", {}) if isinstance(config, Mapping) else {}
    if not isinstance(model_cfg, Mapping):
        model_cfg = {}
    max_new = int(
        env.get("SLM_MAX_NEW_TOKENS")
        or model_cfg.get("max_new_tokens")
        or 2048
    )
    temperature = float(
        env.get("SLM_TEMPERATURE")
        or model_cfg.get("temperature")
        or 0.1
    )
    structured_mode = (
        env.get("SLM_STRUCTURED_MODE")
        or str(config.get("structured_mode") or "grammar")
    ).strip().lower()

    if backend == "mlx-lm":
        repo = env.get("SLM_MLX_REPO") or str(
            model_cfg.get("mlx_repo") or "speakleash/Bielik-11B-v3.0-Instruct-MLX-4bit"
        )
        return MLXLMExtractor(
            model_repo=repo,
            max_new_tokens=max_new,
            temperature=temperature,
            structured_mode=structured_mode,
        )
    if backend == "llama-cpp":
        gguf = env.get("SLM_GGUF_PATH") or str(model_cfg.get("gguf_path") or "")
        device = env.get("SLM_DEVICE") or str(model_cfg.get("device") or "auto")
        return LlamaCppExtractor(
            gguf_path=Path(gguf),
            device=device,
            max_new_tokens=max_new,
            temperature=temperature,
            structured_mode=structured_mode,
        )
    raise PermanentSLMError(
        f"unknown SLM_BACKEND: {backend!r} (expected: mlx-lm | llama-cpp)"
    )


def load_slm_config(config_path: Path) -> dict[str, Any]:
    """Load `configs/slm.yaml` and expand `${VAR}` templates against env.

    Mirrors `on_premise.stt.whisper_engine.load_whisper_config`. The
    10-line `_expand_env_templates` helper is duplicated on purpose —
    shared-module extraction is deferred until we have a third caller.
    """
    import yaml  # local import — keeps module lightweight for tests

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    expanded = _expand_env_templates(raw)
    if not isinstance(expanded, dict):
        raise ValueError(f"slm config must be a mapping, got {type(expanded)}")
    return expanded


def _expand_env_templates(obj: Any) -> Any:
    """Recursively replace `${VAR}` strings with `os.environ[VAR]`.

    Unlike Phase 2a's identical helper, unset templates resolve to
    `None` (not the literal `${VAR}` string) so that `build_extractor`'s
    hardcoded fallbacks take over. This means a user with an older
    `.env` from Phase 2a that doesn't yet have `SLM_BACKEND=mlx-lm`
    still gets a working default instead of a `"unknown SLM_BACKEND:
    '${slm_backend}'"` crash.
    """
    if isinstance(obj, dict):
        return {k: _expand_env_templates(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_templates(v) for v in obj]
    if isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        var = obj[2:-1]
        return os.environ.get(var) or None
    return obj


def _cuda_available() -> bool:
    try:
        import torch  # type: ignore[import-not-found]

        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover
        return False
