"""Preflight: verify local SLM prerequisites on a fresh clone.

Checks:
1. `mlx-lm` importable (Apple Silicon primary path).
2. `llama-cpp-python` importable (portable fallback — warn-only).
3. `lm-format-enforcer` importable (grammar-constrained decoding).
4. Configured MLX repo reachable AND its tokenizer has a `chat_template`.
5. Context window >= 3000 (Bielik native 4K — any less and few-shot
   won't fit alongside the transcript).
6. Smoke generation: 1 token, to confirm weights load and inference runs.

Prints a red/green table and exits non-zero on missing hard requirements.

Usage:
    uv run python application/on_premise/scripts/check_slm_models.py
    uv run python application/on_premise/scripts/check_slm_models.py --skip-load
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

app = typer.Typer(add_completion=False, help="SLM prerequisite checker")
console = Console()


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    fix: str | None = None
    warn_only: bool = False


@app.command()
def check(
    skip_load: bool = typer.Option(
        False,
        "--skip-load",
        help="Skip the actual mlx-lm model load (fast mode — only checks imports + config).",
    ),
) -> None:
    load_dotenv()
    results: list[CheckResult] = []
    results.append(_check_mlx_lm())
    results.append(_check_llama_cpp())
    results.append(_check_lmfe())

    if not skip_load:
        results.extend(_check_mlx_model_and_generation())
    else:
        results.append(
            CheckResult(
                "mlx-lm model load",
                True,
                "skipped (--skip-load)",
                None,
                warn_only=False,
            )
        )

    table = Table(title="SLM prerequisites")
    table.add_column("check")
    table.add_column("status")
    table.add_column("detail")
    for r in results:
        status = (
            "[green]OK[/green]"
            if r.ok
            else ("[yellow]WARN[/yellow]" if r.warn_only else "[red]MISSING[/red]")
        )
        table.add_row(r.name, status, r.detail)
    console.print(table)

    failed = [r for r in results if not r.ok and not r.warn_only]
    if failed:
        console.rule("[bold red]Action required")
        for r in failed:
            if r.fix:
                console.print(f"• [yellow]{r.name}[/yellow]: {r.fix}")
        sys.exit(1)
    console.rule("[bold green]All hard prerequisites satisfied")


def _check_mlx_lm() -> CheckResult:
    try:
        import mlx_lm  # type: ignore[import-not-found]  # noqa: F401
    except ImportError as exc:
        return CheckResult(
            "mlx-lm",
            False,
            str(exc),
            "uv pip install mlx-lm  (Apple Silicon only; Linux/CI can switch SLM_BACKEND=llama-cpp)",
        )
    return CheckResult("mlx-lm", True, "importable", None)


def _check_llama_cpp() -> CheckResult:
    try:
        import llama_cpp  # type: ignore[import-not-found]  # noqa: F401
    except ImportError as exc:
        return CheckResult(
            "llama-cpp-python",
            False,
            str(exc),
            "uv sync --extra portable  (compiles from source ~3-5 min; on Apple Silicon "
            'set CMAKE_ARGS="-DLLAMA_METAL=on" for GPU offload)',
            warn_only=True,  # optional extra
        )
    return CheckResult("llama-cpp-python", True, "importable (optional)", None)


def _check_lmfe() -> CheckResult:
    try:
        import lmformatenforcer  # type: ignore[import-not-found]  # noqa: F401
    except ImportError as exc:
        return CheckResult(
            "lm-format-enforcer",
            False,
            str(exc),
            "uv pip install lm-format-enforcer  (required for SLM_STRUCTURED_MODE=grammar)",
        )
    return CheckResult("lm-format-enforcer", True, "importable", None)


def _check_mlx_model_and_generation() -> list[CheckResult]:
    """Load the configured MLX repo and run a 1-token generation."""
    out: list[CheckResult] = []
    repo = os.environ.get("SLM_MLX_REPO") or "speakleash/Bielik-11B-v3.0-Instruct-MLX-4bit"
    try:
        from mlx_lm import generate, load  # type: ignore[import-not-found]
    except ImportError as exc:
        out.append(CheckResult("mlx-lm model load", False, str(exc), "install mlx-lm first"))
        return out

    try:
        # mlx-lm's load() returns (model, tokenizer) in 0.19-0.20 and
        # (model, tokenizer, config) in >=0.21. Accept either shape.
        loaded = load(repo)
        model, tokenizer = loaded[0], loaded[1]
    except Exception as exc:
        out.append(
            CheckResult(
                "mlx-lm model load",
                False,
                f"{type(exc).__name__}: {exc}",
                f"Check repo {repo!r} exists on HF and is MLX-compatible. "
                "Try: huggingface-cli login; or pick a different mlx-community repo.",
            )
        )
        return out

    # Chat template present?
    template = getattr(tokenizer, "chat_template", None)
    if template is None:
        out.append(
            CheckResult(
                "tokenizer chat_template",
                False,
                "missing on tokenizer",
                f"Repo {repo!r} lacks tokenizer_config.json.chat_template. "
                "Pick a different mlx-community Bielik repo or switch to llama-cpp GGUF.",
            )
        )
    else:
        out.append(CheckResult("tokenizer chat_template", True, "present", None))

    # Context window sanity
    ctx = _probe_context_length(tokenizer, model)
    if ctx is not None:
        if ctx < 3000:
            out.append(
                CheckResult(
                    f"context length ({ctx})",
                    False,
                    f"{ctx} tokens — too tight for system+few-shot+transcript",
                    "Use the RoPE-scaled 8K repo or trim the few-shot example.",
                    warn_only=True,
                )
            )
        else:
            out.append(CheckResult("context length", True, f"{ctx} tokens", None))
    else:
        out.append(
            CheckResult("context length", True, "unknown (could not probe)", None, warn_only=True)
        )

    # Smoke generation — 1 token, confirm weights + sampler wired
    try:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Hej"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        _ = generate(model, tokenizer, prompt=prompt, max_tokens=1, verbose=False)
        out.append(CheckResult("smoke generation", True, "1-token ok", None))
    except Exception as exc:
        out.append(
            CheckResult(
                "smoke generation",
                False,
                f"{type(exc).__name__}: {exc}",
                "Model loaded but generation failed — check mlx-lm version compatibility.",
            )
        )

    return out


def _probe_context_length(tokenizer: object, model: object) -> int | None:
    # Tokenizer exposes it on many HF configs.
    for attr in ("model_max_length", "max_position_embeddings"):
        val = getattr(tokenizer, attr, None)
        if isinstance(val, int) and 0 < val < 1_000_000:
            return val
    # Fall back to model.args on mlx-lm models.
    args = getattr(model, "args", None)
    for attr in ("max_position_embeddings", "context_length", "n_ctx"):
        val = getattr(args, attr, None)
        if isinstance(val, int) and 0 < val < 1_000_000:
            return val
    return None


if __name__ == "__main__":
    app()
