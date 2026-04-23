"""Preflight: verify STT + diarization prerequisites on a fresh clone.

Checks:
1. `mlx-whisper` importable (Apple Silicon path).
2. `faster-whisper` importable (portable path).
3. HuggingFace token present in env.
4. `pyannote/speaker-diarization-3.1` license accepted (pipeline actually
   loads — this is the only reliable gate for HF license acceptance).

Prints a red/green table and exits non-zero if anything is missing, with
an actionable pointer for each failure.

Usage:
    uv run python application/on_premise/scripts/check_stt_models.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

app = typer.Typer(add_completion=False, help="STT prerequisite checker")
console = Console()


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    fix: str | None = None


@app.command()
def check(
    skip_pyannote_load: bool = typer.Option(
        False,
        "--skip-pyannote-load",
        help="Skip the actual pyannote pipeline load (fast mode — only checks import + token presence).",
    ),
) -> None:
    load_dotenv()
    results: list[CheckResult] = []
    results.append(_check_mlx())
    results.append(_check_faster_whisper())
    results.append(_check_hf_token())
    results.append(_check_pyannote(skip_load=skip_pyannote_load))

    table = Table(title="STT prerequisites")
    table.add_column("check")
    table.add_column("status")
    table.add_column("detail")
    for r in results:
        table.add_row(
            r.name,
            "[green]OK[/green]" if r.ok else "[red]MISSING[/red]",
            r.detail,
        )
    console.print(table)

    failed = [r for r in results if not r.ok]
    if failed:
        console.rule("[bold red]Action required")
        for r in failed:
            if r.fix:
                console.print(f"• [yellow]{r.name}[/yellow]: {r.fix}")
        sys.exit(1)
    console.rule("[bold green]All prerequisites satisfied")


def _check_mlx() -> CheckResult:
    try:
        import mlx_whisper  # type: ignore[import-not-found]  # noqa: F401

        return CheckResult("mlx-whisper", True, "importable", None)
    except ImportError as exc:
        return CheckResult(
            "mlx-whisper",
            False,
            str(exc),
            "uv pip install mlx-whisper  (Apple Silicon only; Linux/CI can skip with WHISPER_BACKEND=faster-whisper)",
        )


def _check_faster_whisper() -> CheckResult:
    try:
        import faster_whisper  # type: ignore[import-not-found]  # noqa: F401

        return CheckResult("faster-whisper", True, "importable", None)
    except ImportError as exc:
        return CheckResult(
            "faster-whisper",
            False,
            str(exc),
            "uv pip install faster-whisper",
        )


def _check_hf_token() -> CheckResult:
    token = os.environ.get("HUGGINGFACE_TOKEN", "").strip()
    if token:
        return CheckResult("HUGGINGFACE_TOKEN", True, f"set ({len(token)} chars)", None)
    return CheckResult(
        "HUGGINGFACE_TOKEN",
        False,
        "not set in env",
        "1) Create read token at https://huggingface.co/settings/tokens\n"
        "             2) Export HUGGINGFACE_TOKEN=... or add it to .env",
    )


def _check_pyannote(*, skip_load: bool) -> CheckResult:
    try:
        from pyannote.audio import Pipeline  # type: ignore[import-not-found]
    except ImportError as exc:
        return CheckResult(
            "pyannote.audio",
            False,
            str(exc),
            "uv pip install pyannote.audio",
        )
    if skip_load:
        return CheckResult("pyannote/speaker-diarization-3.1", True, "import only (load skipped)", None)

    token = os.environ.get("HUGGINGFACE_TOKEN", "").strip()
    if not token:
        return CheckResult(
            "pyannote/speaker-diarization-3.1",
            False,
            "no token to validate license",
            "set HUGGINGFACE_TOKEN first",
        )
    try:
        # pyannote.audio 3.x renamed `use_auth_token` -> `token` (huggingface_hub
        # alignment). We try the new name first, fall back to the old one so
        # this preflight works across pyannote versions.
        try:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=token,
            )
        except TypeError:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=token,
            )
    except Exception as exc:
        return CheckResult(
            "pyannote/speaker-diarization-3.1",
            False,
            f"{type(exc).__name__}: {exc}",
            # Accept license on ALL THREE: the 3.1 pipeline chains into both the
            # segmentation model and the community speaker-embedding model. HF's
            # gating is per-repo, so missing any one of them 403s the load.
            "Accept the license on all three:\n"
            "  - https://huggingface.co/pyannote/speaker-diarization-3.1\n"
            "  - https://huggingface.co/pyannote/segmentation-3.0\n"
            "  - https://huggingface.co/pyannote/speaker-diarization-community-1",
        )
    if pipeline is None:
        return CheckResult(
            "pyannote/speaker-diarization-3.1",
            False,
            "Pipeline.from_pretrained returned None",
            "License not accepted or token lacks read scope — see links in the message above.",
        )
    return CheckResult("pyannote/speaker-diarization-3.1", True, "loaded", None)


if __name__ == "__main__":
    app()
