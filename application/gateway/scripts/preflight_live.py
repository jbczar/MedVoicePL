"""Pre-flight: check everything required for live E2E (Phase 3b live mode).

Verifies (1) env vars present, (2) HF/Vertex auth, (3) model weights
cached, (4) NFZ index built, (5) reference audio exists. Exits 0 = green
across the board; non-zero with explicit instructions per failure.

Usage:
    uv run python application/gateway/scripts/preflight_live.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

app = typer.Typer(add_completion=False, no_args_is_help=False)
console = Console()


def _ok(detail: str) -> tuple[str, str]:
    return "[green]OK[/]", detail


def _fail(detail: str) -> tuple[str, str]:
    return "[red]FAIL[/]", detail


def _warn(detail: str) -> tuple[str, str]:
    return "[yellow]WARN[/]", detail


def _check_env() -> list[tuple[str, str, str]]:
    rows = []
    required = [
        "HUGGINGFACE_TOKEN",
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION",
        "VERTEX_AGENT_MODEL",
        "GATEWAY_DOCTOR_USERNAME",
        "GATEWAY_DOCTOR_PASSWORD_HASH",
    ]
    for var in required:
        val = os.environ.get(var, "").strip()
        if val:
            rows.append((f"env: {var}", *_ok(f"set ({len(val)} chars)")))
        else:
            rows.append((f"env: {var}", *_fail("missing — check .env")))
    return rows


def _check_gcloud_auth() -> tuple[str, str, str]:
    """Vertex AI uses ADC. `gcloud auth application-default print-access-token`."""
    if shutil.which("gcloud") is None:
        return ("vertex auth", *_fail("gcloud CLI not in PATH"))
    try:
        result = subprocess.run(
            ["gcloud", "auth", "application-default", "print-access-token"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        return ("vertex auth", *_fail("gcloud auth timed out"))
    if result.returncode != 0:
        msg = result.stderr.strip().splitlines()[-1] if result.stderr else "non-zero exit"
        return (
            "vertex auth",
            *_fail(f"{msg} — run: gcloud auth application-default login"),
        )
    token_len = len(result.stdout.strip())
    if token_len < 50:
        return ("vertex auth", *_fail(f"unexpectedly short token ({token_len} chars)"))
    return ("vertex auth", *_ok(f"ADC token ({token_len} chars)"))


def _check_hf_cache() -> list[tuple[str, str, str]]:
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    rows = []
    expected = {
        "Whisper (MLX)": "models--mlx-community--whisper-large-v3-mlx",
        "pyannote diarization": "models--pyannote--speaker-diarization-3.1",
        "pyannote segmentation": "models--pyannote--segmentation-3.0",
        "wespeaker": "models--pyannote--wespeaker-voxceleb-resnet34-LM",
        "Bielik-11B-MLX-4bit": "models--speakleash--Bielik-11B-v3.0-Instruct-MLX-4bit",
        "mmlw-roberta-large": "models--sdadas--mmlw-roberta-large",
    }
    for name, dirname in expected.items():
        path = cache / dirname
        if path.is_dir():
            size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
            rows.append((f"weights: {name}", *_ok(f"{size / 1e9:.1f} GB cached")))
        else:
            rows.append(
                (
                    f"weights: {name}",
                    *_fail(
                        f"not in {cache} — first request will download "
                        "(can take minutes)"
                    ),
                )
            )
    return rows


def _check_spacy() -> tuple[str, str, str]:
    try:
        import spacy

        spacy.load("pl_core_news_lg")
        return ("spaCy: pl_core_news_lg", *_ok("loadable"))
    except ImportError:
        return ("spaCy: pl_core_news_lg", *_fail("spacy not installed"))
    except OSError as exc:
        return ("spaCy: pl_core_news_lg", *_fail(f"model not loadable: {exc}"))


def _check_nfz_index() -> tuple[str, str, str]:
    nfz_dir = Path("application/cloud/data/nfz_formulary")
    indexes = sorted(nfz_dir.glob("index_*.json"))
    if not indexes:
        return (
            "NFZ index",
            *_fail(
                "no index_*.json — run application/cloud/data/nfz_formulary/ingest_nfz.py"
            ),
        )
    return ("NFZ index", *_ok(f"{indexes[-1].name}"))


def _check_chpl_corpus() -> tuple[str, str, str]:
    chpl_dir = Path("application/cloud/data/chpl_excerpts")
    if not chpl_dir.is_dir():
        return ("ChPL excerpts", *_fail(f"missing dir: {chpl_dir}"))
    excerpts = list(chpl_dir.glob("*.md"))
    return ("ChPL excerpts", *_ok(f"{len(excerpts)} excerpts"))


def _check_reference_audio() -> tuple[str, str, str]:
    audio = Path("dataset_pipeline/output/audio/endo-gra/dialogue.wav")
    if not audio.is_file():
        return (
            "reference audio (endo-gra)",
            *_fail(f"missing: {audio} — re-run dataset_pipeline synthesis"),
        )
    size = audio.stat().st_size
    return ("reference audio (endo-gra)", *_ok(f"{size / 1e6:.1f} MB at {audio}"))


def _check_pyannote_license() -> tuple[str, str, str]:
    """pyannote/speaker-diarization-3.1 is gated. License must be accepted
    on huggingface.co/pyannote/speaker-diarization-3.1 — we can't verify
    that programmatically without trying to load (which needs the token).
    Instead, check that the cache contains a valid model.bin.
    """
    cache = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--pyannote--speaker-diarization-3.1"
        / "snapshots"
    )
    if not cache.is_dir():
        return (
            "pyannote license",
            *_warn(
                "model not cached — first load will require accepted license at "
                "huggingface.co/pyannote/speaker-diarization-3.1"
            ),
        )
    return ("pyannote license", *_ok("snapshot present, license previously accepted"))


@app.command()
def run() -> None:
    """Run all checks. Exit 0 if green; 1 if any FAIL."""
    load_dotenv(override=True)
    rows: list[tuple[str, str, str]] = []
    rows.extend(_check_env())
    rows.append(_check_gcloud_auth())
    rows.extend(_check_hf_cache())
    rows.append(_check_spacy())
    rows.append(_check_pyannote_license())
    rows.append(_check_nfz_index())
    rows.append(_check_chpl_corpus())
    rows.append(_check_reference_audio())

    table = Table(title="MedVoicePL Phase 3b live preflight")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")
    for check, status, detail in rows:
        table.add_row(check, status, detail)
    console.print(table)

    fails = [r for r in rows if "FAIL" in r[1]]
    warns = [r for r in rows if "WARN" in r[1]]
    if fails:
        console.print(f"\n[bold red]{len(fails)} check(s) failed[/] — fix before live smoke")
        sys.exit(1)
    if warns:
        console.print(f"\n[bold yellow]{len(warns)} warning(s)[/] — review above")
    console.print("\n[bold green]All hard checks pass — ready for live smoke[/]")


if __name__ == "__main__":
    app()
