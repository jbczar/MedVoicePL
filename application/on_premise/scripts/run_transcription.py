"""CLI entrypoint: transcribe + diarize the Phase 1b audio bundles.

Usage:
    # smoke test: one bundle, clean only
    uv run python application/on_premise/scripts/run_transcription.py --limit 1 --variant clean

    # target a single scenario by id prefix
    uv run python application/on_premise/scripts/run_transcription.py \
        --scenario-id endo-acr --variant both

    # override env-provided backend
    uv run python application/on_premise/scripts/run_transcription.py \
        --backend faster-whisper --limit 1

Writes `<transcripts-dir>/<scenario_id[:8]>/transcript.json` for clean and
`transcript_noisy.json` for the noisy variant. Transcripts contain
un-anonymized text and live under `application/on_premise/output/`, which is
gitignored — this is the trust-boundary posture for Phase 2a.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from on_premise.stt.diarization import build_diarizer
from on_premise.stt.pipeline import transcribe_and_diarize
from on_premise.stt.whisper_engine import build_transcriber, load_whisper_config

app = typer.Typer(add_completion=False, help="MedVoicePL on-premise STT CLI")
console = Console()

DEFAULT_AUDIO_DIR = Path("dataset_pipeline/output/audio")
DEFAULT_TRANSCRIPTS_DIR = Path("application/on_premise/output/transcripts")
DEFAULT_WHISPER_CONFIG = Path("application/on_premise/configs/whisper.yaml")


@app.command()
def transcribe(
    audio_dir: Path = typer.Option(DEFAULT_AUDIO_DIR, "--audio-dir"),
    transcripts_dir: Path = typer.Option(DEFAULT_TRANSCRIPTS_DIR, "--transcripts-dir"),
    whisper_config: Path = typer.Option(DEFAULT_WHISPER_CONFIG, "--whisper-config"),
    variant: str = typer.Option(
        "clean",
        "--variant",
        help="clean | noisy | both. Matches the Phase 1b.1 dialogue_noisy.wav layout.",
    ),
    scenario_id: list[str] = typer.Option(
        None,
        "--scenario-id",
        "-s",
        help="Only process bundles whose scenario_id contains this prefix (repeatable).",
    ),
    backend: str | None = typer.Option(
        None,
        "--backend",
        help="mlx | faster-whisper. Overrides $WHISPER_BACKEND for this run.",
    ),
    limit: int | None = typer.Option(None, "--limit", "-n"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run Whisper + pyannote on each audio bundle and write JSON transcripts."""
    load_dotenv()
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if variant not in {"clean", "noisy", "both"}:
        console.print(f"[red]--variant must be clean|noisy|both, got {variant!r}[/red]")
        raise typer.Exit(code=2)

    bundles = sorted(audio_dir.glob("*/bundle.json"))
    if not bundles:
        console.print(f"[red]No audio bundles under {audio_dir}[/red]")
        raise typer.Exit(code=1)
    if scenario_id:
        wanted = tuple(scenario_id)
        bundles = [
            b
            for b in bundles
            if any(
                prefix in json.loads(b.read_text(encoding="utf-8"))["scenario_id"]
                for prefix in wanted
            )
        ]
    if limit:
        bundles = bundles[:limit]

    # Load config, apply env overrides, instantiate heavy models ONCE.
    import os

    env = dict(os.environ)
    if backend:
        env["WHISPER_BACKEND"] = backend

    config = load_whisper_config(whisper_config)
    transcriber = build_transcriber(config, env)
    diarizer = build_diarizer(config=None, env=env)

    transcripts_dir.mkdir(parents=True, exist_ok=True)
    successes: list[str] = []
    failures: list[tuple[str, str]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Transcribing", total=len(bundles))
        for bundle_path in bundles:
            data = json.loads(bundle_path.read_text(encoding="utf-8"))
            sid = data["scenario_id"]
            bundle_dir = bundle_path.parent
            out_dir = transcripts_dir / sid[:8]
            out_dir.mkdir(parents=True, exist_ok=True)
            try:
                if variant in {"clean", "both"}:
                    _write_transcript(
                        bundle_dir / "dialogue.wav",
                        out_dir / "transcript.json",
                        transcriber=transcriber,
                        diarizer=diarizer,
                        scenario_id=sid,
                    )
                if variant in {"noisy", "both"}:
                    noisy_path = bundle_dir / "dialogue_noisy.wav"
                    if noisy_path.exists():
                        _write_transcript(
                            noisy_path,
                            out_dir / "transcript_noisy.json",
                            transcriber=transcriber,
                            diarizer=diarizer,
                            scenario_id=sid,
                        )
                    elif variant == "noisy":
                        failures.append((sid, "no dialogue_noisy.wav (run Phase 1b.1 first)"))
                        progress.update(task, advance=1)
                        continue
                successes.append(sid)
            except Exception as exc:
                failures.append((sid, f"{type(exc).__name__}: {exc}"))
            progress.update(task, advance=1)

    console.rule("[bold]Transcription summary")
    console.print(f"[green]Success:[/green] {len(successes)}")
    console.print(f"[red]Failures:[/red] {len(failures)}")
    for sid, msg in failures:
        console.print(f"  - [red]{sid}[/red]: {msg}")
    if failures:
        raise typer.Exit(code=1)


def _write_transcript(
    wav_path: Path,
    out_path: Path,
    *,
    transcriber,  # type: ignore[no-untyped-def]
    diarizer,  # type: ignore[no-untyped-def]
    scenario_id: str,
) -> None:
    transcript = transcribe_and_diarize(
        wav_path,
        transcriber=transcriber,
        diarizer=diarizer,
        scenario_id=scenario_id,
    )
    out_path.write_text(
        transcript.model_dump_json(indent=2),
        encoding="utf-8",
    )
    console.log(f"wrote {out_path} ({len(transcript.segments)} segs, {transcript.duration_s:.1f}s)")


if __name__ == "__main__":
    app()
