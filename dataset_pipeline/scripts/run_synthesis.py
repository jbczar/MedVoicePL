"""CLI entrypoint: synthesize audio from Phase 1a scenario JSONs.

Usage:
    uv run python dataset_pipeline/scripts/run_synthesis.py --limit 1
    uv run python dataset_pipeline/scripts/run_synthesis.py \
        --scenarios-dir dataset_pipeline/output/scenarios \
        --audio-dir dataset_pipeline/output/audio \
        --scenario-id kardiologia_abc123
"""

from __future__ import annotations

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

from dataset_pipeline.pipeline import run_audio

app = typer.Typer(add_completion=False, help="MedVoicePL audio synthesis CLI")
console = Console()

DEFAULT_SCENARIOS_DIR = Path("dataset_pipeline/output/scenarios")
DEFAULT_AUDIO_DIR = Path("dataset_pipeline/output/audio")
DEFAULT_TTS_CONFIG = Path("dataset_pipeline/configs/tts_profiles.yaml")
DEFAULT_POSTPROCESS_CONFIG = Path("dataset_pipeline/configs/audio_postprocess.yaml")


@app.command()
def synthesize(
    scenarios_dir: Path = typer.Option(DEFAULT_SCENARIOS_DIR, "--scenarios-dir"),
    audio_dir: Path = typer.Option(DEFAULT_AUDIO_DIR, "--audio-dir"),
    tts_config: Path = typer.Option(DEFAULT_TTS_CONFIG, "--tts-config"),
    postprocess_config: Path = typer.Option(
        DEFAULT_POSTPROCESS_CONFIG, "--postprocess-config"
    ),
    scenario_id: list[str] = typer.Option(
        None,
        "--scenario-id",
        "-s",
        help="Only synthesize scenarios whose filename contains this prefix (repeatable).",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        "-n",
        help="Stop after this many successful bundles. Use 1 for a smoke test.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Synthesize per-turn WAVs + mixed dialogue for each scenario."""
    load_dotenv()
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    audio_dir.mkdir(parents=True, exist_ok=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        # We don't know the total up front without walking the dir twice, so
        # use an indeterminate spinner by setting total=None and rely on the
        # final summary.
        task = progress.add_task("Synthesizing", total=None)
        report = run_audio(
            scenarios_dir=scenarios_dir,
            audio_out_dir=audio_dir,
            tts_config_path=tts_config,
            postprocess_config_path=postprocess_config,
            scenario_filter=list(scenario_id) if scenario_id else None,
            limit=limit,
        )
        progress.update(task, completed=report.success_count)

    console.rule("[bold]Synthesis summary")
    console.print(f"[green]Bundles:[/green] {report.success_count}")
    console.print(f"[red]Failures:[/red] {len(report.failures)}")
    for scenario_id_str, msg in report.failures:
        console.print(f"  - [red]{scenario_id_str}[/red]: {msg}")
    if report.failures:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
