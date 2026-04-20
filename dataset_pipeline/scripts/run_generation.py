"""CLI entry point: run the full dataset generation pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console

from dataset_pipeline.pipeline import run_scenarios

app = typer.Typer(add_completion=False, help="MedVoicePL dataset pipeline CLI")
console = Console()

DEFAULT_CONFIG = Path("dataset_pipeline/configs/scenarios.yaml")
DEFAULT_OUTPUT = Path("dataset_pipeline/output/scenarios")


@app.command()
def generate(
    config: Path = typer.Option(DEFAULT_CONFIG, "--config", "-c", exists=True),
    output: Path = typer.Option(DEFAULT_OUTPUT, "--output", "-o"),
    per_specialization: int | None = typer.Option(
        None,
        "--per-specialization",
        "-n",
        help="Override scenarios_per_specialization from YAML (MVP default: 2).",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Generate scenarios end-to-end and write JSON files to the output dir."""
    load_dotenv()
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    report = run_scenarios(
        config_path=config,
        output_dir=output,
        per_specialization=per_specialization,
    )

    console.rule("[bold]Generation summary")
    console.print(f"[green]Written:[/green] {report.success_count}")
    console.print(f"[red]Failed:[/red]  {len(report.failures)}")
    if report.failures:
        console.print(f"See [yellow]{output / 'failures.json'}[/yellow]")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
