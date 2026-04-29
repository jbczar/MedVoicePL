"""CLI: run the cloud agent on anonymized SOAP notes.

Usage::

    uv run python application/cloud/scripts/run_agent.py --limit 1
    uv run python application/cloud/scripts/run_agent.py --scenario-id endo-gra --force

Reads `application/on_premise/output/anonymized/<sid>/anonymized_soap.json`
and writes `application/cloud/output/agent_suggestions/<sid>/suggestion.json`
(gitignored). Prints a Rich table with drug counts + duration.

Mirrors `run_anonymization.py` argv shape.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from shared.schemas import AgentSuggestion, AnonymizedSOAPNote

app = typer.Typer(add_completion=False, help="MedVoicePL cloud agent CLI")
console = Console()

DEFAULT_ANONYMIZED_DIR = Path("application/on_premise/output/anonymized")
DEFAULT_OUTPUT_DIR = Path("application/cloud/output/agent_suggestions")


@app.command()
def run(
    anonymized_dir: Path = typer.Option(DEFAULT_ANONYMIZED_DIR, "--anonymized-dir"),
    output_dir: Path = typer.Option(DEFAULT_OUTPUT_DIR, "--output-dir"),
    scenario_id: list[str] | None = typer.Option(
        None, "--scenario-id", "-s", help="Filter to scenarios whose dir starts with this."
    ),
    limit: int | None = typer.Option(None, "--limit", "-n"),
    force: bool = typer.Option(False, "--force", help="Re-run even if cached."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Iterate every anonymized scenario and run the cloud agent."""
    load_dotenv()
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not anonymized_dir.is_dir():
        console.print(f"[red]anonymized dir not found:[/red] {anonymized_dir}")
        raise typer.Exit(code=1)

    targets = _select_scenarios(anonymized_dir, scenario_id=scenario_id, limit=limit)
    if not targets:
        console.print("[yellow]No scenarios matched.[/yellow]")
        raise typer.Exit(code=1)

    agent = _build_agent()

    table = Table(title="Cloud agent — suggestions")
    table.add_column("scenario")
    table.add_column("drugs", justify="right")
    table.add_column("referrals", justify="right")
    table.add_column("tests", justify="right")
    table.add_column("notes", justify="right")
    table.add_column("duration_s", justify="right")
    table.add_column("status")

    for sid, soap_path in targets:
        out_dir = output_dir / sid
        out_path = out_dir / "suggestion.json"
        if out_path.is_file() and not force:
            cached = AgentSuggestion.model_validate_json(out_path.read_text("utf-8"))
            table.add_row(
                sid,
                str(len(cached.drugs)),
                str(len(cached.referrals)),
                str(len(cached.tests)),
                str(len(cached.notes)),
                f"{cached.metadata.duration_s:.2f}",
                "[dim]cached[/dim]",
            )
            continue
        soap = AnonymizedSOAPNote.model_validate_json(soap_path.read_text("utf-8"))
        started = time.time()
        try:
            suggestion = agent.run(soap)
            status = "[green]ok[/green]"
        except Exception as exc:
            console.print(f"[red]{sid} failed:[/red] {exc}")
            table.add_row(sid, "-", "-", "-", "-", f"{time.time() - started:.2f}", "[red]err[/red]")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(suggestion.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(tmp, out_path)
        table.add_row(
            sid,
            str(len(suggestion.drugs)),
            str(len(suggestion.referrals)),
            str(len(suggestion.tests)),
            str(len(suggestion.notes)),
            f"{suggestion.metadata.duration_s:.2f}",
            status,
        )

    console.print(table)


def _select_scenarios(
    anonymized_dir: Path,
    *,
    scenario_id: list[str] | None,
    limit: int | None,
) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for child in sorted(anonymized_dir.iterdir()):
        if not child.is_dir():
            continue
        soap_path = child / "anonymized_soap.json"
        if not soap_path.is_file():
            continue
        if scenario_id and not any(child.name.startswith(s) for s in scenario_id):
            continue
        out.append((child.name, soap_path))
        if limit is not None and len(out) >= limit:
            break
    return out


def _build_agent():  # type: ignore[no-untyped-def]
    """Lazy-import so `--help` works without Vertex auth."""
    from cloud.api.app import get_agent  # reuses the production factory

    return get_agent()


if __name__ == "__main__":  # pragma: no cover
    app()
