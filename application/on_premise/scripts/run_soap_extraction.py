"""CLI entrypoint: Transcript JSON -> validated SOAPNote JSON via local SLM.

Usage:
    # smoke test: one transcript, grammar-constrained mode
    uv run python application/on_premise/scripts/run_soap_extraction.py \
        --limit 1 --variant clean --structured-mode grammar

    # target one scenario, noisy variant
    uv run python application/on_premise/scripts/run_soap_extraction.py \
        --scenario-id endo-acr --variant noisy

    # A/B raw vs grammar
    uv run python application/on_premise/scripts/run_soap_extraction.py \
        --limit 1 --structured-mode raw

Reads transcripts from `application/on_premise/output/transcripts/<sid[:8]>/`
(Phase 2a output, gitignored) and writes SOAP notes to
`application/on_premise/output/soap_notes/<sid[:8]>/`. Both directories
are gitignored — un-anonymized SOAP notes stay on-prem.
"""

from __future__ import annotations

import json
import logging
import os
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

from on_premise.structuring.pipeline import transcript_to_soap
from on_premise.structuring.slm_engine import build_extractor, load_slm_config
from on_premise.stt.schema import Transcript

app = typer.Typer(add_completion=False, help="MedVoicePL on-premise SOAP extraction CLI")
console = Console()

DEFAULT_TRANSCRIPTS_DIR = Path("application/on_premise/output/transcripts")
DEFAULT_SOAP_DIR = Path("application/on_premise/output/soap_notes")
DEFAULT_SLM_CONFIG = Path("application/on_premise/configs/slm.yaml")


@app.command()
def extract(
    transcripts_dir: Path = typer.Option(DEFAULT_TRANSCRIPTS_DIR, "--transcripts-dir"),
    soap_dir: Path = typer.Option(DEFAULT_SOAP_DIR, "--soap-dir"),
    slm_config: Path = typer.Option(DEFAULT_SLM_CONFIG, "--slm-config"),
    variant: str = typer.Option(
        "clean",
        "--variant",
        help="clean | noisy | both. Matches the Phase 2a transcript.json / transcript_noisy.json layout.",
    ),
    scenario_id: list[str] = typer.Option(
        None,
        "--scenario-id",
        "-s",
        help="Only process transcripts whose scenario_id contains this prefix (repeatable).",
    ),
    backend: str | None = typer.Option(
        None,
        "--backend",
        help="mlx-lm | llama-cpp. Overrides $SLM_BACKEND for this run.",
    ),
    structured_mode: str | None = typer.Option(
        None,
        "--structured-mode",
        help="grammar | raw. Overrides $SLM_STRUCTURED_MODE for this run.",
    ),
    limit: int | None = typer.Option(None, "--limit", "-n"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the local SLM over every transcript bundle and write JSON SOAP notes."""
    load_dotenv()
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if variant not in {"clean", "noisy", "both"}:
        console.print(f"[red]--variant must be clean|noisy|both, got {variant!r}[/red]")
        raise typer.Exit(code=2)

    variants_to_run = ["clean", "noisy"] if variant == "both" else [variant]
    transcript_filenames = {
        "clean": "transcript.json",
        "noisy": "transcript_noisy.json",
    }

    transcript_dirs = (
        sorted(p for p in transcripts_dir.iterdir() if p.is_dir())
        if transcripts_dir.exists()
        else []
    )
    if not transcript_dirs:
        console.print(f"[red]No transcripts under {transcripts_dir}[/red]")
        raise typer.Exit(code=1)

    # Filter by scenario_id prefix
    if scenario_id:
        wanted = tuple(scenario_id)
        filtered: list[Path] = []
        for d in transcript_dirs:
            tpath = d / "transcript.json"
            if not tpath.exists():
                continue
            data = json.loads(tpath.read_text(encoding="utf-8"))
            sid = data.get("scenario_id") or ""
            if any(pref in sid for pref in wanted):
                filtered.append(d)
        transcript_dirs = filtered
    if limit:
        transcript_dirs = transcript_dirs[:limit]

    env = dict(os.environ)
    if backend:
        env["SLM_BACKEND"] = backend
    if structured_mode:
        env["SLM_STRUCTURED_MODE"] = structured_mode

    config = load_slm_config(slm_config)
    extractor = build_extractor(config, env)

    soap_dir.mkdir(parents=True, exist_ok=True)
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
        task = progress.add_task(
            "Structuring SOAP", total=len(transcript_dirs) * len(variants_to_run)
        )
        for tdir in transcript_dirs:
            for v in variants_to_run:
                t_path = tdir / transcript_filenames[v]
                if not t_path.exists():
                    progress.update(task, advance=1)
                    if variant == v:  # only log as failure if explicitly requested
                        failures.append((tdir.name, f"no {transcript_filenames[v]}"))
                    continue
                try:
                    transcript = Transcript.model_validate_json(t_path.read_text(encoding="utf-8"))
                    soap = transcript_to_soap(transcript, extractor=extractor)
                    out_path = (
                        soap_dir
                        / tdir.name
                        / ("soap_note_noisy.json" if v == "noisy" else "soap_note.json")
                    )
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(
                        soap.model_dump_json(indent=2),
                        encoding="utf-8",
                    )
                    console.log(
                        f"wrote {out_path} "
                        f"(dx={soap.assessment.primary_diagnosis!r}, "
                        f"icd={soap.assessment.icd10_code})"
                    )
                    successes.append(f"{tdir.name}:{v}")
                except Exception as exc:
                    failures.append((f"{tdir.name}:{v}", f"{type(exc).__name__}: {exc}"))
                progress.update(task, advance=1)

    console.rule("[bold]SOAP extraction summary")
    console.print(f"[green]Success:[/green] {len(successes)}")
    console.print(f"[red]Failures:[/red] {len(failures)}")
    for sid, msg in failures:
        console.print(f"  - [red]{sid}[/red]: {msg}")
    if failures:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
