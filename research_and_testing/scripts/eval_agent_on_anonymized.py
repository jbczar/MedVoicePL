"""CLI: evaluate the cloud agent against hand-curated drug references.

Iterates `application/cloud/output/agent_suggestions/<sid>/suggestion.json`
joined with `research_and_testing/test_cases/expected_outputs/drug_references/<sid>.json`
and writes `research_and_testing/results/agent_eval_<timestamp>.json`.

Mirrors `eval_slm_on_transcripts.py` argv + output shape.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from evaluation.rag_eval.benchmark_rag import EvalRow, evaluate_one, summarize
from evaluation.rag_eval.suggestion_accuracy import GroundTruth
from shared.schemas import AgentSuggestion

app = typer.Typer(add_completion=False, help="MedVoicePL cloud agent eval")
console = Console()

DEFAULT_SUGGESTIONS_DIR = Path("application/cloud/output/agent_suggestions")
DEFAULT_GT_DIR = Path("research_and_testing/test_cases/expected_outputs/drug_references")
DEFAULT_RESULTS_DIR = Path("research_and_testing/results")


@app.command()
def evaluate(
    suggestions_dir: Path = typer.Option(DEFAULT_SUGGESTIONS_DIR, "--suggestions-dir"),
    gt_dir: Path = typer.Option(DEFAULT_GT_DIR, "--gt-dir"),
    results_dir: Path = typer.Option(DEFAULT_RESULTS_DIR, "--results-dir"),
    limit: int | None = typer.Option(None, "--limit", "-n"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    load_dotenv()
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    rows: list[EvalRow] = []
    targets = sorted(suggestions_dir.iterdir()) if suggestions_dir.is_dir() else []
    for child in targets:
        if not child.is_dir():
            continue
        sid = child.name
        suggestion_path = child / "suggestion.json"
        gt_path = gt_dir / f"{sid}.json"
        if not (suggestion_path.is_file() and gt_path.is_file()):
            continue

        suggestion = AgentSuggestion.model_validate_json(
            suggestion_path.read_text("utf-8")
        )
        expected = GroundTruth.from_dict(json.loads(gt_path.read_text("utf-8")))
        rows.append(evaluate_one(scenario_id=sid, suggestion=suggestion, expected=expected))
        if limit is not None and len(rows) >= limit:
            break

    summary = summarize(rows)

    table = Table(title="Cloud agent — eval")
    table.add_column("scenario")
    table.add_column("recall", justify="right")
    table.add_column("prec", justify="right")
    table.add_column("nfz", justify="center")
    table.add_column("empty_ok", justify="center")
    table.add_column("prov_ok", justify="center")
    table.add_column("n_drugs", justify="right")
    table.add_column("dur_s", justify="right")
    for r in rows:
        table.add_row(
            r.scenario_id,
            f"{r.drug_recall:.2f}",
            f"{r.drug_precision:.2f}",
            "[green]Y[/green]" if r.nfz_accuracy else "[red]N[/red]",
            "[green]Y[/green]" if r.empty_list_correct else "[red]N[/red]",
            "[green]Y[/green]" if r.provenance_valid else "[red]N[/red]",
            str(r.num_drugs_suggested),
            f"{r.duration_s:.2f}",
        )
    console.print(table)
    console.print(
        f"[bold]Aggregate:[/bold] recall={summary.mean_recall():.2f} "
        f"precision={summary.mean_precision():.2f} "
        f"nfz_acc={summary.nfz_accuracy_rate():.2f} "
        f"empty_correct={summary.empty_list_correctness_rate():.2f} "
        f"provenance={summary.provenance_validity_rate():.2f} "
        f"mean_dur={summary.mean_duration_s():.2f}s"
    )

    if rows:
        results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        out_path = results_dir / f"agent_eval_{timestamp}.json"
        out_path.write_text(
            json.dumps(summary.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        console.print(f"[dim]wrote {out_path}[/dim]")


if __name__ == "__main__":  # pragma: no cover
    app()
