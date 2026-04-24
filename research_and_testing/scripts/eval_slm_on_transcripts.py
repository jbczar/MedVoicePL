"""SOAP-extraction eval over the Phase 2a transcripts.

For each transcript bundle under
`application/on_premise/output/transcripts/<sid[:8]>/` the script:

1. Loads the `transcript.json` (and `transcript_noisy.json` for the
   `both` variant) produced by Phase 2a.
2. Loads the matching scenario JSON from `dataset_pipeline/output/scenarios/`.
3. Runs `on_premise.structuring.pipeline.transcript_to_soap()` with the
   current SLM extractor (`SLM_BACKEND` from env).
4. Scores the output against the scenario ground truth (CC recall,
   diagnosis match, ICD-10 match, plan-nonempty).
5. Emits a Rich table + JSON artifact under
   `research_and_testing/results/slm_eval_<timestamp>.json`.

Primarily a feedback loop for prompt + grammar tuning: `--structured-mode grammar`
vs `raw` is the A/B that tells us whether lm-format-enforcer is
earning its dep.

Usage:
    uv run python research_and_testing/scripts/eval_slm_on_transcripts.py --limit 1
    uv run python research_and_testing/scripts/eval_slm_on_transcripts.py --structured-mode raw
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from dataset_pipeline.scenario_generator.schema import ScenarioModel
from evaluation.slm_eval.benchmark_slm import evaluate_one, summarize
from on_premise.structuring.slm_engine import build_extractor, load_slm_config
from on_premise.stt.schema import Transcript

app = typer.Typer(add_completion=False, help="SOAP eval over Phase 2a transcripts")
console = Console()
log = logging.getLogger(__name__)

DEFAULT_TRANSCRIPTS_DIR = Path("application/on_premise/output/transcripts")
DEFAULT_SCENARIOS_DIR = Path("dataset_pipeline/output/scenarios")
DEFAULT_RESULTS_DIR = Path("research_and_testing/results")
DEFAULT_SLM_CONFIG = Path("application/on_premise/configs/slm.yaml")


@app.command()
def run(
    transcripts_dir: Path = typer.Option(DEFAULT_TRANSCRIPTS_DIR, "--transcripts-dir"),
    scenarios_dir: Path = typer.Option(DEFAULT_SCENARIOS_DIR, "--scenarios-dir"),
    slm_config: Path = typer.Option(DEFAULT_SLM_CONFIG, "--slm-config"),
    results_dir: Path = typer.Option(DEFAULT_RESULTS_DIR, "--results-dir"),
    variant: str = typer.Option("clean", "--variant", help="clean | noisy | both"),
    structured_mode: str | None = typer.Option(
        None,
        "--structured-mode",
        help="grammar | raw. Overrides $SLM_STRUCTURED_MODE for this run.",
    ),
    backend: str | None = typer.Option(
        None,
        "--backend",
        help="mlx-lm | llama-cpp. Overrides $SLM_BACKEND.",
    ),
    limit: int | None = typer.Option(None, "--limit", "-n"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run SOAP extraction over transcripts and score against scenarios."""
    # Load .env inside run() — Phase 2a lesson: module-level load_dotenv
    # misses .env when the CLI is invoked from a subdirectory.
    load_dotenv()
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if variant not in {"clean", "noisy", "both"}:
        console.print(f"[red]--variant must be clean|noisy|both, got {variant!r}[/red]")
        raise typer.Exit(code=2)

    env = dict(os.environ)
    if structured_mode:
        env["SLM_STRUCTURED_MODE"] = structured_mode
    if backend:
        env["SLM_BACKEND"] = backend

    scenarios = _load_scenarios(scenarios_dir)
    transcript_dirs = sorted(p for p in transcripts_dir.iterdir() if p.is_dir()) \
        if transcripts_dir.exists() else []
    if limit:
        transcript_dirs = transcript_dirs[:limit]
    if not transcript_dirs:
        console.print(f"[red]No transcripts under {transcripts_dir}[/red]")
        raise typer.Exit(code=1)

    config = load_slm_config(slm_config)
    extractor = build_extractor(config, env)

    variants_to_run = ["clean", "noisy"] if variant == "both" else [variant]
    fname = {"clean": "transcript.json", "noisy": "transcript_noisy.json"}

    rows = []
    for tdir in transcript_dirs:
        for v in variants_to_run:
            tpath = tdir / fname[v]
            if not tpath.exists():
                continue
            transcript = Transcript.model_validate_json(tpath.read_text(encoding="utf-8"))
            sid = transcript.scenario_id
            if not sid or sid not in scenarios:
                log.warning("no matching scenario for %s; skipping", sid)
                continue
            scenario = scenarios[sid]
            row = evaluate_one(scenario, transcript, extractor=extractor, variant=v)
            rows.append(row)
            console.log(
                f"[{v}] {sid[:8]}: schema={row.schema_ok} "
                f"cc={row.cc_recall:.2f} dx={row.diag_match} "
                f"icd={row.icd10_match} plan={row.plan_nonempty} "
                f"({row.duration_s:.1f}s)"
            )

    if not rows:
        console.print("[red]No eval rows produced[/red]")
        raise typer.Exit(code=1)

    summary = summarize(rows)

    table = Table(title=f"SLM SOAP eval — mode={env.get('SLM_STRUCTURED_MODE', 'grammar')}")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("n", str(summary.n))
    table.add_row("schema compliance", f"{summary.schema_compliance_rate * 100:.1f}%")
    table.add_row("mean CC recall", f"{summary.mean_cc_recall:.3f}")
    table.add_row("diagnosis match", f"{summary.diagnosis_match_rate * 100:.1f}%")
    table.add_row("ICD-10 match", f"{summary.icd10_match_rate * 100:.1f}%")
    table.add_row("plan non-empty", f"{summary.plan_nonempty_rate * 100:.1f}%")
    table.add_row("mean duration", f"{summary.mean_duration_s:.1f}s")
    console.print(table)

    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"slm_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(
        json.dumps(
            {
                "timestamp": datetime.now().astimezone().isoformat(),
                "backend": extractor.name,
                "device": extractor.device,
                "structured_mode": extractor.structured_mode,
                "summary": {
                    "n": summary.n,
                    "schema_compliance_rate": summary.schema_compliance_rate,
                    "mean_cc_recall": summary.mean_cc_recall,
                    "diagnosis_match_rate": summary.diagnosis_match_rate,
                    "icd10_match_rate": summary.icd10_match_rate,
                    "plan_nonempty_rate": summary.plan_nonempty_rate,
                    "mean_duration_s": summary.mean_duration_s,
                },
                "rows": [asdict(r) for r in rows],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    console.print(f"[green]wrote {out_path}[/green]")


def _load_scenarios(scenarios_dir: Path) -> dict[str, ScenarioModel]:
    out: dict[str, ScenarioModel] = {}
    for path in sorted(scenarios_dir.glob("*.json")):
        if path.name == "failures.json":
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        model = ScenarioModel.model_validate(data)
        out[model.id] = model
    return out


if __name__ == "__main__":
    app()
