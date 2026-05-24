"""Phase 4 step 5 — generate ground-truth `expected_slots` for each scenario.

For each `dataset_pipeline/output/scenarios/*.json`, asks Vertex Gemini 2.5 Pro
to produce a structured `GroundTruth` payload (matching the existing 8 hand-
curated files in `research_and_testing/test_cases/expected_outputs/drug_references/`).

The generated ground-truth backs `eval_agent_on_anonymized.py` recall/precision
metrics. Each slot lists *acceptable* ATC codes (clinically equivalent first-
line drugs); the agent's prediction counts as a hit if it matches any code in
any slot.

Idempotent — skips files that already exist on disk. The 8 legacy ground-truth
files were authored by hand during Phase 3a and are kept; new 40 scenarios get
LLM-generated GT with manual review on the top-5 most-uncertain cases.

Run::

    uv run python research_and_testing/scripts/generate_ground_truth.py \\
        --concurrency 8
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path

import typer
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

console = Console()
log = logging.getLogger("generate_ground_truth")
app = typer.Typer(add_completion=False, help="Generate `expected_slots` ground truth for scenarios.")

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIOS_DIR = REPO_ROOT / "dataset_pipeline/output/scenarios"
GT_DIR = REPO_ROOT / "research_and_testing/test_cases/expected_outputs/drug_references"
DEFAULT_MODEL = os.environ.get("VERTEX_AGENT_MODEL", "gemini-2.5-pro")


# ---------------------------------------------------------------------------
# Schemas — mirror the existing hand-curated JSON shape
# ---------------------------------------------------------------------------


class DrugSlot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(
        description="Human description of the therapeutic role this slot fulfils."
    )
    acceptable: list[str] = Field(
        min_length=1,
        description="ATC codes (7-char format, e.g. C09AA05) that satisfy this slot.",
    )


class GroundTruth(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    diagnosis_label: str = Field(
        description="One-line clinical summary of the expected diagnosis."
    )
    diagnostic_only: bool = Field(
        description=(
            "True if the visit is exploratory (e.g. unexplained symptoms requiring "
            "workup before therapy). False if pharmacotherapy is indicated."
        )
    )
    expected_empty_drug_list: bool = Field(
        description=(
            "True ONLY if no drug should be prescribed (e.g. lifestyle counselling, "
            "watchful waiting, referral-first scenarios). False otherwise."
        )
    )
    expected_slots: list[DrugSlot] = Field(
        default_factory=list,
        description=(
            "Drug slots — one per distinct therapeutic intervention. Empty list "
            "only when `expected_empty_drug_list=true`."
        ),
    )
    expected_referrals: list[str] = Field(
        default_factory=list,
        description="Specialist referrals if indicated (e.g. 'kardiologia', 'onkologia').",
    )
    expected_tests: list[str] = Field(
        default_factory=list,
        description="Diagnostic tests/imaging if indicated (e.g. 'EKG', 'TSH').",
    )


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


SYSTEM_PROMPT = """\
Jesteś klinicznym farmaceutą przygotowującym REFERENCJĘ GROUND-TRUTH dla
ewaluacji polskiego Systemu Wsparcia Decyzji Klinicznych (CDSS).

Dla każdego scenariusza wizyty lekarskiej:
1. Zidentyfikuj poprawne klinicznie leki PIERWSZEJ LINII zgodne z polskimi
   wytycznymi (PTK, PTG-E, PTD, PTN, NICE-equivalent w PL).
2. Pogrupuj leki w SLOTY — każdy slot reprezentuje JEDNĄ rolę terapeutyczną;
   ATC w obrębie slotu są klinicznymi równoważnikami (interchangeable).
3. Dla terapii skojarzonych (różne mechanizmy działania) — osobne sloty.

PRZYKŁADY GRUPOWANIA:
- GERD: 1 slot "inhibitor pompy protonowej" z acceptable: [A02BC01, A02BC02, A02BC04, A02BC05, A02BC03]
- Migotanie przedsionków: 2 sloty:
  - "DOAC (antykoagulacja)" z [B01AF01, B01AF02, B01AF03, B01AE07]
  - "beta-bloker (kontrola rytmu)" z [C07AB02, C07AB07, C07AB12]
- Graves: 2 sloty: tyreostatyk + beta-bloker
- POChP łagodne: 1 slot "wziewny bronchodilator" z opcjami SAMA/LAMA/SABA/LABA

REGUŁY ATC:
- Używaj 7-znakowych kodów ATC (np. C09AA05, NIE C09A).
- Dla leku często wszystkie generyczne równoważniki tej samej klasy — wymień je.
- Sortuj ATC alfanumerycznie w obrębie slotu.

CZERWONE FLAGI:
- expected_empty_drug_list=true TYLKO gdy: wizyta diagnostyczna z planem
  badań przed terapią ALBO interwencja lifestyle (waga/dieta/ćwiczenia)
  ALBO konsultacja specjalistyczna bez recepty.
- diagnostic_only=true gdy diagnoza jeszcze nie jest potwierdzona —
  potrzebne dalsze badania PRZED rozpoczęciem terapii.

CZERWONE FLAGI dla CHILDREN (pediatria <18r.ż.):
- Wiele leków przeciwwskazanych poniżej 12-18 r.ż. — zaproponuj alternatywy.
- Częste leki pediatryczne: amoksycylina (J01CA04), paracetamol (N02BE01),
  ibuprofen (M01AE01), montelukast (R03DC03), budesonid wziewny (R03BA02).

Język polski. Zwięźle, klinicznie.
"""


def render_user_prompt(scenario: dict) -> str:
    """Render a single scenario into a clinical brief for ground-truth generation."""
    turns_summary = []
    for i, t in enumerate(scenario.get("turns", [])):
        speaker = t["speaker"][:1].upper()
        text = t["text"]
        if len(text) > 120:
            text = text[:117] + "..."
        turns_summary.append(f"[{i}] {speaker}: {text}")
    turns_str = "\n".join(turns_summary[:50])  # cap context length

    return f"""\
SCENARIUSZ WIZYTY:
- ID: {scenario["id"]}
- Specjalizacja: {scenario["specialization"]}
- Wiek pacjenta: {scenario["patient_age"]}
- Płeć: {scenario["patient_sex"]}
- Chief complaint: {scenario["chief_complaint"]}
- Oczekiwana diagnoza: {scenario["expected_diagnosis"]}
- Kod ICD-10: {scenario.get("icd10_code", "—")}

DIALOG (skrócony):
{turns_str}

ZADANIE: Wygeneruj `GroundTruth` zgodny ze schematem, zawierający expected_slots,
expected_referrals, expected_tests. Pamiętaj o regułach systemu.
"""


# ---------------------------------------------------------------------------
# Generation (Vertex VertexClient with response_schema)
# ---------------------------------------------------------------------------


def _build_vertex_client():
    """Reuse the cloud module's VertexClient."""
    import sys

    cloud_src = str(REPO_ROOT / "application/cloud/src")
    if cloud_src not in sys.path:
        sys.path.insert(0, cloud_src)
    from cloud.llm_integration.llm_client import (
        TransientLLMError,
        VertexClient,
        VertexSettings,
    )

    settings = VertexSettings.from_env(model_override=DEFAULT_MODEL)
    return VertexClient(settings=settings), TransientLLMError


async def _generate_one(
    client,
    scenario: dict,
    *,
    sem: asyncio.Semaphore,
) -> tuple[str, GroundTruth | None, str | None]:
    """Generate ground truth for one scenario."""
    async with sem:
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                None,
                lambda: client.generate(
                    render_user_prompt(scenario),
                    response_schema=GroundTruth,
                    system_instruction=SYSTEM_PROMPT,
                ),
            )
            # The model returns a GroundTruth with scenario_id field set by LLM —
            # overwrite with the canonical short id from the input scenario.
            gt = result.model_copy(update={"scenario_id": scenario["id"][:8]})
            return scenario["id"], gt, None
        except Exception as exc:
            log.warning("scenario %s failed: %s", scenario["id"][:8], exc)
            return scenario["id"], None, str(exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    concurrency: int = typer.Option(
        8, "--concurrency", "-c", help="Concurrent Vertex requests."
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Re-generate even if GT file already exists."
    ),
    limit: int = typer.Option(
        0, "--limit", help="Max scenarios to process (0 = all)."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Generate ground-truth `expected_slots` for every scenario JSON."""
    load_dotenv(override=True)
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    GT_DIR.mkdir(parents=True, exist_ok=True)

    scenarios = []
    for p in sorted(SCENARIOS_DIR.glob("*.json")):
        try:
            scenarios.append(json.loads(p.read_text(encoding="utf-8")))
        except json.JSONDecodeError as exc:
            log.warning("skip malformed %s: %s", p.name, exc)

    pending: list[dict] = []
    skipped = 0
    for s in scenarios:
        gt_file = GT_DIR / f"{s['id'][:8]}.json"
        if gt_file.exists() and not force:
            skipped += 1
        else:
            pending.append(s)
    if limit > 0:
        pending = pending[:limit]

    console.print(
        f"[bold]Scenarios:[/bold] {len(scenarios)}  "
        f"[green]existing[/green]: {skipped}  "
        f"[yellow]pending[/yellow]: {len(pending)}"
    )
    if not pending:
        console.print("[green]All scenarios already have ground truth.[/green]")
        raise typer.Exit(code=0)

    client, _ = _build_vertex_client()
    sem = asyncio.Semaphore(concurrency)

    started = time.time()
    results: list[tuple[str, GroundTruth | None, str | None]] = []

    async def _run():
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            TextColumn("[bold]{task.completed}/{task.total}[/bold]"),
            TextColumn("([progress.percentage]{task.percentage:>3.0f}%)"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Generating GT (model={DEFAULT_MODEL}, conc={concurrency})",
                total=len(pending),
            )
            tasks = [
                asyncio.create_task(_generate_one(client, s, sem=sem))
                for s in pending
            ]
            for coro in asyncio.as_completed(tasks):
                sid, gt, err = await coro
                results.append((sid, gt, err))
                if gt is not None:
                    out_path = GT_DIR / f"{sid[:8]}.json"
                    out_path.write_text(
                        json.dumps(gt.model_dump(), indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                progress.advance(task)

    asyncio.run(_run())
    elapsed = time.time() - started

    successes = [r for r in results if r[1] is not None]
    failures = [r for r in results if r[1] is None]

    # Summary table
    table = Table(title="Generated ground truth (sample)", show_lines=False)
    table.add_column("sid")
    table.add_column("diagnosis")
    table.add_column("slots", justify="right")
    table.add_column("empty?", justify="center")
    table.add_column("referrals", justify="right")
    table.add_column("tests", justify="right")
    for sid, gt, _ in sorted(successes, key=lambda r: r[0]):
        if gt is None:
            continue
        empty = "yes" if gt.expected_empty_drug_list else ""
        diag = gt.diagnosis_label
        if len(diag) > 60:
            diag = diag[:57] + "..."
        table.add_row(
            sid[:8],
            diag,
            str(len(gt.expected_slots)),
            empty,
            str(len(gt.expected_referrals)),
            str(len(gt.expected_tests)),
        )
    console.print(table)

    console.print(
        f"\n[bold green]Done.[/bold green] Generated {len(successes)} "
        f"in {elapsed/60:.1f} min  (failures: {len(failures)})."
    )

    if failures:
        console.print("[red]Failures:[/red]")
        for sid, _, err in failures:
            console.print(f"  {sid[:8]}: {err}")


if __name__ == "__main__":
    app()
