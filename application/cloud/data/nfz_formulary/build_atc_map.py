"""CLI: bootstrap full substance→ATC mapping from the NFZ wykaz via Gemini batch.

The hand-curated `substance_atc.json` covers ~96 drugs. The actual NFZ
wykaz has ~1500-3000 unique substance names. This script:

1. Parses every sheet of the NFZ XLSX, collects all unique
   `Substancja czynna` strings.
2. Filters out substances we already have mapped.
3. Batches the rest (default 80 per call) to Gemini with a
   `response_schema=AtcBatch` for deterministic JSON output.
4. Merges the resulting map back into `substance_atc.json` (atomic write).

One-time cost: ~$0.30-0.50 on `gemini-2.5-pro` for the full wykaz.
Cheaper: pass `--model gemini-2.5-flash` (~$0.05).

Usage::

    # Dry run — count substances that would be enriched
    uv run python application/cloud/data/nfz_formulary/build_atc_map.py \\
        --xlsx-path application/cloud/data/nfz_formulary/raw/2026-04-01.xlsx \\
        --dry-run

    # Real run (uses VERTEX_AGENT_MODEL from .env)
    uv run python application/cloud/data/nfz_formulary/build_atc_map.py \\
        --xlsx-path application/cloud/data/nfz_formulary/raw/2026-04-01.xlsx

    # Cheap run with limit (test first)
    uv run python application/cloud/data/nfz_formulary/build_atc_map.py \\
        --xlsx-path application/cloud/data/nfz_formulary/raw/2026-04-01.xlsx \\
        --model gemini-2.5-flash --max-batches 2
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import openpyxl
import typer
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

app = typer.Typer(add_completion=False, help="Bootstrap NFZ substance→ATC map via Gemini")
console = Console()


# Pydantic schemas for Gemini structured output.
class AtcEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    substance_latin: str = Field(description="Original Latin substance name as given.")
    atc_code: str = Field(
        pattern=r"^[A-Z]\d{2}[A-Z]{2}\d{2}$|^UNKNOWN$",
        description="7-char ATC code, or 'UNKNOWN' if no canonical ATC exists.",
    )
    display_pl: str = Field(
        description="Polish display name (single substance, no formulation)."
    )


class AtcBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    entries: list[AtcEntry] = Field(default_factory=list)


_BATCH_PROMPT = """\
Dla każdej z poniższych łacińskich nazw substancji czynnych z polskiego \
wykazu NFZ podaj kanoniczny 7-znakowy kod ATC (WHO ATC/DDD) oraz polską \
nazwę zwyczajową.

ZASADY:
1. Kod ATC musi być w formacie 7 znaków: jedna wielka litera, dwie cyfry, \
   dwie wielkie litery, dwie cyfry (np. `C07AB07`, `B01AF02`).
2. Jeśli substancja jest preparatem złożonym (np. "X + Y"), podaj ATC \
   właściwego preparatu złożonego (np. dla "Amlodipinum + Valsartanum" \
   to `C09DB01`, NIE ATC pojedynczego składnika).
3. Jeśli to dieta lecznicza / wyrób medyczny / surowiec bez ATC, zwróć `UNKNOWN`.
4. Polska nazwa: podstawowa nazwa substancji w mianowniku (np. "Amlodypina").
5. Jeden wpis na każdą substancję wejściową, w tej samej kolejności.

SUBSTANCJE:
{substances}
"""


@app.command()
def build(
    xlsx_path: Path = typer.Option(
        ...,
        "--xlsx-path",
        help="Path to the NFZ wykaz XLSX (uses raw/<date>.xlsx).",
    ),
    map_path: Path = typer.Option(
        Path("application/cloud/data/nfz_formulary/substance_atc.json"),
        "--map-path",
        help="Existing substance_atc.json to merge into.",
    ),
    batch_size: int = typer.Option(80, "--batch-size", help="Substances per Gemini call."),
    max_batches: int | None = typer.Option(
        None, "--max-batches", help="Limit batches (for test runs)."
    ),
    model_override: str | None = typer.Option(
        None, "--model", help="Override VERTEX_AGENT_MODEL (e.g. gemini-2.5-flash)."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Count substances without calling Gemini."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Build a comprehensive substance→ATC map by enriching the wykaz via Gemini."""
    load_dotenv()
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not xlsx_path.is_file():
        console.print(f"[red]XLSX not found:[/red] {xlsx_path}")
        raise typer.Exit(code=1)

    existing_map = _load_existing_map(map_path)
    console.print(f"[dim]Existing map: {len(existing_map)} substances[/dim]")

    all_substances = _extract_unique_substances(xlsx_path)
    console.print(f"[dim]Wykaz unique substances: {len(all_substances)}[/dim]")

    # Filter out what we already have (case-insensitive).
    existing_lower = {k.casefold() for k in existing_map}
    todo = sorted({s for s in all_substances if s.casefold() not in existing_lower})
    console.print(
        f"[dim]To enrich: {len(todo)} substances "
        f"(skipped {len(all_substances) - len(todo)} already mapped)[/dim]"
    )

    if not todo:
        console.print("[green]Nothing to do — map is already complete.[/green]")
        return

    n_batches = (len(todo) + batch_size - 1) // batch_size
    if max_batches is not None:
        n_batches = min(n_batches, max_batches)
        todo = todo[: n_batches * batch_size]
    console.print(
        f"[dim]Will issue {n_batches} batch(es) of up to {batch_size} substances each[/dim]"
    )

    if dry_run:
        console.print("[yellow]Dry run — exiting without calling Gemini.[/yellow]")
        # Show a sample of substances we'd query.
        sample = todo[:10]
        console.print(f"[dim]Sample of unmapped substances:[/dim]")
        for s in sample:
            console.print(f"  • {s}")
        return

    # Build LLM client (uses VERTEX_AGENT_MODEL or override).
    from cloud.llm_integration.llm_client import (
        PermanentLLMError,
        TransientLLMError,
        build_llm_client,
    )

    client = build_llm_client(model_override=model_override)
    console.print(f"[dim]LLM: {client.name}[/dim]")

    new_entries: dict[str, dict[str, str]] = {}
    failed: list[str] = []

    with Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("enriching", total=n_batches)
        for i in range(n_batches):
            batch = todo[i * batch_size : (i + 1) * batch_size]
            try:
                result = _enrich_batch(client, batch)
            except (TransientLLMError, PermanentLLMError) as exc:
                logging.error("batch %d failed: %s", i, exc)
                failed.extend(batch)
                progress.advance(task)
                continue
            for entry in result.entries:
                if entry.atc_code == "UNKNOWN":
                    continue
                key = entry.substance_latin.strip()
                new_entries[key] = {
                    "atc": entry.atc_code,
                    "display": entry.display_pl.strip(),
                }
            progress.advance(task)

    console.print(
        f"[green]New mappings: {len(new_entries)}[/green] "
        f"[yellow]Failed/UNKNOWN: {len(todo) - len(new_entries)}[/yellow]"
    )

    # Merge + write atomically.
    merged = {**existing_map, **new_entries}
    _atomic_write_json(map_path, merged)
    console.print(
        f"[green]wrote {len(merged)} total entries → {map_path}[/green] "
        f"(was {len(existing_map)}, added {len(new_entries)})"
    )


def _enrich_batch(client, substances: list[str]) -> AtcBatch:  # type: ignore[no-untyped-def]
    """Call Gemini with a batch of substance names; expect AtcBatch JSON back."""
    listing = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(substances))
    prompt = _BATCH_PROMPT.format(substances=listing)
    result = client.generate(prompt, response_schema=AtcBatch)
    if not isinstance(result, AtcBatch):
        raise RuntimeError(f"unexpected response type: {type(result).__name__}")
    return result


def _load_existing_map(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def _extract_unique_substances(xlsx_path: Path) -> list[str]:
    """Walk every sheet of the wykaz, collect non-empty `Substancja czynna`."""
    workbook = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    out: set[str] = set()
    for sheet_name in workbook.sheetnames:
        sheet = workbook[sheet_name]
        rows = list(sheet.iter_rows(values_only=True))
        if len(rows) < 3:
            continue
        # Header row 2; substance column index varies but is usually col B (idx 1).
        header = rows[1]
        sub_idx = _find_substance_col(header)
        if sub_idx is None:
            continue
        for row in rows[2:]:
            if sub_idx >= len(row):
                continue
            value = row[sub_idx]
            if value is None:
                continue
            text = str(value).strip()
            if not text or text.casefold() == "substancja czynna":
                continue
            out.add(text)
    return sorted(out)


def _find_substance_col(header: tuple) -> int | None:
    for idx, cell in enumerate(header):
        if cell is None:
            continue
        norm = str(cell).casefold().strip()
        if "substancja" in norm:
            return idx
    return None


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


if __name__ == "__main__":  # pragma: no cover
    app()
