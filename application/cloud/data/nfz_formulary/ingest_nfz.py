"""CLI: download NFZ refundacja XLSX → normalized JSON index.

Usage::

    # offline (use already-downloaded XLSX)
    uv run python application/cloud/data/nfz_formulary/ingest_nfz.py \\
        --xlsx-path application/cloud/data/nfz_formulary/raw/2026-04-01.xlsx

    # online — uses NFZ_XLSX_URL env (or --url override)
    uv run python application/cloud/data/nfz_formulary/ingest_nfz.py

Reads the bimonthly NFZ list (Obwieszczenie Ministra Zdrowia w sprawie
wykazu refundowanych leków...) and produces a normalized JSON index at
``data/nfz_formulary/index_<YYYY-MM-DD>.json`` (atomic write).

The XLSX format has been stable across 2024-2026 issues — columns:

* Lp.
* Nazwa, postać i dawka (`{trade name}, {dosage form}, {dose}`)
* Substancja czynna
* Kod EAN
* Kod ATC
* Wskazania
* Poziom odpłatności
* Cena hurtowa brutto / detaliczna
* Wysokość dopłaty świadczeniobiorcy
* ... (~10 columns total)

We only need columns: ATC, substance, trade-name (parsed from "Nazwa,
postać i dawka"), reimbursement level (parsed from "Poziom odpłatności"),
patient copay, indication restrictions.

Snapshot date: parsed from the obwieszczenie metadata if available, else
defaults to today.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Iterable
from datetime import date
from decimal import Decimal
from pathlib import Path

import openpyxl
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from cloud.nfz.formulary_search import NFZEntry, serialize_index
from cloud.nfz.reimbursement import parse_copay_pln, parse_reimbursement_string

app = typer.Typer(add_completion=False, help="MedVoicePL NFZ refundacja XLSX → JSON ingest")
console = Console()


DEFAULT_OUTPUT_DIR = Path("application/cloud/data/nfz_formulary")
DEFAULT_RAW_DIR = DEFAULT_OUTPUT_DIR / "raw"


# Column-name fragments we look for in the header row. The XLSX uses
# Polish; matching by substring (after casefold) tolerates minor
# wording shifts between releases.
_COLUMN_FRAGMENTS = {
    "atc_code": ["kod atc", "atc"],
    "name_dosage": ["nazwa, postać i dawka", "nazwa postac"],
    "substance": ["substancja czynna", "substancja"],
    "indication": ["wskazania"],
    "reimbursement": ["poziom odpłatności", "odplatnosc"],
    "copay": ["dopłaty świadczeniobiorcy", "doplaty", "wysokosc doplaty"],
}


@app.command()
def ingest(
    xlsx_path: Path | None = typer.Option(
        None, "--xlsx-path", help="Path to the NFZ XLSX. If omitted, uses NFZ_XLSX_URL."
    ),
    url: str | None = typer.Option(
        None, "--url", help="Override NFZ_XLSX_URL for this run."
    ),
    output_dir: Path = typer.Option(
        DEFAULT_OUTPUT_DIR, "--output-dir", help="Where to write index_<date>.json."
    ),
    snapshot_date_str: str | None = typer.Option(
        None,
        "--snapshot-date",
        help="ISO date for the snapshot (defaults to today).",
    ),
    sheet_name: str | None = typer.Option(
        None, "--sheet", help="Worksheet name override; defaults to first sheet."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Download (or read locally) the NFZ XLSX and emit a JSON index."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    snapshot_date = (
        date.fromisoformat(snapshot_date_str) if snapshot_date_str else date.today()
    )

    if xlsx_path is None:
        xlsx_path = _resolve_xlsx_path(url=url, snapshot_date=snapshot_date)

    if not xlsx_path.is_file():
        console.print(f"[red]XLSX not found:[/red] {xlsx_path}")
        raise typer.Exit(code=1)

    console.print(f"[dim]parsing {xlsx_path}…[/dim]")
    with Progress(SpinnerColumn(), TextColumn("[bold]{task.description}")) as progress:
        progress.add_task("ingest", total=None)
        entries = list(_parse_xlsx(xlsx_path, sheet_name=sheet_name))

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"index_{snapshot_date.isoformat()}.json"
    _atomic_write_json(out_path, serialize_index(snapshot_date, entries))

    console.print(
        f"[green]wrote {len(entries)} entries → {out_path}[/green]"
    )


# ---------------------------------------------------------------------------
# XLSX parsing
# ---------------------------------------------------------------------------


def _parse_xlsx(path: Path, *, sheet_name: str | None = None) -> Iterable[NFZEntry]:
    """Iterate the XLSX rows, yielding `NFZEntry` per refunded drug.

    Header detection is forgiving: we scan the first 5 rows for one
    that contains every expected column-name fragment. The header row
    index sets the data-row start.
    """
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheet = workbook[sheet_name] if sheet_name else workbook[workbook.sheetnames[0]]

    header_row, columns = _find_header(sheet)
    if header_row is None or columns is None:
        raise RuntimeError(
            "could not locate header row — XLSX format may have changed; "
            "inspect manually + adjust _COLUMN_FRAGMENTS"
        )

    seen_atc: set[str] = set()
    for row in sheet.iter_rows(min_row=header_row + 1, values_only=True):
        atc_raw = row[columns["atc_code"]] if columns["atc_code"] is not None else None
        if atc_raw is None:
            continue
        atc_code = str(atc_raw).strip().upper()
        if not atc_code or atc_code == "ATC":  # skip footer/repeated-header rows
            continue
        # Same ATC across many package sizes — keep only the first.
        if atc_code in seen_atc:
            continue
        seen_atc.add(atc_code)

        name_dosage = _cell(row, columns, "name_dosage")
        substance = _cell(row, columns, "substance")
        reimbursement_raw = _cell(row, columns, "reimbursement")
        copay_raw = _cell(row, columns, "copay")
        indication = _cell(row, columns, "indication")

        trade_name = _extract_trade_name(name_dosage)
        copay: Decimal | None = parse_copay_pln(reimbursement_raw) or parse_copay_pln(copay_raw)
        restrictions: tuple[str, ...] = (
            (indication.strip(),) if indication and indication.strip() else ()
        )

        if not substance.strip():
            continue  # malformed row — substance is required for downstream lookup

        yield NFZEntry(
            atc_code=atc_code,
            substance=substance.strip(),
            trade_names=((trade_name,) if trade_name else ()),
            reimbursement=parse_reimbursement_string(reimbursement_raw),
            patient_copay_pln=copay,
            indication_restrictions=restrictions,
        )


def _find_header(sheet) -> tuple[int | None, dict[str, int | None] | None]:
    """Locate the header row (1-indexed) and column-index mapping."""
    for header_row, row in enumerate(sheet.iter_rows(values_only=True), start=1):
        if header_row > 5:
            break
        normalized = [(str(c).casefold().strip() if c else "") for c in row]
        cols: dict[str, int | None] = dict.fromkeys(_COLUMN_FRAGMENTS, None)
        matched = 0
        for key, fragments in _COLUMN_FRAGMENTS.items():
            for idx, cell in enumerate(normalized):
                if any(frag in cell for frag in fragments):
                    cols[key] = idx
                    matched += 1
                    break
        # Require ATC + substance + reimbursement at minimum.
        if cols["atc_code"] is not None and cols["substance"] is not None and cols["reimbursement"] is not None:
            return header_row, cols
    return None, None


def _cell(row: tuple, columns: dict[str, int | None], key: str) -> str:
    idx = columns.get(key)
    if idx is None:
        return ""
    value = row[idx]
    if value is None:
        return ""
    return str(value)


def _extract_trade_name(name_dosage: str) -> str:
    """Extract just the trade name from the "Nazwa, postać i dawka" cell.

    XLSX cell typically looks like::

        "Bisocard, tabl. powlekane, 5 mg"

    Trade name is everything before the first comma. Strip leading
    capitalisation noise.
    """
    if not name_dosage:
        return ""
    head = name_dosage.split(",", maxsplit=1)[0].strip()
    # Drop dosage/strength suffix that snuck in without a comma
    # (some legacy issues use "tabl. 5 mg" inline).
    head = re.sub(r"\s+(tabl|kaps|amp|inj)\..*$", "", head, flags=re.IGNORECASE)
    return head.strip()


# ---------------------------------------------------------------------------
# Path / IO helpers
# ---------------------------------------------------------------------------


def _resolve_xlsx_path(*, url: str | None, snapshot_date: date) -> Path:
    """Pick an XLSX path: explicit URL → download; else expected raw path."""
    raw_dir = DEFAULT_RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    expected = raw_dir / f"{snapshot_date.isoformat()}.xlsx"
    if expected.is_file():
        return expected

    download_url = url or os.environ.get("NFZ_XLSX_URL")
    if not download_url:
        console.print(
            "[yellow]No --xlsx-path given and NFZ_XLSX_URL is empty.[/yellow] "
            f"Place the XLSX at {expected} or pass --xlsx-path."
        )
        raise typer.Exit(code=1)

    _download(download_url, expected)
    return expected


def _download(url: str, dest: Path) -> None:
    """Stream-download the XLSX to `dest`. ``httpx`` is already in deps."""
    import httpx

    console.print(f"[dim]downloading {url} → {dest}[/dim]")
    with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as resp:
        resp.raise_for_status()
        tmp = dest.with_suffix(".xlsx.tmp")
        with tmp.open("wb") as fh:
            for chunk in resp.iter_bytes():
                fh.write(chunk)
        os.replace(tmp, dest)


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Same atomic-write idiom as on-prem audit logger."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


if __name__ == "__main__":  # pragma: no cover
    app()
