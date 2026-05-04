"""CLI: NFZ refundacja XLSX → normalized JSON index.

Adapted to the actual `Obwieszczenie Ministra Zdrowia` XLSX format
(2024-2026 issues): 11 sheets (A1, A2, A3, B, C, D1, D2, E, F2, G1, G2),
header in row 2, Latin substance names, NO ATC column.

Strategy:

* **Substance → ATC mapping** is hand-curated in `substance_atc.json`
  next to this file. The wykaz uses Latin names (`Propranololi
  hydrochloridum`); the agent matches by ATC. Unmapped substances are
  skipped — Phase 3a only needs the ~30-50 most relevant drugs.
* **Reimbursement tier** comes from the sheet category:
    - A1: standardowe (calculate from cena_detaliczna - limit_finansowania)
    - A2: środki spożywcze (treat as standard)
    - A3: wyroby medyczne (treat as standard)
    - B:  programy lekowe (`Poziom odpłatności` column → parse)
    - C:  chemioterapia (full refund)
    - D1/D2: szczepionki / lecznictwo zamknięte (full)
    - E:  bezpłatne dla 75+ → full
    - F2: bezpłatne dla kobiet w ciąży → full
    - G1/G2: bezpłatne dla dzieci → full
* **Copay** computed from `cena_detaliczna - wysokość_limitu_finansowania`
  on A1; explicit `Wysokość dopłaty` elsewhere when present.

Output: `application/cloud/data/nfz_formulary/index_<YYYY-MM-DD>.json`
keyed by ATC; multiple wykaz rows per ATC collapse to first.

Usage::

    uv run python application/cloud/data/nfz_formulary/ingest_nfz.py \\
        --xlsx-path application/cloud/data/nfz_formulary/raw/2026-04-01.xlsx
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Iterable
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

import openpyxl
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from cloud.nfz.formulary_search import NFZEntry, serialize_index
from cloud.nfz.reimbursement import parse_reimbursement_string
from shared.schemas import ReimbursementLevel

app = typer.Typer(add_completion=False, help="MedVoicePL NFZ refundacja XLSX → JSON ingest")
console = Console()


DEFAULT_OUTPUT_DIR = Path("application/cloud/data/nfz_formulary")
DEFAULT_RAW_DIR = DEFAULT_OUTPUT_DIR / "raw"
DEFAULT_ATC_MAP_PATH = DEFAULT_OUTPUT_DIR / "substance_atc.json"

# Header row is consistently row 2 across every sheet.
_HEADER_ROW = 2

# Per-sheet "default refund tier" when the sheet doesn't have an explicit
# `Poziom odpłatności` column. Driven by the wykaz's category structure.
_SHEET_DEFAULT_TIER: dict[str, ReimbursementLevel] = {
    "A1": ReimbursementLevel.standard,  # calculated per-row from copay
    "A2": ReimbursementLevel.standard,
    "A3": ReimbursementLevel.standard,
    "B":  ReimbursementLevel.full,       # programy lekowe
    "C":  ReimbursementLevel.full,       # chemioterapia
    "D1": ReimbursementLevel.full,
    "D2": ReimbursementLevel.full,
    "E":  ReimbursementLevel.full,       # 75+
    "F2": ReimbursementLevel.full,       # ciąża
    "G1": ReimbursementLevel.full,       # dzieci
    "G2": ReimbursementLevel.full,
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
    atc_map_path: Path = typer.Option(
        DEFAULT_ATC_MAP_PATH,
        "--atc-map",
        help="Path to substance→ATC mapping JSON.",
    ),
    snapshot_date_str: str | None = typer.Option(
        None, "--snapshot-date", help="ISO date for the snapshot (defaults to today)."
    ),
    sheets: list[str] | None = typer.Option(
        None,
        "--sheet",
        help="Process only these sheets (repeatable). Default: all known.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Read NFZ XLSX → normalized JSON index."""
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

    if not atc_map_path.is_file():
        console.print(f"[red]substance→ATC map not found:[/red] {atc_map_path}")
        raise typer.Exit(code=1)
    atc_map = _load_atc_map(atc_map_path)
    console.print(f"[dim]substance→ATC entries: {len(atc_map)}[/dim]")

    console.print(f"[dim]parsing {xlsx_path}…[/dim]")
    with Progress(SpinnerColumn(), TextColumn("[bold]{task.description}")) as progress:
        progress.add_task("ingest", total=None)
        entries = list(
            _parse_workbook(xlsx_path, atc_map=atc_map, sheet_filter=sheets)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"index_{snapshot_date.isoformat()}.json"
    _atomic_write_json(out_path, serialize_index(snapshot_date, entries))

    console.print(
        f"[green]wrote {len(entries)} unique-ATC entries → {out_path}[/green]"
    )


# ---------------------------------------------------------------------------
# XLSX → NFZEntry stream
# ---------------------------------------------------------------------------


def _parse_workbook(
    path: Path,
    *,
    atc_map: dict[str, dict[str, str]],
    sheet_filter: list[str] | None,
) -> Iterable[NFZEntry]:
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    target_sheets = [
        s for s in workbook.sheetnames
        if s in _SHEET_DEFAULT_TIER and (not sheet_filter or s in sheet_filter)
    ]

    seen_atc: set[str] = set()
    seen_unmapped_substances: set[str] = set()

    for sheet_name in target_sheets:
        sheet = workbook[sheet_name]
        rows = list(sheet.iter_rows(values_only=True))
        if len(rows) < _HEADER_ROW + 1:
            continue
        header = rows[_HEADER_ROW - 1]
        cols = _index_columns(header)

        if "substance" not in cols:
            logging.warning("sheet %s: no 'Substancja czynna' column; skipping", sheet_name)
            continue

        default_tier = _SHEET_DEFAULT_TIER[sheet_name]
        sheet_emitted = 0
        sheet_unmapped = 0

        for row in rows[_HEADER_ROW:]:
            substance_raw = _cell(row, cols, "substance")
            if not substance_raw:
                continue
            substance_raw = substance_raw.strip()
            atc_entry = _resolve_atc(substance_raw, atc_map)
            if atc_entry is None:
                if substance_raw not in seen_unmapped_substances:
                    seen_unmapped_substances.add(substance_raw)
                    sheet_unmapped += 1
                continue

            atc_code = atc_entry["atc"]
            if atc_code in seen_atc:
                continue
            seen_atc.add(atc_code)

            display_name = atc_entry["display"]
            trade_name = _extract_trade_name(_cell(row, cols, "name_dosage"))
            indication = _cell(row, cols, "indication").strip()

            # Reimbursement tier: explicit column wins, else sheet default.
            tier_raw = _cell(row, cols, "reimbursement").strip()
            tier = (
                parse_reimbursement_string(tier_raw)
                if tier_raw
                else default_tier
            )

            copay = _compute_copay(row, cols)

            yield NFZEntry(
                atc_code=atc_code,
                substance=display_name,
                trade_names=((trade_name,) if trade_name else ()),
                reimbursement=tier,
                patient_copay_pln=copay,
                indication_restrictions=((indication,) if indication else ()),
            )
            sheet_emitted += 1

        logging.info(
            "sheet %s: emitted=%d unmapped_substances=%d",
            sheet_name, sheet_emitted, sheet_unmapped,
        )


def _index_columns(header: tuple) -> dict[str, int]:
    """Map our short keys to column indices via fuzzy substring match.

    The wykaz uses inconsistent column naming across sheets ("Nazwa
    postać i dawka" with double-space vs "Nazwa, postać i dawka leku");
    we match on lowercased substring.
    """
    fragments = {
        "substance": ["substancja czynna", "substancja"],
        "name_dosage": ["nazwa  postać i dawka", "nazwa postac", "nazwa i postać"],
        "package": ["zawartość opakowania"],
        "indication": ["zakres wskazań", "wskazań objętych refundacją"],
        "reimbursement": ["poziom odpłatności"],
        "price_retail": ["cena detaliczna"],
        "limit_finansowania": ["wysokość limitu finansowania", "limit finansowania"],
        "copay_explicit": ["wysokość dopłaty"],
    }
    norm = [(str(c).casefold().strip() if c else "") for c in header]
    out: dict[str, int] = {}
    for key, frags in fragments.items():
        for idx, cell in enumerate(norm):
            if any(frag in cell for frag in frags):
                out[key] = idx
                break
    return out


def _cell(row: tuple, cols: dict[str, int], key: str) -> str:
    idx = cols.get(key)
    if idx is None or idx >= len(row):
        return ""
    value = row[idx]
    return "" if value is None else str(value)


def _resolve_atc(substance_raw: str, atc_map: dict[str, dict[str, str]]) -> dict[str, str] | None:
    """Match wykaz substance string to our ATC map.

    Wykaz substances are Latin (`Propranololi hydrochloridum`); map keys
    use the same Latin form. Some entries have combined forms ("X + Y")
    or different word order — we try the exact key, then a normalized
    head-of-string lookup.
    """
    if substance_raw in atc_map:
        return atc_map[substance_raw]
    # Case-insensitive fallback.
    normalized = substance_raw.strip().casefold()
    for key, value in atc_map.items():
        if key.casefold() == normalized:
            return value
    return None


def _compute_copay(row: tuple, cols: dict[str, int]) -> Decimal | None:
    """Either explicit `Wysokość dopłaty` column, or
    `cena_detaliczna - wysokość_limitu_finansowania` (A1-style).
    """
    explicit = _cell(row, cols, "copay_explicit")
    if explicit:
        parsed = _parse_decimal_pl(explicit)
        if parsed is not None:
            return parsed

    retail = _parse_decimal_pl(_cell(row, cols, "price_retail"))
    limit = _parse_decimal_pl(_cell(row, cols, "limit_finansowania"))
    if retail is not None and limit is not None:
        diff = retail - limit
        return diff if diff >= 0 else Decimal("0")
    return None


def _parse_decimal_pl(raw: str) -> Decimal | None:
    if not raw:
        return None
    cleaned = raw.replace(" ", "").replace(",", ".")
    try:
        return Decimal(cleaned)
    except (InvalidOperation, ValueError):
        return None


def _extract_trade_name(name_dosage: str) -> str:
    if not name_dosage:
        return ""
    head = name_dosage.split(",", maxsplit=1)[0].strip()
    head = re.sub(r"\s+(tabl|kaps|amp|inj|płyn|aer|krop|maść|żel)\..*$", "", head, flags=re.IGNORECASE)
    return head.strip()


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def _resolve_xlsx_path(*, url: str | None, snapshot_date: date) -> Path:
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
    import httpx

    console.print(f"[dim]downloading {url} → {dest}[/dim]")
    with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as resp:
        resp.raise_for_status()
        tmp = dest.with_suffix(".xlsx.tmp")
        with tmp.open("wb") as fh:
            for chunk in resp.iter_bytes():
                fh.write(chunk)
        os.replace(tmp, dest)


def _load_atc_map(path: Path) -> dict[str, dict[str, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


if __name__ == "__main__":  # pragma: no cover
    app()
