"""CLI entrypoint: SOAPNote JSON -> AnonymizedSOAPNote JSON via Presidio.

Usage:
    # smoke: one scenario
    uv run python application/on_premise/scripts/run_anonymization.py --limit 1

    # target scenario, force re-run (skip cache)
    uv run python application/on_premise/scripts/run_anonymization.py \
        --scenario-id a9b1c2d3 --force

    # full corpus (Phase 2c gate)
    uv run python application/on_premise/scripts/run_anonymization.py

Reads SOAP notes from `application/on_premise/output/soap_notes/<sid>/`
(Phase 2b output, gitignored) and writes anonymized notes +
audit logs:

* `application/on_premise/output/anonymized/<sid>/anonymized_soap.json`
* `application/on_premise/audit/<YYYY-MM-DD>/<session_id>/audit.json`
* `application/on_premise/audit/<YYYY-MM-DD>/<session_id>/token_map.json` (chmod 0600)
* `application/on_premise/audit/<YYYY-MM-DD>/<session_id>/trust_gate.json`

Prints a Rich table:

    | scenario | n_PERSON | n_PESEL | n_PHONE | n_LOC | gate_pass | duration |
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import date, timezone
from pathlib import Path
from uuid import uuid4

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from on_premise.anonymization.audit_logger import AuditLogger
from on_premise.anonymization.presidio_agent import (
    PresidioAnonymizer,
    default_registry,
)
from on_premise.structuring.soap_schema import SOAPNote, SOAPNoteSubmission
from on_premise.trust_gate import (
    TrustBoundaryViolation,
    validate_outbound,
    violations_to_dicts,
)

app = typer.Typer(add_completion=False, help="MedVoicePL on-premise anonymization CLI")
console = Console()

DEFAULT_SOAP_DIR = Path("application/on_premise/output/soap_notes")
DEFAULT_ANONYMIZED_DIR = Path("application/on_premise/output/anonymized")
DEFAULT_AUDIT_DIR = Path("application/on_premise/audit")
DEFAULT_SCENARIOS_DIR = Path("dataset_pipeline/output/scenarios")


@app.command()
def anonymize(
    soap_dir: Path = typer.Option(DEFAULT_SOAP_DIR, "--soap-dir"),
    anonymized_dir: Path = typer.Option(DEFAULT_ANONYMIZED_DIR, "--anonymized-dir"),
    audit_dir: Path = typer.Option(DEFAULT_AUDIT_DIR, "--audit-dir"),
    scenarios_dir: Path = typer.Option(
        DEFAULT_SCENARIOS_DIR,
        "--scenarios-dir",
        help="Where to read patient_full_name from for the spaCy-NER booster.",
    ),
    scenario_id: list[str] | None = typer.Option(
        None,
        "--scenario-id",
        "-s",
        help="Only process scenario directories whose name contains this prefix (repeatable).",
    ),
    limit: int | None = typer.Option(None, "--limit", "-n"),
    force: bool = typer.Option(
        False,
        "--force",
        help="Re-anonymize even if anonymized_soap.json already exists.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run Phase 2c anonymization on every SOAP note in `--soap-dir`."""
    load_dotenv()
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger(__name__)

    if not soap_dir.is_dir():
        console.print(f"[red]SOAP dir not found:[/red] {soap_dir}")
        raise typer.Exit(code=1)

    candidates = _collect_candidates(soap_dir, scenario_id, limit)
    if not candidates:
        console.print(f"[yellow]No SOAP notes matched filters in {soap_dir}[/yellow]")
        raise typer.Exit(code=0)

    anonymizer_version = os.environ.get("ANONYMIZER_VERSION", "2c.0")
    controller = os.environ.get("ANONYMIZER_CONTROLLER", "MedVoicePL")
    retention_days = int(os.environ.get("AUDIT_RETENTION_DAYS", "730"))

    # Build the analyzer + audit logger once. The first call to analyze()
    # triggers the spaCy load (~3 s); subsequent calls reuse the cached
    # AnalyzerEngine.
    console.print(
        f"[dim]building Presidio analyzer "
        f"(seeded from {scenarios_dir if scenarios_dir.is_dir() else 'no scenarios'})…[/dim]"
    )
    registry = default_registry(scenarios_dir=scenarios_dir if scenarios_dir.is_dir() else None)
    from on_premise.anonymization.eponym_filter import load_default
    eponym_filter = load_default()
    anonymizer = PresidioAnonymizer(registry=registry, eponym_filter=eponym_filter)
    console.print(
        f"[dim]eponym filter: {len(eponym_filter.eponyms)} terms[/dim]"
    )
    audit = AuditLogger(
        audit_root=audit_dir, retention_days=retention_days, controller=controller
    )

    rows: list[dict[str, str | int | bool | float]] = []
    for sid, soap_path in candidates:
        row = _process(
            sid=sid,
            soap_path=soap_path,
            anonymized_dir=anonymized_dir,
            anonymizer=anonymizer,
            audit=audit,
            anonymizer_version=anonymizer_version,
            force=force,
        )
        rows.append(row)
        if verbose:
            log.info("processed %s: %s", sid, row)

    _print_table(rows)
    n_failed = sum(1 for r in rows if not r["gate_pass"])
    if n_failed:
        console.print(f"[red]{n_failed}/{len(rows)} scenarios failed the trust gate[/red]")
        raise typer.Exit(code=2)
    console.print(f"[green]{len(rows)} scenarios anonymized + gate-passed[/green]")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_candidates(
    soap_dir: Path, scenario_id: list[str] | None, limit: int | None
) -> list[tuple[str, Path]]:
    """Find every `<sid>/soap_note.json` under `soap_dir`.

    Returns sorted list of `(scenario_id, soap_path)` tuples after
    applying `--scenario-id` filter and `--limit`.
    """
    out: list[tuple[str, Path]] = []
    for sub in sorted(soap_dir.iterdir()):
        if not sub.is_dir():
            continue
        soap_path = sub / "soap_note.json"
        if not soap_path.is_file():
            continue
        if scenario_id and not any(s in sub.name for s in scenario_id):
            continue
        out.append((sub.name, soap_path))
    if limit is not None:
        out = out[:limit]
    return out


def _process(
    *,
    sid: str,
    soap_path: Path,
    anonymized_dir: Path,
    anonymizer: PresidioAnonymizer,
    audit: AuditLogger,
    anonymizer_version: str,
    force: bool,
) -> dict[str, str | int | bool | float]:
    """Run anonymizer + trust gate on one SOAP note. Returns a row dict
    for the summary table.
    """
    out_path = anonymized_dir / sid / "anonymized_soap.json"
    if out_path.is_file() and not force:
        # Cached — re-load and skip the SLM/spaCy work. Still validates
        # via the trust gate (cheap) so we report current state, not
        # state-as-of-write.
        from shared.schemas import AnonymizedSOAPNote

        cached = AnonymizedSOAPNote.model_validate_json(out_path.read_text())
        try:
            validate_outbound(cached)
            return _row(sid, cached, gate_pass=True, duration=0.0, cached=True)
        except TrustBoundaryViolation as exc:
            return _row_failed(sid, gate_pass=False, duration=0.0, error=str(exc), cached=True)

    # Load the SOAP, strip metadata to get a SOAPNoteSubmission.
    data = json.loads(soap_path.read_text())
    soap = SOAPNote.model_validate(data)
    submission = SOAPNoteSubmission.model_validate(
        {k: v for k, v in data.items() if k != "metadata"}
    )

    # Specialization comes from the SOAP metadata's scenario_id prefix
    # — for synthetic data this matches the directory name pattern;
    # for real data the gateway will pass it explicitly.
    specialization = _guess_specialization(soap, sid)

    session_id = uuid4()
    t0 = time.monotonic()
    result = anonymizer.anonymize_soap(
        submission,
        session_id=session_id,
        visit_date=date.today(),
        specialization=specialization,
        anonymizer_version=anonymizer_version,
    )
    duration = time.monotonic() - t0

    # Trust gate.
    try:
        validate_outbound(result.note)
        gate_pass = True
        violations: list[dict[str, str]] = []
    except TrustBoundaryViolation as exc:
        gate_pass = False
        violations = violations_to_dicts(exc.violations)

    # Persist anonymized output (regardless of gate result — we want
    # the artifact for forensic review).
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result.note.model_dump_json(indent=2), encoding="utf-8")

    # Audit log.
    audit.log_anonymization(
        session_id=str(session_id),
        replacements=result.replacements,
        soap_sha256=_sha_of_json(result.note.model_dump_json()),
        transcript_sha256=soap.metadata.source_audio_sha256,
    )
    audit.write_token_map(session_id=str(session_id), token_map=result.token_map)
    audit.log_trust_gate(
        session_id=str(session_id),
        decision="pass" if gate_pass else "fail",
        violations=violations or None,
    )

    if gate_pass:
        return _row(sid, result.note, gate_pass=True, duration=duration, cached=False)
    return _row_failed(
        sid,
        gate_pass=False,
        duration=duration,
        error="; ".join(f"{v['kind']}:{v['detail']}" for v in violations[:3]),
        cached=False,
    )


def _row(
    sid: str,
    note: object,
    *,
    gate_pass: bool,
    duration: float,
    cached: bool,
) -> dict[str, str | int | bool | float]:
    counts = note.metadata.n_entities_per_type  # type: ignore[attr-defined]
    return {
        "scenario": sid,
        "n_PERSON": counts.get("PERSON", 0),
        "n_PESEL": counts.get("PL_PESEL", 0),
        "n_PHONE": counts.get("PHONE_NUMBER", 0),
        "n_LOC": counts.get("LOCATION", 0),
        "n_other": sum(
            v
            for k, v in counts.items()
            if k not in ("PERSON", "PL_PESEL", "PHONE_NUMBER", "LOCATION")
        ),
        "gate_pass": gate_pass,
        "duration_s": round(duration, 2),
        "cached": cached,
    }


def _row_failed(
    sid: str, *, gate_pass: bool, duration: float, error: str, cached: bool
) -> dict[str, str | int | bool | float]:
    return {
        "scenario": sid,
        "n_PERSON": 0,
        "n_PESEL": 0,
        "n_PHONE": 0,
        "n_LOC": 0,
        "n_other": 0,
        "gate_pass": gate_pass,
        "duration_s": round(duration, 2),
        "cached": cached,
        "error": error,
    }


def _guess_specialization(soap: SOAPNote, sid: str) -> str:
    """Infer specialization from the directory name (`endokrynologia_*`,
    `kardiologia_*`, etc.) or default to `medycyna_rodzinna`.

    Phase 2c CLI is for synthetic data only; real data flows through
    the FastAPI endpoint with explicit `specialization` query param.
    """
    name_to_spec = {
        "kardio": "kardiologia",
        "pulmo": "pulmonologia",
        "neuro": "neurologia",
        "endo": "endokrynologia",
        "synth-mr": "medycyna_rodzinna",
    }
    for prefix, spec in name_to_spec.items():
        if sid.lower().startswith(prefix) or prefix in sid.lower():
            return spec
    # Heuristic fallback — first segment of the directory name.
    return "medycyna_rodzinna"


def _sha_of_json(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _print_table(rows: list[dict[str, str | int | bool | float]]) -> None:
    """Render the summary as a Rich table."""
    table = Table(title="Anonymization summary")
    for col in (
        "scenario",
        "n_PERSON",
        "n_PESEL",
        "n_PHONE",
        "n_LOC",
        "n_other",
        "gate_pass",
        "duration_s",
        "cached",
    ):
        table.add_column(col, justify="right" if col != "scenario" else "left")
    for row in rows:
        table.add_row(
            str(row["scenario"]),
            str(row["n_PERSON"]),
            str(row["n_PESEL"]),
            str(row["n_PHONE"]),
            str(row["n_LOC"]),
            str(row["n_other"]),
            "[green]✓[/green]" if row["gate_pass"] else "[red]✗[/red]",
            f"{row['duration_s']}s",
            "[dim]cached[/dim]" if row.get("cached") else "",
        )
    console.print(table)
    # Print first error inline below the table for failed runs.
    for row in rows:
        if not row["gate_pass"] and "error" in row:
            console.print(f"  [red]{row['scenario']}[/red]: {row['error']}")


# Keep the timezone import used implicitly via date.today() (timezone-aware
# audit timestamps come from audit_logger). Unused here directly — but
# kept to minimize the diff against the in-progress run.
_ = timezone


if __name__ == "__main__":  # pragma: no cover
    app()
