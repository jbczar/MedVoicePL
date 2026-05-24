"""Phase 4 — Bulk ChPL excerpt generation via Vertex Gemini with grounding.

Generates ~600 Polish ChPL summary markdown files for every drug in the NFZ
formulary index (~619 entries). Each excerpt has the same structure as the
14 hand-written ones in this directory: short intro + sections
(Wskazania / Dawkowanie / Przeciwwskazania / Ostrzeżenia / Refundacja NFZ).

Key design points:

* **Grounding via Google Search Retrieval** (URPL.gov.pl, ChPL.pl) attached
  as a Vertex `tools=[google_search]` configuration. First 1500 grounded
  queries per day are free; 619 entries fit comfortably under that cap.
* **No `response_schema`**: grounding and structured-output are mutually
  exclusive on current Vertex API. We get free-form markdown back and
  prepend a YAML frontmatter on the script side with `atc`, `substance`,
  `trade_names`, `source_urls`, `generated_at`.
* **Idempotent**: skips any ATC that already has a `<atc>_*.md` file in
  this directory. Existing 14 hand-written files are preserved.
* **Concurrency 8** with `asyncio.Semaphore` + `asyncio.gather`. Each task
  wraps the sync `client.models.generate_content` call in `asyncio.to_thread`
  (google-genai's async support is patchy across versions).
* **Tenacity retry** 4x with exponential backoff on 5xx/429.
* **Atomic writes**: each `.md` flushes as soon as Gemini returns, so a
  crash at item 400/619 leaves 400 valid files on disk.
* **Manifest**: `chpl_manifest.json` records every successful generation
  (atc, file, generated_at, sources, sha256). Failures captured in
  `chpl_failures.json`.

Run::

    uv run python application/cloud/data/chpl_excerpts/generate_chpl.py \\
        --concurrency 8 --limit 619
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

console = Console()
log = logging.getLogger("generate_chpl")
app = typer.Typer(add_completion=False, help="Bulk-generate ChPL excerpts via Vertex Gemini.")

# ---------------------------------------------------------------------------
# Paths + constants
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[4]
CHPL_DIR = REPO_ROOT / "application/cloud/data/chpl_excerpts"
NFZ_INDEX_GLOB = REPO_ROOT / "application/cloud/data/nfz_formulary"
DEFAULT_MODEL = os.environ.get("VERTEX_AGENT_MODEL", "gemini-2.5-pro")

MANIFEST_PATH = CHPL_DIR / "chpl_manifest.json"
FAILURES_PATH = CHPL_DIR / "chpl_failures.json"

# ---------------------------------------------------------------------------
# Prompts (Polish)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_PL = """\
Jesteś klinicznym farmakologiem przygotowującym streszczenia Charakterystyki
Produktu Leczniczego (ChPL) dla polskiego Systemu Wsparcia Decyzji Klinicznych
(CDSS). Twoje odpowiedzi są zwięzłe, profesjonalne, w języku polskim, oparte
na aktualnych ChPL dostępnych w URPL (urpl.gov.pl) i bazach krajowych.

ZASADY:
1. Każdy excerpt MUSI zawierać poniższe sekcje DOKŁADNIE w tej kolejności:
   - Wprowadzenie: 1-2 zdania (klasa terapeutyczna + główne wskazanie kliniczne)
   - **Wskazania**: lista 3-6 najważniejszych zarejestrowanych wskazań
   - **Dawkowanie**: typowe dawki dla 1-3 najczęstszych wskazań
   - **Przeciwwskazania**: 2-4 kluczowe (bezwzględne)
   - **Ostrzeżenia**: 2-4 istotne klinicznie (interakcje, monitorowanie, populacje szczególne)
   - **Refundacja NFZ**: krótka notka (tier refundacyjny, ograniczenia jeśli istotne)

2. Format: czysty markdown bez YAML frontmatter (frontmatter dodawany jest osobno).
3. Długość: 200-400 słów. Bądź zwięzły — to materiał dla CDSS, nie pełna ulotka.
4. Język: profesjonalny medyczny polski. Jednostki SI (mg, ml, mg/dobę).
5. Korzystaj z aktualnych źródeł — preferuj URPL, ChPL i krajowe wytyczne (PTK, PTG-E, PTD itp.).
6. NIE generuj danych pacjenta, dawek niestandardowych ani spekulacji terapeutycznych
   poza zarejestrowanymi wskazaniami.

Output: TYLKO markdown z sekcjami, bez metadanych, bez ```markdown``` fences.
"""

USER_PROMPT_TEMPLATE = """\
Wygeneruj ChPL excerpt dla leku:
- Kod ATC: {atc}
- Substancja czynna (INN): {substance}
- Nazwy handlowe w Polsce: {trade_names}

Streszczenie w formacie markdown, ~250-350 słów, według struktury podanej w
systemowych zasadach.
"""

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class NfzEntry:
    atc_code: str
    substance: str
    trade_names: list[str]
    reimbursement: str
    patient_copay_pln: str


@dataclass
class GenerationResult:
    atc: str
    substance: str
    file: str
    generated_at: str
    sha256: str
    word_count: int
    sources: list[str] = field(default_factory=list)
    grounded: bool = False


@dataclass
class GenerationFailure:
    atc: str
    substance: str
    error: str
    attempted_at: str


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TransientChPLError(RuntimeError):
    """Retryable error (5xx / 429 / temporary grounding hiccup)."""


class PermanentChPLError(RuntimeError):
    """Non-retryable (auth, malformed response, empty body)."""


# ---------------------------------------------------------------------------
# NFZ index loader
# ---------------------------------------------------------------------------


def _latest_nfz_index() -> Path:
    """Return the latest `index_<date>.json` file from the NFZ formulary."""
    candidates = sorted(NFZ_INDEX_GLOB.glob("index_*.json"))
    if not candidates:
        raise RuntimeError(
            f"No NFZ formulary index found under {NFZ_INDEX_GLOB}. "
            "Run `ingest_nfz.py` first."
        )
    return candidates[-1]


def load_nfz_entries(index_path: Path | None = None) -> list[NfzEntry]:
    """Load NFZ entries from the index JSON."""
    path = index_path or _latest_nfz_index()
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("entries", [])
    out: list[NfzEntry] = []
    for raw in entries:
        atc = raw.get("atc_code", "").strip().upper()
        substance = raw.get("substance", "").strip()
        if not atc or not substance:
            continue
        out.append(
            NfzEntry(
                atc_code=atc,
                substance=substance,
                trade_names=raw.get("trade_names", []) or [],
                reimbursement=raw.get("reimbursement", "") or "",
                patient_copay_pln=raw.get("patient_copay_pln", "") or "",
            )
        )
    return out


# ---------------------------------------------------------------------------
# Filename / slug helpers
# ---------------------------------------------------------------------------

_SLUG_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _slug(text: str) -> str:
    """Lowercase + strip diacritics + replace non-alnum with `_`."""
    import unicodedata

    norm = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    norm = norm.lower()
    norm = _SLUG_NON_ALNUM.sub("_", norm).strip("_")
    return norm or "unknown"


def _has_existing_file_for_atc(atc: str) -> Path | None:
    """Check if any `<atc>_*.md` already exists in CHPL_DIR."""
    matches = list(CHPL_DIR.glob(f"{atc}_*.md"))
    return matches[0] if matches else None


# ---------------------------------------------------------------------------
# Vertex client (grounding-enabled — separate from cloud.llm_integration)
# ---------------------------------------------------------------------------


def _build_genai_client() -> Any:
    """Build a Vertex genai client. Reads project/location from env."""
    from google import genai

    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise PermanentChPLError(
            "GOOGLE_CLOUD_PROJECT is not set; copy .env.example to .env"
        )
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    return genai.Client(vertexai=True, project=project, location=location)


@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type(TransientChPLError),
)
def _generate_one_sync(
    client: Any,
    *,
    model: str,
    entry: NfzEntry,
    enable_grounding: bool,
) -> tuple[str, list[str]]:
    """Sync core: call Vertex Gemini with grounding, return (text, sources)."""
    from google.genai import errors as genai_errors
    from google.genai import types

    trade_list = ", ".join(entry.trade_names[:5]) if entry.trade_names else "—"
    user_prompt = USER_PROMPT_TEMPLATE.format(
        atc=entry.atc_code,
        substance=entry.substance,
        trade_names=trade_list,
    )

    config_kwargs: dict[str, Any] = {
        "temperature": 0.2,
        "max_output_tokens": 2048,
        "system_instruction": SYSTEM_PROMPT_PL,
    }
    if enable_grounding:
        # `google_search` tool — works on Vertex gemini-2.5-pro.
        # Cannot combine with `response_schema` (mutually exclusive).
        config_kwargs["tools"] = [
            types.Tool(google_search=types.GoogleSearch())
        ]

    try:
        response = client.models.generate_content(
            model=model,
            contents=user_prompt,
            config=types.GenerateContentConfig(**config_kwargs),
        )
    except genai_errors.APIError as exc:
        status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        if isinstance(status, int) and (status == 429 or 500 <= status < 600):
            raise TransientChPLError(f"{status} on {entry.atc_code}: {exc}") from exc
        raise PermanentChPLError(f"{status} on {entry.atc_code}: {exc}") from exc

    text = getattr(response, "text", None) or ""
    text = text.strip()
    if not text:
        raise PermanentChPLError(f"empty body from Gemini for {entry.atc_code}")

    # Strip accidental ```markdown fences.
    if text.startswith("```"):
        text = re.sub(r"^```(?:markdown|md)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()

    # Extract grounding sources from `candidates[0].grounding_metadata`.
    sources: list[str] = []
    candidates = getattr(response, "candidates", []) or []
    if candidates:
        gm = getattr(candidates[0], "grounding_metadata", None)
        if gm is not None:
            chunks = getattr(gm, "grounding_chunks", None) or []
            for chunk in chunks:
                web = getattr(chunk, "web", None)
                if web is not None:
                    uri = getattr(web, "uri", None)
                    if uri:
                        sources.append(uri)

    return text, sources


# ---------------------------------------------------------------------------
# Async wrapper — semaphore-controlled
# ---------------------------------------------------------------------------


async def _generate_with_sem(
    sem: asyncio.Semaphore,
    client: Any,
    *,
    model: str,
    entry: NfzEntry,
    enable_grounding: bool,
) -> tuple[NfzEntry, str | None, list[str], str | None]:
    """Run sync gen inside a thread under semaphore; return (entry, body, sources, error)."""
    async with sem:
        loop = asyncio.get_running_loop()
        try:
            body, sources = await loop.run_in_executor(
                None,
                lambda: _generate_one_sync(
                    client,
                    model=model,
                    entry=entry,
                    enable_grounding=enable_grounding,
                ),
            )
            return entry, body, sources, None
        except Exception as exc:
            log.warning("generation failed for %s: %s", entry.atc_code, exc)
            return entry, None, [], str(exc)


# ---------------------------------------------------------------------------
# Markdown writer
# ---------------------------------------------------------------------------


def _build_markdown(
    entry: NfzEntry, body: str, sources: list[str]
) -> tuple[str, str]:
    """Build full markdown (frontmatter + body), return (filename, content)."""
    slug = _slug(entry.substance)
    filename = f"{entry.atc_code}_{slug}.md"
    generated_at = datetime.now().astimezone().strftime("%Y-%m-%d")

    fm_lines = [
        "---",
        f"atc_code: {entry.atc_code}",
        f"drug_name: {entry.substance}",
        "source: Vertex Gemini 2.5 Pro (Google Search grounding, URPL/ChPL)",
        f"generated_at: {generated_at}",
        "licensing: derived-summary",
    ]
    if entry.trade_names:
        trade_str = "; ".join(entry.trade_names[:5])
        fm_lines.append(f"trade_names: {trade_str}")
    if sources:
        fm_lines.append("grounding_sources:")
        for src in sources[:10]:
            fm_lines.append(f"  - {src}")
    fm_lines.append("---")
    fm_lines.append("")

    content = "\n".join(fm_lines) + body.strip() + "\n"
    return filename, content


def _write_atomic(path: Path, content: str) -> str:
    """Atomic write, return sha256 of content."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------


def _load_manifest() -> dict[str, Any]:
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log.warning("manifest is malformed — starting fresh")
    return {"entries": [], "generated_at": None}


def _save_manifest(manifest: dict[str, Any]) -> None:
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def _save_failures(failures: list[GenerationFailure]) -> None:
    if not failures:
        if FAILURES_PATH.exists():
            FAILURES_PATH.unlink()
        return
    data = [
        {
            "atc": f.atc,
            "substance": f.substance,
            "error": f.error,
            "attempted_at": f.attempted_at,
        }
        for f in failures
    ]
    FAILURES_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


async def _run_async(
    pending: list[NfzEntry],
    *,
    model: str,
    concurrency: int,
    enable_grounding: bool,
) -> tuple[list[GenerationResult], list[GenerationFailure]]:
    client = _build_genai_client()
    sem = asyncio.Semaphore(concurrency)
    results: list[GenerationResult] = []
    failures: list[GenerationFailure] = []

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
            f"Generating ChPL (model={model}, conc={concurrency})",
            total=len(pending),
        )

        tasks = [
            asyncio.create_task(
                _generate_with_sem(
                    sem,
                    client,
                    model=model,
                    entry=e,
                    enable_grounding=enable_grounding,
                )
            )
            for e in pending
        ]

        for coro in asyncio.as_completed(tasks):
            entry, body, sources, error = await coro
            if error or not body:
                failures.append(
                    GenerationFailure(
                        atc=entry.atc_code,
                        substance=entry.substance,
                        error=error or "empty body",
                        attempted_at=datetime.now().astimezone().isoformat(),
                    )
                )
            else:
                filename, content = _build_markdown(entry, body, sources)
                path = CHPL_DIR / filename
                sha = _write_atomic(path, content)
                results.append(
                    GenerationResult(
                        atc=entry.atc_code,
                        substance=entry.substance,
                        file=filename,
                        generated_at=datetime.now().astimezone().isoformat(),
                        sha256=sha,
                        word_count=len(body.split()),
                        sources=sources,
                        grounded=bool(sources),
                    )
                )
            progress.advance(task)

    return results, failures


@app.command()
def main(
    concurrency: int = typer.Option(
        8, "--concurrency", "-c", help="Concurrent Vertex requests."
    ),
    limit: int = typer.Option(
        0, "--limit", "-l", help="Max entries to generate (0 = all)."
    ),
    model: str = typer.Option(
        DEFAULT_MODEL, "--model", help="Vertex model name."
    ),
    no_grounding: bool = typer.Option(
        False, "--no-grounding", help="Disable Google Search grounding (cheaper, less accurate)."
    ),
    index_path: Path | None = typer.Option(
        None, "--index-path", help="Override NFZ index JSON path."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Generate ChPL excerpts for every NFZ index entry missing one."""
    load_dotenv(override=True)
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    entries = load_nfz_entries(index_path=index_path)
    total = len(entries)

    # Idempotent: skip entries with an existing `<atc>_*.md` file.
    pending: list[NfzEntry] = []
    skipped: list[NfzEntry] = []
    for e in entries:
        existing = _has_existing_file_for_atc(e.atc_code)
        if existing is not None:
            skipped.append(e)
        else:
            pending.append(e)

    if limit > 0:
        pending = pending[:limit]

    console.print(
        f"[bold]NFZ index entries:[/bold] {total}  "
        f"[green]existing[/green]: {len(skipped)}  "
        f"[yellow]pending[/yellow]: {len(pending)}"
    )
    if not pending:
        console.print("[green]All ATC codes already have a ChPL file. Nothing to do.[/green]")
        raise typer.Exit(code=0)

    enable_grounding = not no_grounding
    if enable_grounding:
        console.print(
            "[cyan]Grounding ON[/cyan] (Google Search Retrieval — "
            "first 1500 grounded queries/day are free on Vertex)."
        )
    else:
        console.print("[yellow]Grounding OFF[/yellow] (parametric only).")

    started = time.time()
    results, failures = asyncio.run(
        _run_async(
            pending,
            model=model,
            concurrency=concurrency,
            enable_grounding=enable_grounding,
        )
    )
    elapsed = time.time() - started

    # Update manifest (append new results to whatever was there before).
    manifest = _load_manifest()
    existing_entries = manifest.get("entries", [])
    by_file = {e["file"]: e for e in existing_entries}
    for r in results:
        by_file[r.file] = {
            "atc": r.atc,
            "substance": r.substance,
            "file": r.file,
            "generated_at": r.generated_at,
            "sha256": r.sha256,
            "word_count": r.word_count,
            "grounded": r.grounded,
            "sources": r.sources,
        }
    manifest["entries"] = sorted(by_file.values(), key=lambda x: x["atc"])
    manifest["generated_at"] = datetime.now().astimezone().isoformat()
    manifest["model"] = model
    _save_manifest(manifest)
    _save_failures(failures)

    grounded_count = sum(1 for r in results if r.grounded)
    console.print(
        f"\n[bold green]Done.[/bold green] "
        f"Generated [bold]{len(results)}[/bold] ChPL files in "
        f"[bold]{elapsed/60:.1f} min[/bold]  "
        f"(grounded: {grounded_count}/{len(results)}, "
        f"failures: {len(failures)})."
    )
    if failures:
        console.print(
            f"[red]Failures saved to {FAILURES_PATH.name}[/red] — "
            "re-run to retry."
        )


if __name__ == "__main__":
    app()
