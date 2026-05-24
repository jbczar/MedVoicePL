"""Rebuild the Qdrant ChPL index after Phase 4 corpus expansion.

Reads all `*.md` files under `application/cloud/data/chpl_excerpts/`,
embeds them with `sdadas/mmlw-roberta-large`, then drops + repopulates
the Qdrant collection at `QDRANT_PATH` / `QDRANT_COLLECTION`.

Phase 4 expanded the corpus from 14 hand-curated excerpts to ~621
(14 hand + 607 LLM-generated with URPL grounding). The retriever's
`build_qdrant_retriever()` already does the full
`recreate_collection → embed → upsert` dance — this script is a thin
CLI wrapper with progress reporting and a smoke check on top.

Run::

    uv run python application/cloud/scripts/rebuild_chpl_index.py

After completion, restart the cloud API so the cached `ChPLChain`
picks up the new vectors (`make cloud-down && make cloud-up`).
"""

from __future__ import annotations

import logging
import os
import shutil
import time
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
    TimeRemainingColumn,
)
from rich.table import Table

console = Console()
log = logging.getLogger("rebuild_chpl_index")
app = typer.Typer(add_completion=False, help="Rebuild Qdrant ChPL index from disk.")

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CORPUS_DIR = REPO_ROOT / "application/cloud/data/chpl_excerpts"
DEFAULT_QDRANT_PATH = REPO_ROOT / "application/cloud/data/qdrant"


# Sanity queries — exercised after rebuild to confirm semantic retrieval
# actually works. Each query is a Polish clinical phrase; the top result
# is expected to be from the listed ATC family (we match the prefix, not
# an exact code, because clinically equivalent drugs in the same class
# all count as "correct" for these smoke probes).
SMOKE_QUERIES: list[tuple[str, str]] = [
    ("nadciśnienie tętnicze", "C09"),  # ACE-I / ARB family
    ("padaczka leczenie", "N03"),  # antiepileptics
    ("cukrzyca typu 2", "A10"),  # antidiabetics
    ("zapalenie płuc antybiotyk", "J01"),  # systemic antibacterials
    ("refluks żołądkowo-przełykowy", "A02"),  # acid-related disorders
]


@app.command()
def main(
    corpus_dir: Path = typer.Option(
        DEFAULT_CORPUS_DIR,
        "--corpus-dir",
        help="Directory with ChPL `.md` files.",
    ),
    qdrant_path: Path | None = typer.Option(
        None,
        "--qdrant-path",
        help="Override QDRANT_PATH from env.",
    ),
    collection: str | None = typer.Option(
        None,
        "--collection",
        help="Override QDRANT_COLLECTION from env.",
    ),
    embedding_model: str | None = typer.Option(
        None,
        "--embedding-model",
        help="Override EMBEDDING_MODEL from env (default sdadas/mmlw-roberta-large).",
    ),
    batch_size: int = typer.Option(
        32,
        "--batch-size",
        help="Embedding batch size (larger = faster, more RAM).",
    ),
    wipe_existing: bool = typer.Option(
        False,
        "--wipe-existing",
        help=(
            "Delete the entire Qdrant directory before rebuild. Use when "
            "Qdrant is in an inconsistent state (collection conflicts, "
            "schema mismatch). Otherwise we just recreate the collection."
        ),
    ),
    skip_smoke: bool = typer.Option(
        False,
        "--skip-smoke",
        help="Skip the post-rebuild semantic-query smoke check.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Rebuild the Qdrant ChPL collection from `corpus_dir`."""
    load_dotenv(override=True)
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # ----- resolve config -----
    qdrant_path = qdrant_path or Path(
        os.environ.get("QDRANT_PATH", str(DEFAULT_QDRANT_PATH))
    )
    # Resolve relative paths (e.g. from .env: QDRANT_PATH=application/cloud/data/qdrant)
    # against REPO_ROOT so downstream `.relative_to(REPO_ROOT)` calls don't blow up.
    if not qdrant_path.is_absolute():
        qdrant_path = (REPO_ROOT / qdrant_path).resolve()
    if not corpus_dir.is_absolute():
        corpus_dir = (REPO_ROOT / corpus_dir).resolve()
    collection = collection or os.environ.get("QDRANT_COLLECTION", "chpl_excerpts")
    embedding_model = embedding_model or os.environ.get(
        "EMBEDDING_MODEL", "sdadas/mmlw-roberta-large"
    )

    if not corpus_dir.is_dir():
        console.print(f"[red]Corpus dir does not exist:[/red] {corpus_dir}")
        raise typer.Exit(code=1)

    # ----- load excerpts -----
    from cloud.rag.chain import load_chpl_excerpts

    console.print(
        f"[cyan]Loading ChPL corpus from[/cyan] {corpus_dir.relative_to(REPO_ROOT)}"
    )
    excerpts = load_chpl_excerpts(corpus_dir)
    if not excerpts:
        console.print(f"[red]No usable ChPL excerpts found in {corpus_dir}[/red]")
        raise typer.Exit(code=1)
    console.print(f"  → [bold]{len(excerpts)}[/bold] excerpts loaded.")

    # ----- optionally wipe -----
    if wipe_existing and qdrant_path.is_dir():
        console.print(f"[yellow]Wiping existing Qdrant dir:[/yellow] {qdrant_path}")
        shutil.rmtree(qdrant_path)

    qdrant_path.mkdir(parents=True, exist_ok=True)

    # ----- build embedder + embed in batches with progress -----
    from cloud.rag.embedder import _RealEmbedder

    console.print(
        f"[cyan]Loading embedder[/cyan] [bold]{embedding_model}[/bold] "
        "(first time downloads ~700 MB)…"
    )
    embedder = _RealEmbedder(model_id=embedding_model)
    # Force the lazy import so the download bar (if any) appears before
    # we start the Rich Progress block.
    _ = embedder.embed_query("warmup")

    texts = [e.text for e in excerpts]
    vectors: list[list[float]] = []

    started = time.time()
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
        task = progress.add_task("Embedding ChPL excerpts", total=len(texts))
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_vecs = embedder.embed_documents(batch)
            vectors.extend(batch_vecs)
            progress.advance(task, advance=len(batch))
    embed_elapsed = time.time() - started
    console.print(
        f"  → embedded {len(vectors)} docs in [bold]{embed_elapsed:.1f}s[/bold]"
    )

    # ----- write to Qdrant (recreate collection + upsert) -----
    console.print(
        f"[cyan]Recreating Qdrant collection[/cyan] [bold]{collection}[/bold] "
        f"at {qdrant_path.relative_to(REPO_ROOT)}…"
    )

    from qdrant_client import QdrantClient
    from qdrant_client import models as qmodels

    client = QdrantClient(path=str(qdrant_path))
    client.recreate_collection(
        collection_name=collection,
        vectors_config=qmodels.VectorParams(
            size=embedder.dim,
            distance=qmodels.Distance.COSINE,
        ),
    )

    # Upsert in chunks of 128 — Qdrant in-process handles larger but it's
    # not free if the OS pages the SQLite-style backing file.
    upsert_chunk = 128
    started = time.time()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TextColumn("[bold]{task.completed}/{task.total}[/bold]"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Upserting to Qdrant", total=len(vectors))
        for i in range(0, len(vectors), upsert_chunk):
            chunk_excerpts = excerpts[i : i + upsert_chunk]
            chunk_vectors = vectors[i : i + upsert_chunk]
            points = [
                qmodels.PointStruct(
                    id=i + idx,
                    vector=vec,
                    payload={
                        "atc_code": e.atc_code,
                        "drug_name": e.drug_name,
                        "text": e.text,
                        "source": e.source,
                    },
                )
                for idx, (e, vec) in enumerate(
                    zip(chunk_excerpts, chunk_vectors, strict=True)
                )
            ]
            client.upsert(collection_name=collection, points=points)
            progress.advance(task, advance=len(points))
    upsert_elapsed = time.time() - started
    console.print(
        f"  → upserted {len(vectors)} points in [bold]{upsert_elapsed:.1f}s[/bold]"
    )

    # ----- smoke check -----
    if not skip_smoke:
        _run_smoke_queries(client, collection, embedder, excerpts)

    # ----- summary -----
    summary = Table(title="Rebuild summary", show_lines=False)
    summary.add_column("metric")
    summary.add_column("value", justify="right")
    summary.add_row("corpus dir", str(corpus_dir.relative_to(REPO_ROOT)))
    summary.add_row("qdrant path", str(qdrant_path.relative_to(REPO_ROOT)))
    summary.add_row("collection", collection)
    summary.add_row("embedding model", embedding_model)
    summary.add_row("docs embedded", str(len(vectors)))
    summary.add_row("embed time", f"{embed_elapsed:.1f}s")
    summary.add_row("upsert time", f"{upsert_elapsed:.1f}s")
    summary.add_row("total time", f"{embed_elapsed + upsert_elapsed:.1f}s")
    console.print()
    console.print(summary)

    console.print(
        "\n[green]Done.[/green]  "
        "Restart cloud API to pick up new vectors: "
        "[bold]make cloud-down && make cloud-up[/bold]"
    )


def _run_smoke_queries(
    client,  # type: ignore[no-untyped-def]
    collection: str,
    embedder,  # type: ignore[no-untyped-def]
    excerpts,  # type: ignore[no-untyped-def]
) -> None:
    """Run a handful of Polish clinical queries to sanity-check retrieval."""
    from qdrant_client import models as qmodels

    console.print("\n[cyan]Smoke probe — semantic retrieval:[/cyan]")
    table = Table(show_lines=False)
    table.add_column("query")
    table.add_column("top hit (ATC)")
    table.add_column("drug")
    table.add_column("score", justify="right")
    table.add_column("match")

    for query, expected_atc_prefix in SMOKE_QUERIES:
        vector = embedder.embed_query(query)
        resp = client.query_points(
            collection_name=collection,
            query=vector,
            limit=1,
            search_params=qmodels.SearchParams(exact=True),
        )
        points = getattr(resp, "points", []) or []
        if not points:
            table.add_row(query, "—", "—", "—", "[red]NO HIT[/red]")
            continue
        hit = points[0]
        atc = (hit.payload or {}).get("atc_code", "")
        drug = (hit.payload or {}).get("drug_name", "")
        score = float(hit.score)
        match = (
            "[green]OK[/green]"
            if atc.startswith(expected_atc_prefix)
            else f"[yellow]≠ {expected_atc_prefix}*[/yellow]"
        )
        table.add_row(query, atc, drug, f"{score:.3f}", match)
    console.print(table)


if __name__ == "__main__":
    app()
