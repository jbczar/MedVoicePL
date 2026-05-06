"""FastAPI app for the cloud agent — Phase 3a.

Endpoints
---------

* ``GET /v1/health`` — returns
  ``{status, llm_loaded, nfz_index_size, chpl_corpus_size, agent_version}``.
  Quick liveness probe; doesn't trigger model load.

* ``POST /v1/agent/suggest`` — body is `AnonymizedSOAPNote` JSON,
  response is `AgentSuggestion` JSON. Idempotency cache keyed by
  ``sha256(payload_canonical_json) + AGENT_VERSION``; ``?force=true``
  bypasses cache.

Auth: deferred to Phase 3b (gateway middleware). API binds
``127.0.0.1:8002`` only; WARN log if a non-localhost host is configured.

Test posture: every "real" dep (`CloudAgent`, `NFZIndex`, `ChPLChain`,
`LLMClient`) goes through a FastAPI dependency override. Tests inject
fakes; no Vertex / Qdrant / sentence-transformers load in CI.
"""

from __future__ import annotations

# Load .env at import time so `uvicorn cloud.api.app:app` picks up
# GOOGLE_CLOUD_PROJECT, VERTEX_AGENT_MODEL, etc. without requiring the
# launcher to source the file. `override=True`: committed `.env` wins
# over stale shell vars (mirror Phase 3b gateway bootstrap).
from dotenv import load_dotenv as _load_dotenv

_load_dotenv(override=True)

import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
from collections.abc import AsyncIterator  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Annotated, Any  # noqa: E402

from fastapi import Depends, FastAPI, HTTPException, Query, Response  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from cloud.orchestrator import CloudAgent  # noqa: E402
from shared.schemas import AgentSuggestion, AnonymizedSOAPNote  # noqa: E402

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _agent_version() -> str:
    return os.environ.get("AGENT_VERSION", "3a.0")


def _cache_root() -> Path:
    return Path(
        os.environ.get(
            "CLOUD_AGENT_CACHE_ROOT",
            "application/cloud/output/agent_suggestions",
        )
    ).resolve()


def _api_host() -> str:
    return os.environ.get("CLOUD_API_HOST", "127.0.0.1")


# ---------------------------------------------------------------------------
# Dependency provider — overridable in tests.
# ---------------------------------------------------------------------------


_AGENT_HOLDER: dict[str, CloudAgent] = {}


def get_agent() -> CloudAgent:
    """Production factory — builds a `CloudAgent` from configured deps.

    Tests override this with `app.dependency_overrides[get_agent] = ...`.
    """
    if "agent" not in _AGENT_HOLDER:
        from cloud.llm_integration.llm_client import build_llm_client
        from cloud.nfz.atc_resolver import build_resolver
        from cloud.nfz.formulary_search import latest_index_path, load_nfz_index
        from cloud.rag.chain import build_chain

        nfz_dir = Path(
            os.environ.get("NFZ_INDEX_DIR", "application/cloud/data/nfz_formulary")
        )
        index_path = latest_index_path(nfz_dir)
        if index_path is None:
            raise RuntimeError(
                f"no NFZ index_*.json found in {nfz_dir}; run ingest_nfz.py first"
            )

        chpl_dir = Path(
            os.environ.get("CHPL_CORPUS_DIR", "application/cloud/data/chpl_excerpts")
        )
        qdrant_path = os.environ.get("QDRANT_PATH")

        llm_client = build_llm_client()
        cache_path = nfz_dir / "atc_cache.json"
        _AGENT_HOLDER["agent"] = CloudAgent(
            llm=llm_client,
            nfz=load_nfz_index(index_path),
            chain=build_chain(
                corpus_dir=chpl_dir,
                qdrant_path=Path(qdrant_path) if qdrant_path else None,
            ),
            atc_resolver=build_resolver(llm=llm_client, cache_path=cache_path),
            agent_version=_agent_version(),
        )
    return _AGENT_HOLDER["agent"]


# ---------------------------------------------------------------------------
# Lifespan — host-binding sanity log.
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    host = _api_host()
    if host not in {"127.0.0.1", "localhost", "::1"}:
        log.warning(
            "cloud agent API binding to non-localhost host %r — "
            "Phase 3a is localhost-only; gateway should mediate external traffic",
            host,
        )
    yield
    _AGENT_HOLDER.clear()


app = FastAPI(
    title="MedVoicePL Cloud Agent",
    description="Phase 3a — LangGraph agent + RAG + NFZ.",
    version=_agent_version(),
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/v1/health")
def health(agent: Annotated[CloudAgent | None, Depends(get_agent)] = None) -> dict[str, Any]:
    """Liveness probe. Returns agent stats without triggering work."""
    return {
        "status": "ok",
        "llm_loaded": agent is not None and getattr(agent, "llm", None) is not None,
        "nfz_index_size": len(agent.nfz) if agent else 0,
        "chpl_corpus_size": len(agent.chain.retriever) if agent else 0,
        "agent_version": _agent_version(),
    }


# ---------------------------------------------------------------------------
# /v1/agent/suggest
# ---------------------------------------------------------------------------


@app.post("/v1/agent/suggest", response_model=AgentSuggestion)
def suggest(
    payload: AnonymizedSOAPNote,
    response: Response,
    agent: Annotated[CloudAgent, Depends(get_agent)],
    force: Annotated[bool, Query(description="Bypass idempotency cache.")] = False,
) -> AgentSuggestion:
    """Run the cloud agent on an `AnonymizedSOAPNote`.

    Returns the resulting `AgentSuggestion`. Idempotent: identical
    request body + same `AGENT_VERSION` returns the cached result with
    `X-Cached: true` header.
    """
    cache_key = _cache_key(payload, _agent_version())
    cache_path = _cache_root() / f"{cache_key}.json"

    if not force and cache_path.is_file():
        try:
            cached = AgentSuggestion.model_validate_json(cache_path.read_text("utf-8"))
            response.headers["X-Cached"] = "true"
            return cached
        except (ValidationError, OSError):
            log.warning("stale cache at %s — recomputing", cache_path)

    try:
        suggestion = agent.run(payload)
    except Exception as exc:
        log.exception("agent run failed")
        raise HTTPException(
            status_code=500,
            detail={"error": "agent_failed", "message": str(exc)},
        ) from exc

    _persist_cache(cache_path, suggestion)
    response.headers["X-Cached"] = "false"
    return suggestion


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _cache_key(payload: AnonymizedSOAPNote, agent_version: str) -> str:
    canonical = payload.model_dump_json()
    return hashlib.sha256(
        f"{canonical}|{agent_version}".encode()
    ).hexdigest()[:32]


def _persist_cache(path: Path, suggestion: AgentSuggestion) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(suggestion.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(tmp, path)
