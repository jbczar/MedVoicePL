"""FastAPI app: localhost-only endpoint for the gateway to call.

Endpoints
---------

* ``GET /v1/health`` — returns
  ``{status, analyzer_loaded, anonymizer_version}``. Used by the gateway
  + a deploy-time smoke. Trivial; does NOT trigger spaCy load.

* ``POST /v1/visits/{transcript_id}/anonymized-soap?session_id=<uuid>&specialization=<s>&visit_date=<iso>&force=false``
  — runs the full chain:

  1. Read ``output/transcripts/<transcript_id>/transcript.json``.
     ``404`` if missing.
  2. Idempotency: if
     ``output/anonymized/<transcript_id>/anonymized_soap.json``
     exists AND the request matches the cached version, return it
     with ``X-Cached: true``. Cache key is the transcript SHA + the
     `ANONYMIZER_VERSION` env. Bumping the env invalidates.
  3. Run ``transcript_to_soap`` (Phase 2b) with the configured
     `StructuredExtractor`.
  4. Run ``PresidioAnonymizer.anonymize_soap`` (Phase 2c).
  5. Run ``trust_gate.validate_outbound``. On failure, log the audit
     row + return ``500``.
  6. Persist anonymized JSON, audit log, and token map.
  7. Return the ``AnonymizedSOAPNote`` JSON.

Auth: deferred to Phase 3 (gateway middleware). 2c binds
``127.0.0.1:8001`` only with WARN log if a non-localhost host is
configured.

Test posture (DD-5): every "real" dependency (`StructuredExtractor`,
`PresidioAnonymizer`, `AuditLogger`) goes through a FastAPI
dependency. Tests use `app.dependency_overrides` to inject stubs and
fake registries — no spaCy / mlx-lm load needed in CI.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Annotated, Any
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, Query, Response
from pydantic import BaseModel

from on_premise.anonymization.audit_logger import AuditLogger
from on_premise.anonymization.presidio_agent import (
    PresidioAnonymizer,
    default_registry,
    warm_analyzer,
)
from on_premise.structuring.pipeline import (
    SOAPExtractionError,
    transcript_to_soap,
)
from on_premise.stt.schema import Transcript
from on_premise.trust_gate import (
    TrustBoundaryViolation,
    validate_outbound,
    violations_to_dicts,
)
from shared.schemas import AnonymizedSOAPNote

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration — read from env, with sensible localhost defaults.
# ---------------------------------------------------------------------------


def _anonymizer_version() -> str:
    return os.environ.get("ANONYMIZER_VERSION", "2c.0")


def _audit_root() -> Path:
    return Path(
        os.environ.get(
            "AUDIT_ROOT",
            "application/on_premise/audit",
        )
    ).resolve()


def _audit_retention_days() -> int:
    return int(os.environ.get("AUDIT_RETENTION_DAYS", "730"))


def _audit_controller() -> str:
    return os.environ.get("ANONYMIZER_CONTROLLER", "MedVoicePL")


def _transcripts_root() -> Path:
    return Path(
        os.environ.get(
            "ON_PREMISE_TRANSCRIPTS_ROOT",
            "application/on_premise/output/transcripts",
        )
    ).resolve()


def _anonymized_root() -> Path:
    return Path(
        os.environ.get(
            "ON_PREMISE_ANONYMIZED_ROOT",
            "application/on_premise/output/anonymized",
        )
    ).resolve()


def _scenarios_root() -> Path | None:
    """Optional — feeds the name booster. Not required."""
    raw = os.environ.get(
        "DATASET_SCENARIOS_ROOT",
        "dataset_pipeline/output/scenarios",
    )
    p = Path(raw).resolve()
    return p if p.is_dir() else None


# ---------------------------------------------------------------------------
# Dependencies — overridable in tests via app.dependency_overrides.
# ---------------------------------------------------------------------------


def get_anonymizer() -> PresidioAnonymizer:
    """Default factory. Tests override with a stub-registry-backed instance."""
    return PresidioAnonymizer(registry=default_registry(scenarios_dir=_scenarios_root()))


def get_audit_logger() -> AuditLogger:
    return AuditLogger(
        audit_root=_audit_root(),
        retention_days=_audit_retention_days(),
        controller=_audit_controller(),
    )


def get_extractor() -> Any:
    """Default extractor for SOAP generation.

    Lazy-imported because building a real `StructuredExtractor` would
    load mlx-lm + ~6 GB Bielik weights — we don't want that on import.
    Tests override with a `_StubExtractor`.
    """
    from on_premise.structuring.slm_engine import build_extractor, load_slm_config

    config = load_slm_config(Path("application/on_premise/configs/slm.yaml"))
    return build_extractor(config, env=os.environ)


# ---------------------------------------------------------------------------
# Request/response models for OpenAPI docs.
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str
    analyzer_loaded: bool
    anonymizer_version: str


class TrustGateError(BaseModel):
    """Server-side trust-gate failure detail. HTTP 500."""

    detail: str
    violations: list[dict[str, str]]


# ---------------------------------------------------------------------------
# App factory.
# ---------------------------------------------------------------------------


def _maybe_log_non_localhost_warning() -> None:
    host = os.environ.get("ON_PREMISE_API_HOST", "127.0.0.1")
    if host not in ("127.0.0.1", "localhost", "::1"):
        log.warning(
            "ON_PREMISE_API_HOST=%s is NOT localhost. Phase 2c expects "
            "the gateway (Phase 3) to mediate any external access.",
            host,
        )


_analyzer_warm = False


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Warm the spaCy analyzer at startup so the first request is fast.

    Failure here is non-fatal — the analyzer load can take 2-3 s and
    fail under transient conditions (e.g. spaCy model wheel not yet
    materialized in cache). Log + continue; the first real request
    surfaces any persistent error.
    """
    global _analyzer_warm
    _maybe_log_non_localhost_warning()
    try:
        warm_analyzer()
        _analyzer_warm = True
        log.info("analyzer warmed at startup")
    except Exception as exc:  # pragma: no cover — exercised in smoke
        log.warning("could not warm analyzer at startup: %s", exc)
    yield


def create_app() -> FastAPI:
    """FastAPI app factory. Used both by `uvicorn` startup and tests.

    Tests can construct the app without invoking the lifespan hook by
    using `TestClient(app)` outside a `with` block — FastAPI runs
    lifespan only when entered as a context manager. They override
    dependencies before that fires.
    """
    app = FastAPI(
        title="MedVoicePL on-premise",
        description="Trust-boundary-bound API. Localhost-only by default.",
        version="2c.0",
        lifespan=_lifespan,
    )

    @app.get("/v1/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            analyzer_loaded=_analyzer_warm,
            anonymizer_version=_anonymizer_version(),
        )

    @app.post(
        "/v1/visits/{transcript_id}/anonymized-soap",
        response_model=AnonymizedSOAPNote,
        responses={
            404: {"description": "Transcript not found"},
            500: {"model": TrustGateError, "description": "Trust gate violation"},
        },
    )
    async def anonymize_visit(
        transcript_id: str,
        response: Response,
        anonymizer: Annotated[PresidioAnonymizer, Depends(get_anonymizer)],
        audit: Annotated[AuditLogger, Depends(get_audit_logger)],
        extractor: Annotated[Any, Depends(get_extractor)],
        session_id: UUID = Query(..., description="Gateway-minted session UUID"),
        specialization: str = Query(..., description="Medical specialization"),
        visit_date: date = Query(..., description="Visit date (YYYY-MM-DD)"),
        force: bool = Query(False, description="Bypass cache and re-run pipeline"),
    ) -> AnonymizedSOAPNote:
        # 1. Load transcript.
        transcript_path = _transcripts_root() / transcript_id / "transcript.json"
        if not transcript_path.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"transcript not found: {transcript_id}",
            )
        try:
            transcript = Transcript.model_validate_json(transcript_path.read_text())
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"could not parse transcript: {exc}",
            ) from exc

        # 2. Idempotency check.
        out_path = _anonymized_root() / transcript_id / "anonymized_soap.json"
        cached = _maybe_load_cached(out_path, transcript) if not force else None
        if cached is not None:
            response.headers["X-Cached"] = "true"
            return cached

        # 3. SOAP extraction (Phase 2b).
        try:
            soap = transcript_to_soap(transcript, extractor=extractor)
        except SOAPExtractionError as exc:
            raise HTTPException(status_code=500, detail=f"SOAP extraction failed: {exc}") from exc

        # Strip metadata — anonymize_soap takes the submission shape.
        from on_premise.structuring.soap_schema import SOAPNoteSubmission

        submission_dict = soap.model_dump()
        submission_dict.pop("metadata", None)
        submission = SOAPNoteSubmission.model_validate(submission_dict)

        # 4. Anonymize.
        result = anonymizer.anonymize_soap(
            submission,
            session_id=session_id,
            visit_date=visit_date,
            specialization=specialization,
            anonymizer_version=_anonymizer_version(),
        )

        # 5. Trust gate. Failure here is a SERVER bug, not a client error
        #    — anonymized output should always pass the gate.
        try:
            validate_outbound(result.note)
        except TrustBoundaryViolation as exc:
            audit.log_trust_gate(
                session_id=str(session_id),
                decision="fail",
                violations=violations_to_dicts(exc.violations),
            )
            raise HTTPException(
                status_code=500,
                detail={
                    "detail": "trust gate violation",
                    "violations": violations_to_dicts(exc.violations),
                },
            ) from exc

        # 6. Persist + audit.
        _persist(out_path, result.note, transcript)
        audit.log_anonymization(
            session_id=str(session_id),
            replacements=result.replacements,
            soap_sha256=_payload_sha256(result.note),
            transcript_sha256=transcript.audio_sha256,
        )
        audit.write_token_map(
            session_id=str(session_id),
            token_map=result.token_map,
        )
        audit.log_trust_gate(session_id=str(session_id), decision="pass")

        response.headers["X-Cached"] = "false"
        return result.note

    return app


# ---------------------------------------------------------------------------
# Persistence helpers.
# ---------------------------------------------------------------------------


def _persist(out_path: Path, note: AnonymizedSOAPNote, transcript: Transcript) -> None:
    """Write `anonymized_soap.json` plus a `cache_key.json` sidecar.

    The sidecar stores `{anonymizer_version, transcript_sha256}` so the
    next request can verify cache validity in O(1) without parsing the
    full payload.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(note.model_dump_json(indent=2), encoding="utf-8")
    os.replace(tmp, out_path)
    sidecar = out_path.parent / "cache_key.json"
    sidecar.write_text(
        json.dumps(
            {
                "anonymizer_version": _anonymizer_version(),
                "transcript_sha256": transcript.audio_sha256,
            }
        ),
        encoding="utf-8",
    )


def _maybe_load_cached(out_path: Path, transcript: Transcript) -> AnonymizedSOAPNote | None:
    """Return the cached `AnonymizedSOAPNote` if its sidecar matches the
    current request — version + transcript SHA. Otherwise None.
    """
    sidecar = out_path.parent / "cache_key.json"
    if not (out_path.is_file() and sidecar.is_file()):
        return None
    try:
        key = json.loads(sidecar.read_text())
    except (OSError, ValueError):
        return None
    if (
        key.get("anonymizer_version") != _anonymizer_version()
        or key.get("transcript_sha256") != transcript.audio_sha256
    ):
        return None
    try:
        return AnonymizedSOAPNote.model_validate_json(out_path.read_text())
    except (OSError, ValueError) as exc:
        log.warning("cached anonymized SOAP unreadable: %s", exc)
        return None


def _payload_sha256(note: AnonymizedSOAPNote) -> str:
    """SHA-256 of the canonical JSON form. Used as the audit `soap_sha256`
    so retroactive verification of "was this exact payload audited"
    is possible.
    """
    return hashlib.sha256(note.model_dump_json().encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Module-level app instance for `uvicorn on_premise.api.app:app`.
# ---------------------------------------------------------------------------


app = create_app()
