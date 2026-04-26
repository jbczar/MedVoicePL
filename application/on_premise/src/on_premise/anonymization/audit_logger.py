"""GDPR-compliant audit logging for anonymization operations.

`AuditLogger` records two kinds of events per session:

1. **Anonymization records** — one per PII replacement, written to
   `audit/<YYYY-MM-DD>/<session_id>/audit.json`. Each carries:

   * `entity_type`, `token` (the placeholder, e.g. `<PERSON_1>`),
     `location_path` (JSONPath-ish into the SOAP, e.g.
     `subjective.history_of_present_illness`), `location_offset_start`,
     `location_offset_end`.
   * `original_sha256` — SHA-256 of the redacted text. **Never the
     original itself.** Even the audit log doesn't hold raw PII.
   * GDPR Article 30 fields: `purpose`, `legal_basis`, `data_categories`,
     `controller`, `retention_until` (ISO date = today + retention_days).

2. **Trust-gate decisions** — `audit/<YYYY-MM-DD>/<session_id>/trust_gate.json`
   recording pass/fail and any `Violation` list raised by
   `validate_outbound`.

3. **Token map** — `audit/<YYYY-MM-DD>/<session_id>/token_map.json`
   stores `{token: original}` so the doctor's UI (Phase 3, on-premise
   side) can reverse-map. **chmod 0600** after atomic write — never
   readable by anyone but the on-prem process owner. Stays ON-PREM.

Path partitioning by date keeps the directory walkable even at 1k
visits/day. Sibling `purge_audit.py` (out of 2c scope) handles
retention enforcement.

Atomic write idiom: write to `.tmp`, then `os.replace`. Survives
crashes mid-write — either the new file is fully there or the old
one stays.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger(__name__)

# Default GDPR-Article-30 fields. `legal_basis` `Art_9_2_h` is the medical
# clause: "processing necessary for the purposes of preventive or
# occupational medicine, medical diagnosis, the provision of health or
# social care or treatment". The default `data_categories` covers the
# entity types our recognizers can produce; per-record this gets
# narrowed to the actual category surfaced.
_DEFAULT_PURPOSE = "anonymization"
_DEFAULT_LEGAL_BASIS = "Art_9_2_h"
_ALL_CATEGORIES = (
    "PERSON",
    "PHONE_NUMBER",
    "PL_PESEL",
    "PL_NIP",
    "PL_ID_CARD",
    "LOCATION",
    "EMAIL_ADDRESS",
)


@dataclass
class Replacement:
    """One PII replacement — what the audit logger persists.

    Lives in this module (NOT `shared/`); listed in `FORBIDDEN_NAMES`
    of the trust-boundary lint so it can never accidentally cross.
    """

    entity_type: str
    """Presidio entity type, e.g. ``"PERSON"``, ``"PL_PESEL"``."""

    token: str
    """Placeholder string, e.g. ``"<PERSON_1>"``."""

    original_sha256: str
    """SHA-256 hex digest of the original text. Audit holds the digest,
    not the plaintext."""

    location_path: str
    """JSONPath-ish location, e.g.
    ``"subjective.history_of_present_illness"``.
    """

    location_offset_start: int
    """Offset within the field's text where the original PII began."""

    location_offset_end: int
    """Offset within the field's text where the original PII ended."""


@dataclass
class _AuditRecord:
    """A complete audit record — one Replacement plus GDPR-Art.30 fields."""

    timestamp: str
    session_id: str
    soap_sha256: str
    transcript_sha256: str
    replacement: Replacement
    purpose: str
    legal_basis: str
    data_categories: list[str]
    controller: str
    retention_until: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "soap_sha256": self.soap_sha256,
            "transcript_sha256": self.transcript_sha256,
            "entity_type": self.replacement.entity_type,
            "token": self.replacement.token,
            "original_sha256": self.replacement.original_sha256,
            "location_path": self.replacement.location_path,
            "location_offset_start": self.replacement.location_offset_start,
            "location_offset_end": self.replacement.location_offset_end,
            "purpose": self.purpose,
            "legal_basis": self.legal_basis,
            "data_categories": self.data_categories,
            "controller": self.controller,
            "retention_until": self.retention_until,
        }


@dataclass
class TrustGateRecord:
    """One trust-gate decision row."""

    timestamp: str
    session_id: str
    decision: str  # "pass" | "fail"
    violations: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "decision": self.decision,
            "violations": self.violations,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def hash_pii(original: str) -> str:
    """Stable SHA-256 of the original PII string. Used in audit records.

    `original` is the verbatim PII (e.g. `"Krzysztof Nowakowski"`). The
    return value is the only fingerprint the audit log retains.
    """
    return hashlib.sha256(original.encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    """Write JSON to `path` via a `.tmp` sibling + `os.replace`.

    Atomic on POSIX filesystems: if the process dies after open() but
    before replace(), the original `path` (if any) is intact and the
    `.tmp` file can be cleaned up.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _utc_now_iso() -> str:
    """Microsecond-precise UTC timestamp for record `timestamp` field."""
    return datetime.now(tz=UTC).isoformat()


# ---------------------------------------------------------------------------
# AuditLogger
# ---------------------------------------------------------------------------


class AuditLogger:
    """Writes anonymization + trust-gate audit records and the token map.

    Construct once per process (or per FastAPI app instance); call
    `log_anonymization` / `log_trust_gate` per-session.

    Args:
        audit_root: Directory under which date-partitioned session
            directories will be written. Defaults to
            `application/on_premise/audit/`.
        retention_days: How many days each record's `retention_until`
            field looks ahead. Default 730 (2 years; refine with legal
            review for production deploys).
        controller: GDPR Article 30 data-controller name. Appears on
            every record.
    """

    def __init__(
        self,
        audit_root: Path,
        *,
        retention_days: int = 730,
        controller: str = "MedVoicePL",
    ) -> None:
        self._audit_root = audit_root
        self._retention_days = retention_days
        self._controller = controller

    # ----- public API ------------------------------------------------------

    def log_anonymization(
        self,
        *,
        session_id: str,
        replacements: list[Replacement],
        soap_sha256: str,
        transcript_sha256: str,
        when: date | None = None,
    ) -> Path:
        """Persist `audit.json` for a session and return the path."""
        d = when or date.today()
        records = [
            _AuditRecord(
                timestamp=_utc_now_iso(),
                session_id=session_id,
                soap_sha256=soap_sha256,
                transcript_sha256=transcript_sha256,
                replacement=r,
                purpose=_DEFAULT_PURPOSE,
                legal_basis=_DEFAULT_LEGAL_BASIS,
                data_categories=[r.entity_type],
                controller=self._controller,
                retention_until=(d + timedelta(days=self._retention_days)).isoformat(),
            )
            for r in replacements
        ]
        path = self._session_dir(d, session_id) / "audit.json"
        payload = {
            "schema_version": "2c.0",
            "session_id": session_id,
            "records": [r.to_dict() for r in records],
        }
        _atomic_write_json(path, payload)
        log.info(
            "audit: wrote %d anonymization record(s) for session %s -> %s",
            len(records),
            session_id,
            path,
        )
        return path

    def log_trust_gate(
        self,
        *,
        session_id: str,
        decision: str,
        violations: list[dict[str, str]] | None = None,
        when: date | None = None,
    ) -> Path:
        """Persist `trust_gate.json` for a session and return the path."""
        if decision not in ("pass", "fail"):
            raise ValueError(f"decision must be 'pass' or 'fail', got {decision!r}")
        d = when or date.today()
        record = TrustGateRecord(
            timestamp=_utc_now_iso(),
            session_id=session_id,
            decision=decision,
            violations=violations or [],
        )
        path = self._session_dir(d, session_id) / "trust_gate.json"
        _atomic_write_json(path, record.to_dict())
        log.info(
            "audit: trust_gate.%s for session %s -> %s",
            decision,
            session_id,
            path,
        )
        return path

    def write_token_map(
        self,
        *,
        session_id: str,
        token_map: dict[str, str],
        when: date | None = None,
    ) -> Path:
        """Persist `token_map.json` (placeholder → original) and chmod 0600.

        Stays ON-PREM. The map is the doctor-side reverse-mapping that
        Phase 3 will surface in the UI; it must NEVER cross the trust
        boundary into the cloud.
        """
        d = when or date.today()
        path = self._session_dir(d, session_id) / "token_map.json"
        _atomic_write_json(path, token_map)
        # chmod 0600: only the file owner reads/writes. POSIX-only
        # semantics; on Windows this is a no-op via Python (real ACL
        # work would be needed for prod Windows deploys, deferred).
        try:
            os.chmod(path, 0o600)
        except OSError as exc:  # pragma: no cover — defensive
            log.warning("could not chmod 0600 on %s: %s", path, exc)
        log.info(
            "audit: wrote token_map (%d entries) for session %s -> %s",
            len(token_map),
            session_id,
            path,
        )
        return path

    # ----- internals -------------------------------------------------------

    def _session_dir(self, when: date, session_id: str) -> Path:
        """`audit_root/<YYYY-MM-DD>/<session_id>/` — created on first use."""
        return self._audit_root / when.isoformat() / session_id
