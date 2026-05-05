"""Gateway-level audit logger.

Writes per-visit JSON sidecars under
``application/gateway/audit/<YYYY-MM-DD>/<visit_id>/gateway_audit.json``.

Records contain operational facts only — NO PII:
  - login_attempt: username + outcome + IP (already non-PII).
  - visit_create: visit_id + session_id + specialization.
  - visit_complete: visit_id + drug-count + provenance counts (no
    drug names, no patient names).
  - failure: visit_id + error_type + redacted message.

We deliberately do not store revealed token-mapped names here. The
on-prem audit logger holds the token map (Phase 2c). Gateway audit
sticks to who/what/when at the orchestration level.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

from gateway.models import Visit
from gateway.pii_filter import strip_pii_from_string
from gateway.storage import _atomic_write_json
from shared.schemas import AgentSuggestion

log = logging.getLogger(__name__)


@dataclass
class GatewayAuditLogger:
    """File-backed audit logger.

    `controller`: org responsible for processing (per GDPR Art. 30 it's
    the data controller name). For thesis demo this is the project
    name.
    """

    audit_dir: Path
    retention_days: int = 730
    controller: str = "MedVoicePL Demo"

    def _visit_dir(self, visit_id: UUID, *, when: date | None = None) -> Path:
        when = when or datetime.now(UTC).date()
        return self.audit_dir / when.isoformat() / str(visit_id)

    def _retention_until(self) -> str:
        return (datetime.now(UTC).date() + timedelta(days=self.retention_days)).isoformat()

    def _persist(self, visit_id: UUID, filename: str, payload: dict[str, Any]) -> Path:
        target_dir = self._visit_dir(visit_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / filename
        _atomic_write_json(path, payload)
        try:
            os.chmod(path, 0o600)
        except OSError as exc:
            log.warning("chmod 0600 failed for %s: %s", path, exc)
        return path

    def log_login_attempt(
        self, *, username: str, ip: str, success: bool
    ) -> None:
        # Login attempts are auth events — keyed under a "_login" pseudo-visit.
        target_dir = self.audit_dir / datetime.now(UTC).date().isoformat() / "_login"
        target_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "event": "login_attempt",
            "timestamp": datetime.now(UTC).isoformat(),
            "username": username,
            "ip": ip,
            "success": success,
            "controller": self.controller,
            "retention_until": self._retention_until(),
        }
        # Username is by design (operator handle); stripper is a defense-in-depth check.
        record_clean = {k: (strip_pii_from_string(v) if isinstance(v, str) else v) for k, v in record.items()}
        # Append-style: one file per day, json-lines.
        log_file = target_dir / "logins.jsonl"
        with log_file.open("a", encoding="utf-8") as fh:
            import json
            fh.write(json.dumps(record_clean, ensure_ascii=False) + "\n")

    def log_visit_create(
        self, *, visit_id: UUID, session_id: UUID, specialization: str | None
    ) -> None:
        self._persist(
            visit_id,
            "visit_create.json",
            {
                "event": "visit_create",
                "timestamp": datetime.now(UTC).isoformat(),
                "visit_id": str(visit_id),
                "session_id": str(session_id),
                "specialization": specialization,
                "controller": self.controller,
                "retention_until": self._retention_until(),
            },
        )

    def log_visit_complete(
        self, visit: Visit, suggestion: AgentSuggestion
    ) -> None:
        # Aggregate stats only — drug names included (ATC code / display
        # name; ATC is opaque, name is generic medication name, not
        # patient PII). If you want drug names redacted, configure the
        # pii_filter to redact them — but they're operational data.
        provenance_count = sum(len(d.provenance) for d in suggestion.drugs)
        self._persist(
            visit.visit_id,
            "visit_complete.json",
            {
                "event": "visit_complete",
                "timestamp": datetime.now(UTC).isoformat(),
                "visit_id": str(visit.visit_id),
                "session_id": str(visit.session_id),
                "specialization": visit.specialization.value if visit.specialization else None,
                "duration_ms": (
                    int((visit.completed_at - visit.created_at).total_seconds() * 1000)
                    if visit.completed_at
                    else None
                ),
                "drug_count": len(suggestion.drugs),
                "provenance_total": provenance_count,
                "referral_count": len(suggestion.referrals),
                "test_count": len(suggestion.tests),
                "controller": self.controller,
                "retention_until": self._retention_until(),
            },
        )

    def log_failure(
        self, visit: Visit, error_type: str, message: str
    ) -> None:
        self._persist(
            visit.visit_id,
            "visit_failure.json",
            {
                "event": "visit_failure",
                "timestamp": datetime.now(UTC).isoformat(),
                "visit_id": str(visit.visit_id),
                "session_id": str(visit.session_id),
                "error_type": error_type,
                "error_message": strip_pii_from_string(message),
                "controller": self.controller,
                "retention_until": self._retention_until(),
            },
        )


__all__ = ["GatewayAuditLogger"]
