"""Tests for GatewayAuditLogger (Step 8)."""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from uuid import uuid4

from gateway.audit import GatewayAuditLogger
from gateway.models import Visit
from shared.schemas import AgentSuggestion, Specialization, VisitStatus

FIXTURES = Path(__file__).parent / "fixtures"


def _load_suggestion() -> AgentSuggestion:
    return AgentSuggestion.model_validate_json(
        (FIXTURES / "agent_suggestion_endo_gra.json").read_text()
    )


def _make_visit() -> Visit:
    now = datetime.now(UTC)
    return Visit(
        visit_id=uuid4(),
        session_id=uuid4(),
        status=VisitStatus.completed,
        specialization=Specialization.endokrynologia,
        visit_date=date(2026, 5, 5),
        created_at=now,
        completed_at=now,
    )


def test_log_visit_complete_writes_json_no_pii(tmp_path: Path) -> None:
    audit = GatewayAuditLogger(audit_dir=tmp_path)
    visit = _make_visit()
    audit.log_visit_complete(visit, _load_suggestion())

    candidates = list(tmp_path.rglob("visit_complete.json"))
    assert len(candidates) == 1
    payload = json.loads(candidates[0].read_text())
    assert payload["event"] == "visit_complete"
    assert payload["visit_id"] == str(visit.visit_id)
    assert payload["drug_count"] >= 1

    # No PESEL/phone leakage even if accidentally added.
    raw = candidates[0].read_text()
    assert not re.search(r"\b\d{11}\b", raw)
    assert not re.search(r"\+?\d{2}[\s-]?\d{3}[\s-]?\d{3}[\s-]?\d{3}", raw)


def test_log_failure_redacts_message(tmp_path: Path) -> None:
    audit = GatewayAuditLogger(audit_dir=tmp_path)
    visit = _make_visit()
    audit.log_failure(visit, "test_type", "patient PESEL 12345678901 in error msg")
    candidates = list(tmp_path.rglob("visit_failure.json"))
    assert len(candidates) == 1
    payload = json.loads(candidates[0].read_text())
    assert "12345678901" not in payload["error_message"]


def test_log_visit_create_persists_metadata(tmp_path: Path) -> None:
    audit = GatewayAuditLogger(audit_dir=tmp_path)
    vid = uuid4()
    sid = uuid4()
    audit.log_visit_create(visit_id=vid, session_id=sid, specialization="kardiologia")
    candidates = list(tmp_path.rglob("visit_create.json"))
    assert len(candidates) == 1
    payload = json.loads(candidates[0].read_text())
    assert payload["visit_id"] == str(vid)
    assert payload["session_id"] == str(sid)
    assert payload["specialization"] == "kardiologia"
    assert payload["controller"]
    assert payload["retention_until"]


def test_log_login_attempt_appends_jsonl(tmp_path: Path) -> None:
    audit = GatewayAuditLogger(audit_dir=tmp_path)
    audit.log_login_attempt(username="demo_doctor", ip="127.0.0.1", success=True)
    audit.log_login_attempt(username="demo_doctor", ip="127.0.0.1", success=False)
    files = list(tmp_path.rglob("logins.jsonl"))
    assert len(files) == 1
    lines = files[0].read_text().strip().split("\n")
    assert len(lines) == 2


def test_audit_files_chmod_0600(tmp_path: Path) -> None:
    if sys.platform == "win32":
        return
    audit = GatewayAuditLogger(audit_dir=tmp_path)
    audit.log_visit_complete(_make_visit(), _load_suggestion())
    file = next(tmp_path.rglob("visit_complete.json"))
    mode = os.stat(file).st_mode & 0o777
    assert mode == 0o600


def test_audit_dir_partitioned_by_date(tmp_path: Path) -> None:
    audit = GatewayAuditLogger(audit_dir=tmp_path)
    audit.log_visit_complete(_make_visit(), _load_suggestion())
    today = datetime.now(UTC).date().isoformat()
    assert (tmp_path / today).is_dir()
