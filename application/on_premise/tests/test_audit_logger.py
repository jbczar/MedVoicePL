"""Tests for `on_premise.anonymization.audit_logger`.

Covers:
  - Atomic write (no `.tmp` files left after success).
  - chmod 0600 on `token_map.json`.
  - GDPR Article 30 fields present on every audit record.
  - **No raw PII strings** appear in the audit JSON (regex sweep over
    the dumped file).
  - SHA-256 determinism.
"""

from __future__ import annotations

import json
import os
import re
import stat
import sys
from datetime import date
from pathlib import Path

import pytest

from on_premise.anonymization.audit_logger import (
    AuditLogger,
    Replacement,
    hash_pii,
)


def _replacement(
    *,
    entity_type: str = "PERSON",
    token: str = "<PERSON_1>",
    original: str = "Krzysztof Nowakowski",
) -> Replacement:
    return Replacement(
        entity_type=entity_type,
        token=token,
        original_sha256=hash_pii(original),
        location_path="subjective.history_of_present_illness",
        location_offset_start=10,
        location_offset_end=10 + len(original),
    )


# ---------------------------------------------------------------------------
# hash_pii — pure function determinism.
# ---------------------------------------------------------------------------


def test_hash_pii_is_deterministic() -> None:
    h1 = hash_pii("Krzysztof Nowakowski")
    h2 = hash_pii("Krzysztof Nowakowski")
    assert h1 == h2
    assert len(h1) == 64  # sha256 hex digest


def test_hash_pii_is_collision_resistant_on_neighbors() -> None:
    """Adjacent inputs hash to different digests (no truncation issue)."""
    assert hash_pii("Krzysztof Nowakowski") != hash_pii("Krzysztof Nowakowska")


# ---------------------------------------------------------------------------
# log_anonymization
# ---------------------------------------------------------------------------


class TestLogAnonymization:
    def test_writes_audit_json_with_records(self, tmp_path: Path) -> None:
        logger = AuditLogger(audit_root=tmp_path, retention_days=730)
        path = logger.log_anonymization(
            session_id="00000000-0000-0000-0000-000000000001",
            replacements=[_replacement()],
            soap_sha256="b" * 64,
            transcript_sha256="c" * 64,
            when=date(2026, 4, 26),
        )
        assert path.exists()
        assert path.parent.parent.name == "2026-04-26"
        assert path.parent.name == "00000000-0000-0000-0000-000000000001"

        payload = json.loads(path.read_text())
        assert payload["schema_version"] == "2c.0"
        assert len(payload["records"]) == 1
        record = payload["records"][0]
        # Ensure each replacement field landed in the record.
        assert record["entity_type"] == "PERSON"
        assert record["token"] == "<PERSON_1>"
        assert record["location_path"] == "subjective.history_of_present_illness"

    def test_gdpr_article_30_fields_present(self, tmp_path: Path) -> None:
        """Every record carries `purpose`, `legal_basis`, `data_categories`,
        `controller`, `retention_until`."""
        logger = AuditLogger(audit_root=tmp_path, retention_days=365, controller="MedVoicePL-Test")
        path = logger.log_anonymization(
            session_id="abc",
            replacements=[_replacement()],
            soap_sha256="b" * 64,
            transcript_sha256="c" * 64,
            when=date(2026, 4, 26),
        )
        record = json.loads(path.read_text())["records"][0]
        assert record["purpose"] == "anonymization"
        assert record["legal_basis"] == "Art_9_2_h"
        assert record["data_categories"] == ["PERSON"]
        assert record["controller"] == "MedVoicePL-Test"
        # retention_until = when + 365 days = 2027-04-26
        assert record["retention_until"] == "2027-04-26"

    def test_atomic_write_no_tmp_left_behind(self, tmp_path: Path) -> None:
        logger = AuditLogger(audit_root=tmp_path)
        logger.log_anonymization(
            session_id="abc",
            replacements=[_replacement()],
            soap_sha256="b" * 64,
            transcript_sha256="c" * 64,
        )
        # No `.tmp` files should remain after a successful write.
        leftovers = list(tmp_path.rglob("*.tmp"))
        assert leftovers == []

    def test_audit_json_contains_no_raw_pii(self, tmp_path: Path) -> None:
        """The audit must hold SHA-256 digests, never plaintext PII.

        SHA-256 hex digests are 64 hex chars and routinely contain
        digit runs that look phone- or PESEL-shaped on a naive regex
        sweep. The right check is: parse the JSON, look up each record
        by the keys we expect to hold PII-derived material, and confirm
        every such field is a 64-char hex digest (not the original).
        """
        original_name = "Krzysztof Nowakowski"
        original_pesel = "44051401359"
        original_phone = "501-234-567"
        replacements = [
            Replacement(
                entity_type="PERSON",
                token="<PERSON_1>",
                original_sha256=hash_pii(original_name),
                location_path="subjective.history_of_present_illness",
                location_offset_start=10,
                location_offset_end=30,
            ),
            Replacement(
                entity_type="PL_PESEL",
                token="<PL_PESEL_1>",
                original_sha256=hash_pii(original_pesel),
                location_path="subjective.history_of_present_illness",
                location_offset_start=50,
                location_offset_end=61,
            ),
            Replacement(
                entity_type="PHONE_NUMBER",
                token="<PHONE_NUMBER_1>",
                original_sha256=hash_pii(original_phone),
                location_path="plan.follow_up",
                location_offset_start=0,
                location_offset_end=11,
            ),
        ]
        logger = AuditLogger(audit_root=tmp_path)
        path = logger.log_anonymization(
            session_id="abc",
            replacements=replacements,
            soap_sha256="b" * 64,
            transcript_sha256="c" * 64,
        )
        body = path.read_text()

        # 1. None of the original plaintext PII strings appear anywhere.
        for plaintext in (original_name, original_pesel, original_phone):
            assert plaintext not in body, f"audit log leaked plaintext PII: {plaintext!r}"

        # 2. Each record's `original_sha256` is exactly 64 hex chars
        #    — confirms we stored the digest, not anything shorter that
        #    might have been the raw value.
        records = json.loads(body)["records"]
        sha_re = re.compile(r"^[0-9a-f]{64}$")
        for record in records:
            assert sha_re.match(record["original_sha256"]), (
                f"original_sha256 not a 64-char hex digest: {record}"
            )


# ---------------------------------------------------------------------------
# log_trust_gate
# ---------------------------------------------------------------------------


class TestLogTrustGate:
    def test_pass_decision_writes_record(self, tmp_path: Path) -> None:
        logger = AuditLogger(audit_root=tmp_path)
        path = logger.log_trust_gate(session_id="abc", decision="pass", when=date(2026, 4, 26))
        record = json.loads(path.read_text())
        assert record["decision"] == "pass"
        assert record["violations"] == []

    def test_fail_decision_with_violations(self, tmp_path: Path) -> None:
        logger = AuditLogger(audit_root=tmp_path)
        violations = [
            {"location": "subjective.chief_complaint", "kind": "regex", "detail": "pesel"},
        ]
        path = logger.log_trust_gate(session_id="abc", decision="fail", violations=violations)
        record = json.loads(path.read_text())
        assert record["decision"] == "fail"
        assert record["violations"] == violations

    def test_invalid_decision_raises(self, tmp_path: Path) -> None:
        logger = AuditLogger(audit_root=tmp_path)
        with pytest.raises(ValueError):
            logger.log_trust_gate(session_id="abc", decision="maybe")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# write_token_map
# ---------------------------------------------------------------------------


class TestTokenMap:
    def test_writes_token_map_json(self, tmp_path: Path) -> None:
        logger = AuditLogger(audit_root=tmp_path)
        token_map = {"<PERSON_1>": "Krzysztof Nowakowski", "<PHONE_NUMBER_1>": "501-234-567"}
        path = logger.write_token_map(session_id="abc", token_map=token_map)
        loaded = json.loads(path.read_text())
        assert loaded == token_map

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX file perms only")
    def test_token_map_has_owner_only_permissions(self, tmp_path: Path) -> None:
        logger = AuditLogger(audit_root=tmp_path)
        path = logger.write_token_map(
            session_id="abc",
            token_map={"<PERSON_1>": "Krzysztof Nowakowski"},
        )
        mode = stat.S_IMODE(os.stat(path).st_mode)
        assert mode == 0o600, f"expected 0600, got {oct(mode)}"


# ---------------------------------------------------------------------------
# Path partitioning by date.
# ---------------------------------------------------------------------------


def test_session_dir_partitioned_by_date(tmp_path: Path) -> None:
    """Two sessions on different days land in different directories."""
    logger = AuditLogger(audit_root=tmp_path)
    p1 = logger.log_anonymization(
        session_id="s1",
        replacements=[_replacement()],
        soap_sha256="b" * 64,
        transcript_sha256="c" * 64,
        when=date(2026, 4, 26),
    )
    p2 = logger.log_anonymization(
        session_id="s2",
        replacements=[_replacement()],
        soap_sha256="b" * 64,
        transcript_sha256="c" * 64,
        when=date(2026, 4, 27),
    )
    assert p1.parent.parent.name == "2026-04-26"
    assert p2.parent.parent.name == "2026-04-27"
