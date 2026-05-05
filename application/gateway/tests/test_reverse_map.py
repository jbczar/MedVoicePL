"""Tests for token reverse-map (Step 6)."""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from uuid import uuid4

import pytest

from gateway.reverse_map import (
    TokenMapNotFoundError,
    load_token_map,
    reveal_for_doctor,
    reveal_text,
)
from shared.schemas import AgentSuggestion

FIXTURES = Path(__file__).parent / "fixtures"


def _load_suggestion() -> AgentSuggestion:
    return AgentSuggestion.model_validate_json(
        (FIXTURES / "agent_suggestion_endo_gra.json").read_text()
    )


def test_reveal_text_replaces_known_tokens() -> None:
    text = "u <PERSON_1> w lokalizacji <LOCATION_2>"
    out, unmapped = reveal_text(text, {"<PERSON_1>": "Jan Kowalski", "<LOCATION_2>": "Kraków"})
    assert out == "u Jan Kowalski w lokalizacji Kraków"
    assert unmapped == 0


def test_reveal_text_unmapped_tokens_count() -> None:
    text = "<PERSON_1> i <PERSON_5>"
    out, unmapped = reveal_text(text, {"<PERSON_1>": "Anna"})
    assert out == "Anna i <PERSON_5>"
    assert unmapped == 1


def test_reveal_text_no_tokens_passthrough() -> None:
    out, unmapped = reveal_text("brak placeholderów", {})
    assert out == "brak placeholderów"
    assert unmapped == 0


def test_reveal_text_repeated_token() -> None:
    text = "<PERSON_1> i znów <PERSON_1>"
    out, unmapped = reveal_text(text, {"<PERSON_1>": "Marek"})
    assert out == "Marek i znów Marek"
    assert unmapped == 0


def test_reveal_for_doctor_round_trip(tmp_path: Path) -> None:
    suggestion = _load_suggestion()
    # No token in suggestion (eponym filter preserves the diagnoses)
    # — call should be a no-op and produce 0 unmapped.
    revealed, unmapped = reveal_for_doctor(suggestion, token_map={})
    assert revealed == suggestion
    assert unmapped == 0


def test_reveal_for_doctor_replaces_in_nested_strings() -> None:
    suggestion = _load_suggestion()
    # Mutate one rationale string to include a placeholder.
    dumped = suggestion.model_dump(mode="json")
    dumped["drugs"][0]["rationale"] += " — pacjent <PERSON_42>"
    suggestion = AgentSuggestion.model_validate(dumped)

    revealed, unmapped = reveal_for_doctor(
        suggestion, token_map={"<PERSON_42>": "Anna Nowak"}
    )
    assert "Anna Nowak" in revealed.drugs[0].rationale
    assert "<PERSON_42>" not in revealed.drugs[0].rationale
    assert unmapped == 0


def test_reveal_for_doctor_unmapped_passes_through() -> None:
    suggestion = _load_suggestion()
    dumped = suggestion.model_dump(mode="json")
    dumped["drugs"][0]["rationale"] += " <PERSON_99>"
    suggestion = AgentSuggestion.model_validate(dumped)

    revealed, unmapped = reveal_for_doctor(suggestion, token_map={})
    assert "<PERSON_99>" in revealed.drugs[0].rationale
    assert unmapped == 1


def _write_token_map(root: Path, day: str, sid: str, payload: dict) -> Path:
    target = root / day / sid
    target.mkdir(parents=True, exist_ok=True)
    out = target / "token_map.json"
    out.write_text(json.dumps(payload), encoding="utf-8")
    os.chmod(out, 0o600)
    return out


def test_load_token_map_with_explicit_date(tmp_path: Path) -> None:
    sid = uuid4()
    _write_token_map(
        tmp_path, "2026-05-05", str(sid), {"<PERSON_1>": "Anna"}
    )
    out = load_token_map(
        audit_root=tmp_path, session_id=sid, visit_date=date(2026, 5, 5)
    )
    assert out == {"<PERSON_1>": "Anna"}


def test_load_token_map_falls_back_to_scan(tmp_path: Path) -> None:
    sid = uuid4()
    _write_token_map(tmp_path, "2026-05-04", str(sid), {"<PERSON_1>": "Bartek"})
    out = load_token_map(audit_root=tmp_path, session_id=sid)
    assert out == {"<PERSON_1>": "Bartek"}


def test_load_token_map_picks_latest_when_multiple(tmp_path: Path) -> None:
    sid = uuid4()
    _write_token_map(tmp_path, "2026-05-04", str(sid), {"<PERSON_1>": "Old"})
    _write_token_map(tmp_path, "2026-05-05", str(sid), {"<PERSON_1>": "New"})
    out = load_token_map(audit_root=tmp_path, session_id=sid)
    assert out == {"<PERSON_1>": "New"}


def test_load_token_map_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(TokenMapNotFoundError):
        load_token_map(audit_root=tmp_path, session_id=uuid4())


def test_load_token_map_corrupt_json_raises(tmp_path: Path) -> None:
    sid = uuid4()
    target = tmp_path / "2026-05-05" / str(sid) / "token_map.json"
    target.parent.mkdir(parents=True)
    target.write_text("not json", encoding="utf-8")
    with pytest.raises(TokenMapNotFoundError):
        load_token_map(audit_root=tmp_path, session_id=sid, visit_date=date(2026, 5, 5))
