"""Tests for the structlog PII-stripping processor (Step 6)."""

from __future__ import annotations

from gateway.pii_filter import REDACTED, pii_stripper, strip_pii_from_string


def test_strip_pesel() -> None:
    assert strip_pii_from_string("PESEL 12345678901 leaked") == f"PESEL {REDACTED} leaked"


def test_strip_phone_full() -> None:
    out = strip_pii_from_string("kontakt: +48 600 700 800")
    assert REDACTED in out
    assert "600" not in out


def test_strip_phone_compact() -> None:
    assert strip_pii_from_string("tel 600700800") == f"tel {REDACTED}"


def test_strip_email() -> None:
    out = strip_pii_from_string("send to test.pacjent@example.com please")
    assert out == f"send to {REDACTED} please"


def test_strip_nip() -> None:
    out = strip_pii_from_string("NIP 1234567890 invoice")
    assert REDACTED in out


def test_no_match_passthrough() -> None:
    txt = "po polsku, bez PII, tylko klinika"
    assert strip_pii_from_string(txt) == txt


def test_processor_walks_dict_values() -> None:
    event = {
        "event": "request",
        "trace_id": "abc",
        "patient_pesel": "12345678901",
        "nested": {"phone": "+48 600 700 800", "ok": "tekst"},
        "tags": ["plain", "98765432109876"],
    }
    out = pii_stripper(None, "info", event)
    assert out["patient_pesel"] == REDACTED
    assert out["nested"]["phone"] != "+48 600 700 800"
    assert out["nested"]["ok"] == "tekst"
    assert out["tags"][0] == "plain"
    assert REDACTED in out["tags"][1]


def test_processor_does_not_mutate_input() -> None:
    event = {"a": "12345678901"}
    out = pii_stripper(None, "info", event)
    assert event == {"a": "12345678901"}
    assert out == {"a": REDACTED}


def test_processor_preserves_non_string_types() -> None:
    event = {"count": 5, "ok": True, "ratio": 0.5}
    out = pii_stripper(None, "info", event)
    assert out == event


def test_pesel_regex_does_not_match_short_numbers() -> None:
    txt = "1234"
    assert strip_pii_from_string(txt) == txt
