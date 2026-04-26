"""Unit tests for Polish PII recognizers — pure regex + checksum, no spaCy.

Tests call `recognizer.analyze(text, entities, nlp_artifacts=None)` directly,
which exercises the regex pattern and `validate_result` hook without
needing a loaded NLP model. Context-word boosting requires `nlp_artifacts`,
so context-related behavior is tested separately by inspecting the
analyzer's resulting score (no boost without artifacts).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from on_premise.anonymization.pl_recognizers import (
    PolishAddressRecognizer,
    PolishIDCardRecognizer,
    PolishNameListBooster,
    PolishNIPRecognizer,
    PolishPESELRecognizer,
    PolishPhoneRecognizer,
    _nip_checksum_ok,
    _pesel_checksum_ok,
    build_polish_recognizers,
    load_scenario_names,
)

# ---------------------------------------------------------------------------
# Checksum helpers
# ---------------------------------------------------------------------------


class TestPESELChecksum:
    """Polish PESEL checksum: weights 1,3,7,9,1,3,7,9,1,3 mod 10."""

    def test_known_valid_pesel(self) -> None:
        # 44051401359 is the canonical example from official PESEL docs.
        assert _pesel_checksum_ok("44051401359")

    def test_synthetic_pesel_from_dataset_fails_checksum(self) -> None:
        # Confirms our test-data assumption: dataset_pipeline generates
        # PESELs without bothering to pass the checksum (so they can
        # never collide with a real person). Recognizer fail-soft logic
        # (validate_result returning None instead of False) is what
        # catches these via context-word boost.
        assert not _pesel_checksum_ok("75081512345")

    def test_wrong_length_rejected(self) -> None:
        assert not _pesel_checksum_ok("4405140135")  # 10 digits
        assert not _pesel_checksum_ok("440514013590")  # 12 digits

    def test_non_digits_rejected(self) -> None:
        assert not _pesel_checksum_ok("4405140135X")
        assert not _pesel_checksum_ok("           ")


class TestNIPChecksum:
    """Polish NIP checksum: weights 6,5,7,2,3,4,5,6,7 mod 11."""

    def test_known_valid_nip(self) -> None:
        # Common valid example used in Polish NIP test fixtures.
        assert _nip_checksum_ok("5260250995")

    def test_invalid_nip(self) -> None:
        assert not _nip_checksum_ok("1234567890")

    def test_wrong_length_rejected(self) -> None:
        assert not _nip_checksum_ok("123456789")
        assert not _nip_checksum_ok("12345678901")

    # The residue-10 rule (NIPs that mathematically would have check
    # digit 10 are never issued) is implemented but hard to demonstrate
    # with a constructed example without re-deriving the checksum here.
    # The integration tests in `test_presidio_agent.py` cover the
    # end-to-end behavior on real-shaped inputs.


# ---------------------------------------------------------------------------
# PolishPESELRecognizer
# ---------------------------------------------------------------------------


class TestPESELRecognizer:
    def setup_method(self) -> None:
        self.r = PolishPESELRecognizer()

    def test_valid_checksum_high_score(self) -> None:
        results = self.r.analyze("PESEL 44051401359 to numer.", entities=["PL_PESEL"])
        assert len(results) == 1
        assert results[0].entity_type == "PL_PESEL"
        assert results[0].score >= 0.85  # checksum bump

    def test_invalid_checksum_keeps_regex_score(self) -> None:
        # Synthetic PESEL fails checksum; recognizer returns None to
        # preserve the bare regex score so context boost can rescue it.
        results = self.r.analyze("Pacjent ma 12345678901 w karcie.", entities=["PL_PESEL"])
        assert len(results) == 1
        # Regex-only score = 0.4 (below default 0.5 analyzer threshold,
        # but the recognizer itself doesn't drop it — that's the engine's job).
        assert results[0].score == pytest.approx(0.4)

    def test_no_eleven_digit_run_no_match(self) -> None:
        results = self.r.analyze("Brak liczb tutaj.", entities=["PL_PESEL"])
        assert results == []

    def test_does_not_overmatch_short_runs(self) -> None:
        # 9 digits — too short. 12 digits — too long, won't match \b\d{11}\b
        # with word boundaries.
        results = self.r.analyze("Numer 123456789 i 123456789012", entities=["PL_PESEL"])
        assert results == []


# ---------------------------------------------------------------------------
# PolishNIPRecognizer
# ---------------------------------------------------------------------------


class TestNIPRecognizer:
    def setup_method(self) -> None:
        self.r = PolishNIPRecognizer()

    def test_valid_checksum(self) -> None:
        results = self.r.analyze("NIP 5260250995", entities=["PL_NIP"])
        assert len(results) == 1
        assert results[0].score >= 0.85

    def test_invalid_checksum_preserves_regex_score(self) -> None:
        results = self.r.analyze("Numer 1234567890 w bazie.", entities=["PL_NIP"])
        assert len(results) == 1
        assert results[0].score == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# PolishPhoneRecognizer
# ---------------------------------------------------------------------------


class TestPhoneRecognizer:
    def setup_method(self) -> None:
        self.r = PolishPhoneRecognizer()

    @pytest.mark.parametrize(
        "text",
        [
            "Numer to 501-234-567.",  # mobile, dashes
            "Telefon 501 234 567.",  # mobile, spaces
            "Komórka 501234567.",  # mobile, no separators
            "Mój nr to +48 501 234 567.",  # mobile with country code
            "Stary numer +48501234567.",  # mobile, no separators with prefix
        ],
    )
    def test_mobile_variants(self, text: str) -> None:
        results = self.r.analyze(text, entities=["PHONE_NUMBER"])
        assert len(results) >= 1
        assert all(r.entity_type == "PHONE_NUMBER" for r in results)

    @pytest.mark.parametrize(
        "text",
        [
            "Centrala 71 333 44 55.",  # landline
            "Recepcja 22 555 66 77.",
            "Stary nr 71-333-44-55.",
        ],
    )
    def test_landline_variants(self, text: str) -> None:
        results = self.r.analyze(text, entities=["PHONE_NUMBER"])
        assert len(results) >= 1

    def test_dosing_string_does_not_match(self) -> None:
        # Phase-2c risk: clinical text "200 mg co 8 godzin" looks
        # numerically dense but doesn't fit phone grouping.
        results = self.r.analyze("Aspiryna 200 mg co 8 godzin.", entities=["PHONE_NUMBER"])
        # No 9-digit run with phone-shaped grouping → no match.
        assert results == []


# ---------------------------------------------------------------------------
# PolishIDCardRecognizer
# ---------------------------------------------------------------------------


class TestIDCardRecognizer:
    def setup_method(self) -> None:
        self.r = PolishIDCardRecognizer()

    def test_pattern_matches_three_letters_six_digits(self) -> None:
        results = self.r.analyze("Dowód osobisty ABC123456.", entities=["PL_ID_CARD"])
        assert len(results) == 1
        # Bare regex score is 0.3 — below threshold without context boost.
        # Context boost is the analyzer's responsibility; here we just
        # confirm the pattern fires.
        assert results[0].score == pytest.approx(0.3)

    def test_lowercase_letters_match_due_to_case_insensitive(self) -> None:
        # Presidio's PatternRecognizer sets `regex.IGNORECASE` globally,
        # so `[A-Z]{3}` actually matches lowercase too. We accept this:
        # an anonymizer over-redacting a stray "abc123456" string is
        # safer than under-redacting a real ID. The bare regex score is
        # 0.3 — below threshold — so without context boost the result
        # is dropped by the analyzer engine.
        results = self.r.analyze("Kod abc123456.", entities=["PL_ID_CARD"])
        assert len(results) == 1
        assert results[0].score == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# PolishAddressRecognizer
# ---------------------------------------------------------------------------


class TestAddressRecognizer:
    def setup_method(self) -> None:
        self.r = PolishAddressRecognizer()

    @pytest.mark.parametrize(
        "text",
        [
            "Mieszkam przy ulicy Słonecznej 12.",
            "Adres: ul. Marszałkowska 5/3.",
            "Pacjent z osiedla Ruczaj.",
            "Praktyka przy Długiej 15.",
            "Aleja Jana Pawła II 12.",
        ],
    )
    def test_street_prefix_variants(self, text: str) -> None:
        results = self.r.analyze(text, entities=["LOCATION"])
        assert len(results) >= 1
        assert results[0].entity_type == "LOCATION"

    def test_no_street_prefix_no_match(self) -> None:
        # Plain mention of city name without prefix doesn't trigger
        # this recognizer (spaCy LOC handles bare city names).
        results = self.r.analyze("Pacjent przyjechał z Warszawy.", entities=["LOCATION"])
        assert results == []


# ---------------------------------------------------------------------------
# PolishNameListBooster
# ---------------------------------------------------------------------------


class TestNameListBooster:
    def test_full_name_matches(self) -> None:
        r = PolishNameListBooster(names=["Krzysztof Nowakowski"])
        results = r.analyze("Pacjent Krzysztof Nowakowski zgłosił.", entities=["PERSON"])
        assert len(results) >= 1
        # Full-name pattern scores 1.0; the surname token also scores 0.85.
        assert max(rr.score for rr in results) == pytest.approx(1.0)

    def test_surname_token_only_matches(self) -> None:
        r = PolishNameListBooster(names=["Krzysztof Nowakowski"])
        results = r.analyze("Pan Nowakowski wszedł.", entities=["PERSON"])
        assert len(results) >= 1

    def test_short_token_filtered(self) -> None:
        # First name shorter than 3 chars wouldn't be added as a token
        # pattern. (None of our test names trip this; sanity check.)
        r = PolishNameListBooster(names=["Jo Kowalski"])
        # "Jo" is 2 chars, filtered. Full name still matches.
        results = r.analyze("Jo Kowalski zgłosił.", entities=["PERSON"])
        assert len(results) >= 1

    def test_empty_list_is_noop(self) -> None:
        r = PolishNameListBooster(names=[])
        results = r.analyze("Krzysztof Nowakowski.", entities=["PERSON"])
        assert results == []

    def test_none_names_is_noop(self) -> None:
        r = PolishNameListBooster()
        results = r.analyze("Cokolwiek.", entities=["PERSON"])
        assert results == []


# ---------------------------------------------------------------------------
# load_scenario_names
# ---------------------------------------------------------------------------


def test_load_scenario_names_handles_missing_dir(tmp_path: Path) -> None:
    """Missing dir → empty list, never raises."""
    assert load_scenario_names(tmp_path / "does_not_exist") == []


def test_load_scenario_names_reads_patient_full_name(tmp_path: Path) -> None:
    """Reads `patient_full_name` from each *.json in the dir."""
    (tmp_path / "a.json").write_text('{"patient_full_name": "Anna Nowak"}', encoding="utf-8")
    (tmp_path / "b.json").write_text(
        '{"patient_full_name": "Krzysztof Nowakowski"}', encoding="utf-8"
    )
    (tmp_path / "c.json").write_text('{"other": "field"}', encoding="utf-8")  # no name field
    names = load_scenario_names(tmp_path)
    assert sorted(names) == ["Anna Nowak", "Krzysztof Nowakowski"]


def test_load_scenario_names_skips_unparseable(tmp_path: Path) -> None:
    """Unreadable JSON → skipped, no exception."""
    (tmp_path / "good.json").write_text('{"patient_full_name": "Anna Nowak"}', encoding="utf-8")
    (tmp_path / "bad.json").write_text("not json", encoding="utf-8")
    assert load_scenario_names(tmp_path) == ["Anna Nowak"]


# ---------------------------------------------------------------------------
# build_polish_recognizers — factory smoke test.
# ---------------------------------------------------------------------------


def test_factory_returns_six_recognizers() -> None:
    recs = build_polish_recognizers()
    assert len(recs) == 6
    # Priority order matches DD-10: PESEL → NIP → ID_CARD → PHONE → ADDRESS → name list.
    assert isinstance(recs[0], PolishPESELRecognizer)
    assert isinstance(recs[1], PolishNIPRecognizer)
    assert isinstance(recs[2], PolishIDCardRecognizer)
    assert isinstance(recs[3], PolishPhoneRecognizer)
    assert isinstance(recs[4], PolishAddressRecognizer)
    assert isinstance(recs[5], PolishNameListBooster)


def test_factory_seeds_name_booster(tmp_path: Path) -> None:
    """Names supplied to the factory propagate to the booster."""
    recs = build_polish_recognizers(name_booster_names=["Anna Nowak"])
    booster = recs[-1]
    assert isinstance(booster, PolishNameListBooster)
    results = booster.analyze("Pani Anna Nowak.", entities=["PERSON"])
    assert len(results) >= 1
