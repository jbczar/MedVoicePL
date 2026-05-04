"""Tests for `MedicalEponymFilter` — vetoer for clinical eponym PERSON spans."""

from __future__ import annotations

from pathlib import Path

import pytest
from presidio_analyzer import RecognizerResult

from on_premise.anonymization.eponym_filter import (
    MedicalEponymFilter,
    _is_eponym,
    _normalize,
    load_default,
)


@pytest.fixture
def filter_with_common_eponyms() -> MedicalEponymFilter:
    return MedicalEponymFilter(
        eponyms=frozenset(
            {"gravesa", "basedowa", "parkinsona", "hashimoto", "downa"}
        )
    )


def _person(text: str, start: int, end: int) -> RecognizerResult:
    return RecognizerResult(
        entity_type="PERSON", start=start, end=end, score=0.85
    )


class TestNormalize:
    def test_lowercase(self) -> None:
        assert _normalize("Gravesa") == "gravesa"

    def test_strip_polish_diacritics(self) -> None:
        assert _normalize("Ménière'a") == "menierea"

    def test_strip_apostrophes(self) -> None:
        assert _normalize("Whipple'a") == "whipplea"
        assert _normalize("Whippleʼa") == "whipplea"

    def test_polish_letters(self) -> None:
        assert _normalize("Leśniowskiego") == "lesniowskiego"


class TestIsEponym:
    def test_direct_match(self) -> None:
        assert _is_eponym("Gravesa", frozenset({"gravesa"}))

    def test_hyphenated_full_match(self) -> None:
        assert _is_eponym(
            "Gravesa-Basedowa", frozenset({"gravesa-basedowa"})
        )

    def test_hyphenated_partial_match(self) -> None:
        # Either part being in dict suffices.
        assert _is_eponym(
            "Gravesa-Basedowa", frozenset({"basedowa"})
        )

    def test_no_match(self) -> None:
        assert not _is_eponym("Kowalski", frozenset({"gravesa"}))

    def test_empty(self) -> None:
        assert not _is_eponym("", frozenset({"gravesa"}))


class TestEponymFilter:
    def test_drops_disease_eponym_with_context(
        self, filter_with_common_eponyms: MedicalEponymFilter
    ) -> None:
        text = "Pacjent ma chorobę Parkinsona od kilku lat."
        # Span "Parkinsona" at indices...
        start = text.index("Parkinsona")
        end = start + len("Parkinsona")
        results = [_person("Parkinsona", start, end)]
        filtered = filter_with_common_eponyms.filter(results, text=text)
        assert filtered == []

    def test_keeps_eponym_without_clinical_context(
        self, filter_with_common_eponyms: MedicalEponymFilter
    ) -> None:
        # "Parkinsona" without "choroba" — could be a real surname.
        text = "Spotkałem pana Parkinsona wczoraj."
        start = text.index("Parkinsona")
        end = start + len("Parkinsona")
        results = [_person("Parkinsona", start, end)]
        filtered = filter_with_common_eponyms.filter(results, text=text)
        assert len(filtered) == 1  # NOT vetoed — needs anonymization

    def test_drops_hyphenated_eponym(
        self, filter_with_common_eponyms: MedicalEponymFilter
    ) -> None:
        text = "podejrzenie choroby Gravesa-Basedowa z objawami ocznymi"
        start = text.index("Gravesa-Basedowa")
        end = start + len("Gravesa-Basedowa")
        results = [_person("Gravesa-Basedowa", start, end)]
        filtered = filter_with_common_eponyms.filter(results, text=text)
        assert filtered == []

    def test_drops_only_person_type(
        self, filter_with_common_eponyms: MedicalEponymFilter
    ) -> None:
        # PESEL detection should be unaffected.
        text = "PESEL pacjenta z chorobą Parkinsona to 12345678901"
        results = [
            RecognizerResult(entity_type="PESEL", start=42, end=53, score=0.9),
            _person("Parkinsona", text.index("Parkinsona"), text.index("Parkinsona") + len("Parkinsona")),
        ]
        filtered = filter_with_common_eponyms.filter(results, text=text)
        assert len(filtered) == 1
        assert filtered[0].entity_type == "PESEL"

    def test_zespol_trigger(
        self, filter_with_common_eponyms: MedicalEponymFilter
    ) -> None:
        text = "Zespół Downa rozpoznany w wieku noworodkowym."
        start = text.index("Downa")
        end = start + len("Downa")
        results = [_person("Downa", start, end)]
        filtered = filter_with_common_eponyms.filter(results, text=text)
        assert filtered == []

    def test_inflection_chorobe(
        self, filter_with_common_eponyms: MedicalEponymFilter
    ) -> None:
        text = "Rozpoznano chorobę Hashimoto u 45-letniej kobiety."
        start = text.index("Hashimoto")
        end = start + len("Hashimoto")
        results = [_person("Hashimoto", start, end)]
        filtered = filter_with_common_eponyms.filter(results, text=text)
        assert filtered == []

    def test_empty_filter_passes_through(self) -> None:
        f = MedicalEponymFilter.empty()
        results = [_person("Kowalski", 0, 8)]
        assert f.filter(results, text="Kowalski wszedł") == results

    def test_unknown_eponym_not_dropped(
        self, filter_with_common_eponyms: MedicalEponymFilter
    ) -> None:
        # "Kowalski" in "choroba Kowalskiego" — Kowalski is NOT in dict.
        text = "choroba Kowalskiego — pacjent fikcyjny"
        results = [_person("Kowalskiego", 8, 19)]
        filtered = filter_with_common_eponyms.filter(results, text=text)
        # Filter does NOT veto — fall-through to redaction.
        assert len(filtered) == 1


class TestFromDict:
    def test_loads_grouped_dict(self) -> None:
        raw = {
            "_comment": "ignored",
            "diseases": ["Parkinsona", "Gravesa"],
            "syndromes": ["Downa"],
        }
        f = MedicalEponymFilter.from_dict(raw)
        assert "parkinsona" in f.eponyms
        assert "gravesa" in f.eponyms
        assert "downa" in f.eponyms

    def test_empty_dict(self) -> None:
        f = MedicalEponymFilter.from_dict({})
        assert f.eponyms == frozenset()

    def test_skips_meta_keys(self) -> None:
        raw = {"_comment": ["should not be loaded"], "diseases": ["Parkinsona"]}
        f = MedicalEponymFilter.from_dict(raw)
        assert f.eponyms == frozenset({"parkinsona"})


class TestLoadDefault:
    def test_loads_bundled_dictionary(self) -> None:
        f = load_default()
        # Should have at least the key Phase-3a-relevant eponyms.
        assert "parkinsona" in f.eponyms
        assert "gravesa" in f.eponyms or "gravesa-basedowa" in f.eponyms

    def test_missing_path_returns_empty(self, tmp_path: Path) -> None:
        f = load_default(tmp_path / "nonexistent.json")
        assert f.eponyms == frozenset()
