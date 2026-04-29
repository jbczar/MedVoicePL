"""Tests for `cloud.nfz.formulary_search` — `NFZIndex` + JSON loader.

Uses `_FakeNFZIndex` directly (no disk) for the lookup tests, plus a
`tmp_path`-backed test for the real loader to verify JSON round-trip.
"""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from cloud.nfz.formulary_search import (
    NFZEntry,
    _FakeNFZIndex,
    latest_index_path,
    load_nfz_index,
    normalize_drug_name,
    serialize_index,
)
from shared.schemas import ReimbursementLevel

# ---------------------------------------------------------------------------
# normalize_drug_name
# ---------------------------------------------------------------------------


class TestNormalizeDrugName:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Bisocard", "bisocard"),
            ("BISOCARD", "bisocard"),
            ("Bisocard 5 mg", "bisocard 5 mg"),
            ("Łosartan", "losartan"),  # diacritic strip
            ("Ąpolinka", "apolinka"),
            ("Eliquis", "eliquis"),
            ("  Propranolol  ", "propranolol"),
        ],
    )
    def test_normalizes(self, raw: str, expected: str) -> None:
        assert normalize_drug_name(raw) == expected


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _bisoprolol() -> NFZEntry:
    return NFZEntry(
        atc_code="C07AB07",
        substance="Bisoprolol",
        trade_names=("Bisocard", "Concor"),
        reimbursement=ReimbursementLevel.standard,
        patient_copay_pln=Decimal("4.50"),
        indication_restrictions=("niewydolność serca",),
    )


def _apixaban() -> NFZEntry:
    return NFZEntry(
        atc_code="B01AF02",
        substance="Apixaban",
        trade_names=("Eliquis",),
        reimbursement=ReimbursementLevel.standard,
        patient_copay_pln=Decimal("12.30"),
    )


def _omeprazol() -> NFZEntry:
    return NFZEntry(
        atc_code="A02BC01",
        substance="Omeprazol",
        trade_names=("Polprazol", "Helicid"),
        reimbursement=ReimbursementLevel.flat,
        patient_copay_pln=Decimal("3.20"),
    )


# ---------------------------------------------------------------------------
# _FakeNFZIndex
# ---------------------------------------------------------------------------


class TestFakeNFZIndex:
    def test_empty_index(self) -> None:
        idx = _FakeNFZIndex()
        assert len(idx) == 0
        assert idx.lookup_by_atc("C07AB07") is None
        assert idx.lookup_by_substance("bisoprolol") == []
        assert list(idx) == []

    def test_lookup_by_atc(self) -> None:
        idx = _FakeNFZIndex(entries=(_bisoprolol(), _apixaban()))
        assert idx.lookup_by_atc("C07AB07") == _bisoprolol()
        assert idx.lookup_by_atc("c07ab07") == _bisoprolol()  # case-insensitive
        assert idx.lookup_by_atc("  C07AB07  ") == _bisoprolol()  # whitespace
        assert idx.lookup_by_atc("X99XX99") is None

    def test_lookup_by_substance_case_and_diacritic_insensitive(self) -> None:
        idx = _FakeNFZIndex(entries=(_bisoprolol(),))
        assert idx.lookup_by_substance("Bisoprolol") == [_bisoprolol()]
        assert idx.lookup_by_substance("bisoprolol") == [_bisoprolol()]
        assert idx.lookup_by_substance("BISOPROLOL") == [_bisoprolol()]

    def test_lookup_by_trade_name(self) -> None:
        idx = _FakeNFZIndex(entries=(_bisoprolol(),))
        assert idx.lookup_by_trade_name("Bisocard") == [_bisoprolol()]
        assert idx.lookup_by_trade_name("Concor") == [_bisoprolol()]
        assert idx.lookup_by_trade_name("Eliquis") == []  # different drug

    def test_iter_yields_all_entries(self) -> None:
        entries = (_bisoprolol(), _apixaban(), _omeprazol())
        idx = _FakeNFZIndex(entries=entries)
        assert set(idx) == set(entries)

    def test_snapshot_date_default_is_april_2026(self) -> None:
        idx = _FakeNFZIndex()
        assert idx.snapshot_date == date(2026, 4, 1)


# ---------------------------------------------------------------------------
# JSON round-trip — serialize_index + load_nfz_index
# ---------------------------------------------------------------------------


class TestJsonRoundTrip:
    def test_round_trip_preserves_entries(self, tmp_path: Path) -> None:
        entries = [_bisoprolol(), _apixaban(), _omeprazol()]
        snapshot = date(2026, 4, 1)

        json_path = tmp_path / "index_2026-04-01.json"
        json_path.write_text(
            json.dumps(serialize_index(snapshot, entries), ensure_ascii=False),
            encoding="utf-8",
        )

        loaded = load_nfz_index(json_path)
        assert loaded.snapshot_date == snapshot
        assert len(loaded) == 3
        assert loaded.lookup_by_atc("C07AB07") == _bisoprolol()
        assert loaded.lookup_by_substance("Apixaban") == [_apixaban()]
        assert loaded.lookup_by_trade_name("Helicid") == [_omeprazol()]

    def test_load_skips_duplicate_atc_keeps_first(self, tmp_path: Path) -> None:
        """NFZ XLSX has multiple rows per ATC (different package sizes);
        loader keeps the first only.
        """
        bisoprolol_2_5mg = NFZEntry(
            atc_code="C07AB07",
            substance="Bisoprolol",
            trade_names=("Bisocard 2.5mg",),
            reimbursement=ReimbursementLevel.standard,
        )
        bisoprolol_5mg = NFZEntry(
            atc_code="C07AB07",
            substance="Bisoprolol",
            trade_names=("Bisocard 5mg",),
            reimbursement=ReimbursementLevel.standard,
        )
        snapshot = date(2026, 4, 1)
        json_path = tmp_path / "index_2026-04-01.json"
        json_path.write_text(
            json.dumps(serialize_index(snapshot, [bisoprolol_2_5mg, bisoprolol_5mg])),
            encoding="utf-8",
        )

        loaded = load_nfz_index(json_path)
        assert len(loaded) == 1
        first = loaded.lookup_by_atc("C07AB07")
        assert first is not None
        assert first.trade_names == ("Bisocard 2.5mg",)

    def test_serialize_handles_optional_copay(self) -> None:
        """`patient_copay_pln=None` round-trips as JSON `null`."""
        entry_no_copay = NFZEntry(
            atc_code="X01AB02",
            substance="Foo",
            patient_copay_pln=None,
        )
        payload = serialize_index(date(2026, 4, 1), [entry_no_copay])
        assert payload["entries"][0]["patient_copay_pln"] is None


# ---------------------------------------------------------------------------
# latest_index_path
# ---------------------------------------------------------------------------


class TestLatestIndexPath:
    def test_returns_newest(self, tmp_path: Path) -> None:
        old = tmp_path / "index_2026-02-01.json"
        new = tmp_path / "index_2026-04-01.json"
        old.write_text("{}")
        new.write_text("{}")
        assert latest_index_path(tmp_path) == new

    def test_returns_none_when_empty(self, tmp_path: Path) -> None:
        assert latest_index_path(tmp_path) is None

    def test_returns_none_when_dir_missing(self, tmp_path: Path) -> None:
        assert latest_index_path(tmp_path / "does-not-exist") is None

    def test_ignores_unrelated_json(self, tmp_path: Path) -> None:
        (tmp_path / "manifest.json").write_text("{}")
        (tmp_path / "index_2026-04-01.json").write_text("{}")
        result = latest_index_path(tmp_path)
        assert result is not None
        assert result.name == "index_2026-04-01.json"
