"""Tests for `cloud.nfz.drug_matcher` — multi-strategy candidate resolution.

Covers all four strategies declared in `match_candidate`:
  1. ATC exact match (cheapest, most specific)
  2. Substance exact match (after Polish-normalised casefold)
  3. Trade-name lookup (`"Eliquis"` → apixaban entry)
  4. Stem-prefix fallback for Polish inflection
"""

from __future__ import annotations

from decimal import Decimal

from cloud.nfz.drug_matcher import match_candidate
from cloud.nfz.formulary_search import NFZEntry, _FakeNFZIndex
from shared.schemas import ReimbursementLevel


def _bisoprolol() -> NFZEntry:
    return NFZEntry(
        atc_code="C07AB07",
        substance="Bisoprolol",
        trade_names=("Bisocard", "Concor"),
        reimbursement=ReimbursementLevel.standard,
        patient_copay_pln=Decimal("4.50"),
    )


def _apixaban() -> NFZEntry:
    return NFZEntry(
        atc_code="B01AF02",
        substance="Apixaban",
        trade_names=("Eliquis",),
        reimbursement=ReimbursementLevel.standard,
    )


def _propranolol() -> NFZEntry:
    return NFZEntry(
        atc_code="C07AA05",
        substance="Propranolol",
        trade_names=("Propranolol WZF",),
        reimbursement=ReimbursementLevel.standard,
    )


def _full_index() -> _FakeNFZIndex:
    return _FakeNFZIndex(entries=(_bisoprolol(), _apixaban(), _propranolol()))


# ---------------------------------------------------------------------------
# Strategy 1 — ATC exact match.
# ---------------------------------------------------------------------------


class TestATCExactMatch:
    def test_atc_match_wins(self) -> None:
        result = match_candidate(name="Bisoprolol", atc_code="C07AB07", index=_full_index())
        assert result is not None
        assert result.entry == _bisoprolol()
        assert result.strategy == "atc"

    def test_atc_normalised_to_uppercase(self) -> None:
        result = match_candidate(name="bisoprolol", atc_code="c07ab07", index=_full_index())
        assert result is not None
        assert result.strategy == "atc"

    def test_hallucinated_atc_falls_through_to_substance(self) -> None:
        """LLM gave a wrong ATC; we recover via substance match."""
        result = match_candidate(name="Bisoprolol", atc_code="X99XX99", index=_full_index())
        assert result is not None
        assert result.entry == _bisoprolol()
        assert result.strategy == "substance"


# ---------------------------------------------------------------------------
# Strategy 2 — exact substance match.
# ---------------------------------------------------------------------------


class TestSubstanceExactMatch:
    def test_no_atc_resolves_by_substance(self) -> None:
        result = match_candidate(name="Apixaban", atc_code=None, index=_full_index())
        assert result is not None
        assert result.entry == _apixaban()
        assert result.strategy == "substance"

    def test_substance_case_insensitive(self) -> None:
        result = match_candidate(name="APIXABAN", atc_code=None, index=_full_index())
        assert result is not None
        assert result.strategy == "substance"

    def test_substance_diacritic_insensitive(self) -> None:
        idx = _FakeNFZIndex(
            entries=(
                NFZEntry(
                    atc_code="N02BE01",
                    substance="Łomekoksyb",  # synthetic — diacritic test only
                    reimbursement=ReimbursementLevel.standard,
                ),
            )
        )
        result = match_candidate(name="lomekoksyb", atc_code=None, index=idx)
        assert result is not None
        assert result.strategy == "substance"


# ---------------------------------------------------------------------------
# Strategy 3 — trade-name lookup.
# ---------------------------------------------------------------------------


class TestTradeNameMatch:
    def test_trade_name_resolves_to_substance(self) -> None:
        result = match_candidate(name="Eliquis", atc_code=None, index=_full_index())
        assert result is not None
        assert result.entry == _apixaban()
        assert result.strategy == "trade_name"

    def test_trade_name_case_insensitive(self) -> None:
        result = match_candidate(name="ELIQUIS", atc_code=None, index=_full_index())
        assert result is not None
        assert result.strategy == "trade_name"


# ---------------------------------------------------------------------------
# Strategy 4 — stem-prefix fallback for Polish inflection.
# ---------------------------------------------------------------------------


class TestStemFallback:
    def test_inflected_form_resolves(self) -> None:
        """Polish genitive ``"propranololem"`` should still find Propranolol."""
        result = match_candidate(name="propranololem", atc_code=None, index=_full_index())
        assert result is not None
        assert result.entry == _propranolol()
        assert result.strategy == "stem"

    def test_short_name_does_not_stem(self) -> None:
        """3-char input is too short for a meaningful stem match."""
        result = match_candidate(name="Lek", atc_code=None, index=_full_index())
        assert result is None


# ---------------------------------------------------------------------------
# Total miss.
# ---------------------------------------------------------------------------


class TestTotalMiss:
    def test_unknown_drug_returns_none(self) -> None:
        result = match_candidate(name="Krasnoludek", atc_code=None, index=_full_index())
        assert result is None

    def test_unknown_drug_with_unknown_atc(self) -> None:
        result = match_candidate(
            name="Krasnoludek", atc_code="X99XX99", index=_full_index()
        )
        assert result is None

    def test_empty_name_returns_none(self) -> None:
        result = match_candidate(name="", atc_code="C07AB07", index=_full_index())
        # ATC strategy DOES match on the index, but only if name is not pre-empty:
        # check current behaviour — name="" returns None to be safe.
        assert result is None

    def test_whitespace_only_name_returns_none(self) -> None:
        result = match_candidate(name="   ", atc_code=None, index=_full_index())
        assert result is None


# ---------------------------------------------------------------------------
# Strategy ordering — most specific wins.
# ---------------------------------------------------------------------------


class TestStrategyOrdering:
    def test_atc_wins_over_substance_when_both_present(self) -> None:
        """Even with a name-conflict edge case, ATC should win first."""
        result = match_candidate(name="Apixaban", atc_code="C07AB07", index=_full_index())
        # ATC `C07AB07` is bisoprolol — that wins, not the apixaban substance.
        assert result is not None
        assert result.entry == _bisoprolol()
        assert result.strategy == "atc"
