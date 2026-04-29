"""Tests for `cloud.nfz.reimbursement` — string normalization helpers.

Pure unit tests, no fixtures, no I/O.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from cloud.nfz.reimbursement import parse_copay_pln, parse_reimbursement_string
from shared.schemas import ReimbursementLevel

# ---------------------------------------------------------------------------
# parse_reimbursement_string
# ---------------------------------------------------------------------------


class TestParseReimbursementString:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("100%", ReimbursementLevel.full),
            ("bezpłatny", ReimbursementLevel.full),
            ("bezpłatne", ReimbursementLevel.full),
            ("BEZPŁATNY", ReimbursementLevel.full),  # case-insensitive
            # Patient pays 30% means refund is 70% — both spellings map to standard.
            ("30%", ReimbursementLevel.standard),
            ("70%", ReimbursementLevel.standard),
            ("50%", ReimbursementLevel.half),
            ("ryczałt", ReimbursementLevel.flat),
            ("ryczałt 3,20 zł", ReimbursementLevel.flat),
            ("Ryczałt", ReimbursementLevel.flat),
            ("odpłatność pełna", ReimbursementLevel.none),
            ("pełnopłatny", ReimbursementLevel.none),
            ("nierefundowany", ReimbursementLevel.none),
        ],
    )
    def test_canonical_strings(self, raw: str, expected: ReimbursementLevel) -> None:
        assert parse_reimbursement_string(raw) == expected

    def test_none_input_falls_back_to_none_tier(self) -> None:
        assert parse_reimbursement_string(None) == ReimbursementLevel.none

    def test_empty_string_falls_back_to_none_tier(self) -> None:
        assert parse_reimbursement_string("") == ReimbursementLevel.none
        assert parse_reimbursement_string("   ") == ReimbursementLevel.none

    def test_unknown_string_logs_and_defaults_to_none(self, caplog: pytest.LogCaptureFixture) -> None:
        """Unknown values are warned + treated as nierefundowane."""
        import logging

        with caplog.at_level(logging.WARNING):
            result = parse_reimbursement_string("jakiś nowy poziom")
        assert result == ReimbursementLevel.none
        assert any("unknown reimbursement" in msg for msg in caplog.messages)

    def test_specific_marker_wins_over_generic(self) -> None:
        """``odpłatność pełna 30%`` should be 'none', not 'standard'."""
        assert parse_reimbursement_string("odpłatność pełna 30%") == ReimbursementLevel.none


# ---------------------------------------------------------------------------
# parse_copay_pln
# ---------------------------------------------------------------------------


class TestParseCopayPln:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("3,20 zł", Decimal("3.20")),
            ("ryczałt 3,20 zł", Decimal("3.20")),
            ("0,00 zł", Decimal("0.00")),
            ("12.50 PLN", Decimal("12.50")),
            ("3,20", Decimal("3.20")),  # no currency suffix
            ("4,5 zł", Decimal("4.5")),  # one decimal
        ],
    )
    def test_extract_decimal(self, raw: str, expected: Decimal) -> None:
        assert parse_copay_pln(raw) == expected

    def test_no_digits_returns_none(self) -> None:
        assert parse_copay_pln("bezpłatny") is None
        assert parse_copay_pln("ryczałt") is None

    def test_none_input_returns_none(self) -> None:
        assert parse_copay_pln(None) is None
        assert parse_copay_pln("") is None

    def test_polish_decimal_separator_normalized(self) -> None:
        """Comma separator (Polish convention) becomes a dot."""
        assert parse_copay_pln("12,34") == Decimal("12.34")
