"""Normalize NFZ XLSX cell strings into the canonical `ReimbursementLevel`.

The Polish refundacja list (`Obwieszczenie Ministra Zdrowia w sprawie
wykazu leków refundowanych`) ships as XLSX with a free-text "Poziom
odpłatności" column. Real-world values seen in 2025-2026 issues:

* ``"bezpłatny"``           → ``ReimbursementLevel.full``
* ``"100%"``                → ``ReimbursementLevel.full``
* ``"30%"``  (patient pays 30%, refund is 70%) → ``ReimbursementLevel.standard``
* ``"50%"``                 → ``ReimbursementLevel.half``
* ``"ryczałt"``             → ``ReimbursementLevel.flat``
* ``"ryczałt 3,20 zł"``     → ``ReimbursementLevel.flat`` + parsed copay
* ``"odpłatność pełna"``    → ``ReimbursementLevel.none``
* ``""`` / ``None``         → ``ReimbursementLevel.none``

The cell-string side is messy on purpose — sheets across releases use
different capitalisation, comma-vs-dot decimals, and Polish prefix
words (``"do "``, ``"od "``). This module is the only place that
normalisation knows about. Everything downstream reasons in terms of
`ReimbursementLevel`.
"""

from __future__ import annotations

import logging
import re
from decimal import Decimal

from shared.schemas import ReimbursementLevel

log = logging.getLogger(__name__)


# Cell-string fragments that map to each tier. Longest/most-specific
# matched first (see `parse_reimbursement_string` ordering).
_FULL_REFUND_MARKERS = (
    "bezpłatn",  # bezpłatny / bezpłatne
    "100%",
)

_STANDARD_REFUND_MARKERS = (
    "30%",  # patient pays 30% → 70% refund == ReimbursementLevel.standard
    "70%",
)

_HALF_REFUND_MARKERS = ("50%",)

_FLAT_MARKERS = ("ryczałt",)

_NONE_MARKERS = (
    "odpłatność pełna",
    "pełnopłatn",
    "nierefundowan",
)

# `ryczałt 3,20 zł` / `ryczałt 3.20 PLN` / `3,20 zł` — extract decimal copay.
_COPAY_RE = re.compile(
    r"(\d+(?:[,.]\d{1,2})?)\s*(?:zł|pln|złotych)?",
    re.IGNORECASE,
)


def parse_reimbursement_string(raw: str | None) -> ReimbursementLevel:
    """Map an NFZ cell value to a canonical ``ReimbursementLevel``.

    Defensive: ``None``, empty, or unrecognised strings fall back to
    ``ReimbursementLevel.none`` (patient pays full price). Logs a
    warning on unrecognised input so we can spot schema drift.
    """
    if raw is None:
        return ReimbursementLevel.none
    text = raw.strip().lower()
    if not text:
        return ReimbursementLevel.none

    # Order matters: check most specific markers first. ``"odpłatność
    # pełna"`` would otherwise be missed if we matched ``"30%"`` against
    # a string that also contains it (none observed, but defence in
    # depth).
    for marker in _NONE_MARKERS:
        if marker in text:
            return ReimbursementLevel.none
    for marker in _FULL_REFUND_MARKERS:
        if marker in text:
            return ReimbursementLevel.full
    for marker in _FLAT_MARKERS:
        if marker in text:
            return ReimbursementLevel.flat
    for marker in _HALF_REFUND_MARKERS:
        if marker in text:
            return ReimbursementLevel.half
    for marker in _STANDARD_REFUND_MARKERS:
        if marker in text:
            return ReimbursementLevel.standard

    log.warning("unknown reimbursement string %r — defaulting to 'none'", raw)
    return ReimbursementLevel.none


def parse_copay_pln(raw: str | None) -> Decimal | None:
    """Extract the patient copay in PLN from an NFZ cell, if present.

    Returns ``None`` when no decimal-shaped run of digits is found. The
    NFZ list uses Polish convention (comma decimal separator) — we
    normalise to dot before parsing to ``Decimal``.

    Examples::

        parse_copay_pln("ryczałt 3,20 zł")  -> Decimal("3.20")
        parse_copay_pln("0,00 zł")          -> Decimal("0.00")
        parse_copay_pln("bezpłatny")        -> None
        parse_copay_pln("")                 -> None
    """
    if not raw:
        return None
    match = _COPAY_RE.search(raw)
    if not match:
        return None
    value = match.group(1).replace(",", ".")
    try:
        return Decimal(value)
    except (ValueError, ArithmeticError):  # pragma: no cover — regex guarantees digits
        return None
