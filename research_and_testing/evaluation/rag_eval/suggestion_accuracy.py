"""Pure-Python metrics for the cloud agent's drug suggestions.

Mirrors `wer_calculator` / `der_calculator` posture: no third-party deps
beyond Pydantic (already in the workspace).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExpectedDrug:
    name: str
    atc: str
    must_be_refundable: bool = True

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ExpectedDrug:
        return cls(
            name=str(raw["name"]),
            atc=str(raw["atc"]).upper(),
            must_be_refundable=bool(raw.get("must_be_refundable", True)),
        )


@dataclass(frozen=True)
class DrugSlot:
    """A clinical 'slot' the suggestion must fill — e.g. 'antikoagulant in
    AFib' has multiple acceptable drugs (apiksaban OR rywaroksaban OR
    dabigatran). Recall counts a slot as fulfilled when ANY of its
    `acceptable` ATCs appears in the suggestion.
    """

    label: str
    acceptable: tuple[str, ...]  # ATC codes — any one fulfills the slot

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> DrugSlot:
        return cls(
            label=str(raw.get("label", "")),
            acceptable=tuple(str(a).upper() for a in raw.get("acceptable", [])),
        )


@dataclass(frozen=True)
class GroundTruth:
    """Hand-curated reference for one scenario.

    Two ways to express expected drugs:
    * `expected_drugs` (legacy) — list of single drugs, all required.
    * `expected_slots` (new) — list of `DrugSlot`s, each slot has multiple
      acceptable drugs (any one fulfills it).

    Use `expected_slots` for scenarios where multiple drugs in the same
    class are clinically equivalent (PPI, DOAC, BB-selective, etc.).
    """

    scenario_id: str
    diagnostic_only: bool
    expected_empty_drug_list: bool
    expected_drugs: tuple[ExpectedDrug, ...] = ()
    expected_slots: tuple[DrugSlot, ...] = ()
    expected_referrals: tuple[str, ...] = ()
    expected_tests: tuple[str, ...] = ()
    diagnosis_label: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> GroundTruth:
        return cls(
            scenario_id=str(raw["scenario_id"]),
            diagnostic_only=bool(raw.get("diagnostic_only", False)),
            expected_empty_drug_list=bool(raw.get("expected_empty_drug_list", False)),
            expected_drugs=tuple(
                ExpectedDrug.from_dict(d) for d in raw.get("expected_drugs", [])
            ),
            expected_slots=tuple(
                DrugSlot.from_dict(s) for s in raw.get("expected_slots", [])
            ),
            expected_referrals=tuple(raw.get("expected_referrals", []) or []),
            expected_tests=tuple(raw.get("expected_tests", []) or []),
            diagnosis_label=str(raw.get("diagnosis_label", "")),
        )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def drug_recall(suggestion_atc: list[str], expected: GroundTruth) -> float:
    """Slot-aware recall.

    * If `expected.expected_slots` is set, returns the fraction of slots
      fulfilled (a slot is fulfilled when ANY of its `acceptable` ATCs
      appears in the suggestion). This is the right semantics when the
      ground truth allows clinically equivalent alternatives.
    * Else falls back to legacy strict match against `expected_drugs`.
    * Returns 1.0 when nothing was expected.
    """
    suggested = {a.upper() for a in suggestion_atc}
    if expected.expected_slots:
        fulfilled = sum(
            1 for slot in expected.expected_slots
            if any(atc in suggested for atc in slot.acceptable)
        )
        return fulfilled / len(expected.expected_slots)
    if not expected.expected_drugs:
        return 1.0
    matched = sum(1 for d in expected.expected_drugs if d.atc.upper() in suggested)
    return matched / len(expected.expected_drugs)


def drug_precision(suggestion_atc: list[str], expected: GroundTruth) -> float:
    """Slot-aware precision.

    * With `expected_slots`: a suggestion is "correct" if it appears in
      ANY slot's `acceptable` list. Suggestions outside all slots count
      against precision.
    * Falls back to strict legacy match against `expected_drugs`.
    * Returns 1.0 on empty suggestion (nothing wrong was suggested).
    """
    if not suggestion_atc:
        return 1.0
    if expected.expected_slots:
        all_acceptable = {
            atc.upper()
            for slot in expected.expected_slots
            for atc in slot.acceptable
        }
        matched = sum(1 for atc in suggestion_atc if atc.upper() in all_acceptable)
        return matched / len(suggestion_atc)
    expected_set = {d.atc.upper() for d in expected.expected_drugs}
    if not expected_set:
        # Nothing expected, but something suggested — over-suggestion.
        return 0.0
    matched = sum(1 for atc in suggestion_atc if atc.upper() in expected_set)
    return matched / len(suggestion_atc)


def nfz_refund_accuracy(
    suggestion_pairs: list[tuple[str, str]],
    expected: GroundTruth | None = None,
) -> bool:
    """For each suggested drug we received `(atc, refund_tier)`. Accuracy
    here means: every drug that has a known NFZ tier reports a non-`none`
    tier — UNLESS the drug is in `expected.expected_slots` (clinically
    valid even if not currently on the wykaz, e.g. Tiamazol in some
    NFZ snapshots).

    Returns True when:
    * suggestion list is empty, OR
    * every suggested drug has tier ≠ `nierefundowane` OR is in an
      expected slot (acknowledging genuine clinical necessity even
      without refundacja in this NFZ edition).
    """
    if not suggestion_pairs:
        return True
    expected_atcs: set[str] = set()
    if expected is not None and expected.expected_slots:
        expected_atcs = {
            atc.upper()
            for slot in expected.expected_slots
            for atc in slot.acceptable
        }
    for atc, tier in suggestion_pairs:
        if tier != "nierefundowane":
            continue
        # Tier is nierefundowane — pass only if it's a clinically
        # expected drug (not a hallucinated one).
        if atc.upper() not in expected_atcs:
            return False
    return True


def empty_list_correctness(num_drugs: int, expected: GroundTruth) -> bool:
    """Was the empty/non-empty decision correct?"""
    if expected.expected_empty_drug_list:
        return num_drugs == 0
    return num_drugs > 0


def provenance_validity(provenance_lists: list[list[str]]) -> bool:
    """Every drug must have ≥1 provenance pointer."""
    if not provenance_lists:
        return True
    return all(len(p) >= 1 for p in provenance_lists)
