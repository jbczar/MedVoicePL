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
class GroundTruth:
    """Hand-curated reference for one scenario."""

    scenario_id: str
    diagnostic_only: bool
    expected_empty_drug_list: bool
    expected_drugs: tuple[ExpectedDrug, ...]
    expected_referrals: tuple[str, ...]
    expected_tests: tuple[str, ...]
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
            expected_referrals=tuple(raw.get("expected_referrals", []) or []),
            expected_tests=tuple(raw.get("expected_tests", []) or []),
            diagnosis_label=str(raw.get("diagnosis_label", "")),
        )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def drug_recall(suggestion_atc: list[str], expected: GroundTruth) -> float:
    """Fraction of `expected.expected_drugs` whose ATC code appears in
    `suggestion_atc`. Returns 1.0 when nothing was expected.
    """
    if not expected.expected_drugs:
        return 1.0
    suggested = {a.upper() for a in suggestion_atc}
    matched = sum(1 for d in expected.expected_drugs if d.atc.upper() in suggested)
    return matched / len(expected.expected_drugs)


def drug_precision(suggestion_atc: list[str], expected: GroundTruth) -> float:
    """Fraction of suggested ATCs that are in the expected list.

    Returns 1.0 on empty suggestion (nothing wrong was suggested).
    """
    if not suggestion_atc:
        return 1.0
    expected_set = {d.atc.upper() for d in expected.expected_drugs}
    if not expected_set:
        # Nothing expected, but something suggested — over-suggestion.
        return 0.0
    matched = sum(1 for atc in suggestion_atc if atc.upper() in expected_set)
    return matched / len(suggestion_atc)


def nfz_refund_accuracy(suggestion_pairs: list[tuple[str, str]]) -> bool:
    """For each suggested drug we received `(atc, refund_tier)`. Accuracy
    here means: every drug that has a known NFZ tier reports a non-`none`
    tier. (We can't independently verify the tier without a second NFZ
    index in the eval — that responsibility lives in the agent's
    `enrich_from_nfz` node and is sanity-checked at agent runtime.)

    Returns True when every suggested drug has tier ≠ `nierefundowane`,
    or when the suggestion list is empty.
    """
    if not suggestion_pairs:
        return True
    return all(tier != "nierefundowane" for _, tier in suggestion_pairs)


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
