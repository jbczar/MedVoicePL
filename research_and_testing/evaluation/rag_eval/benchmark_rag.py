"""Per-scenario eval runner — one row per scenario, summary aggregate."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean
from typing import Any

from evaluation.rag_eval.suggestion_accuracy import (
    GroundTruth,
    drug_precision,
    drug_recall,
    empty_list_correctness,
    nfz_refund_accuracy,
    provenance_validity,
)


@dataclass
class EvalRow:
    scenario_id: str
    drug_recall: float
    drug_precision: float
    nfz_accuracy: bool
    empty_list_correct: bool
    provenance_valid: bool
    duration_s: float
    num_drugs_suggested: int
    error: str | None = None


@dataclass
class EvalSummary:
    rows: list[EvalRow] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.rows)

    @property
    def n_errors(self) -> int:
        return sum(1 for r in self.rows if r.error)

    def mean_recall(self) -> float:
        valid = [r.drug_recall for r in self.rows if r.error is None]
        return fmean(valid) if valid else 0.0

    def mean_precision(self) -> float:
        valid = [r.drug_precision for r in self.rows if r.error is None]
        return fmean(valid) if valid else 0.0

    def nfz_accuracy_rate(self) -> float:
        valid = [r.nfz_accuracy for r in self.rows if r.error is None]
        return sum(1 for x in valid if x) / len(valid) if valid else 0.0

    def empty_list_correctness_rate(self) -> float:
        valid = [r.empty_list_correct for r in self.rows if r.error is None]
        return sum(1 for x in valid if x) / len(valid) if valid else 0.0

    def provenance_validity_rate(self) -> float:
        valid = [r.provenance_valid for r in self.rows if r.error is None]
        return sum(1 for x in valid if x) / len(valid) if valid else 0.0

    def mean_duration_s(self) -> float:
        valid = [r.duration_s for r in self.rows if r.error is None]
        return fmean(valid) if valid else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "n_errors": self.n_errors,
            "mean_drug_recall": self.mean_recall(),
            "mean_drug_precision": self.mean_precision(),
            "nfz_accuracy_rate": self.nfz_accuracy_rate(),
            "empty_list_correctness_rate": self.empty_list_correctness_rate(),
            "provenance_validity_rate": self.provenance_validity_rate(),
            "mean_duration_s": self.mean_duration_s(),
            "rows": [r.__dict__ for r in self.rows],
        }


def evaluate_one(
    *,
    scenario_id: str,
    suggestion: Any,  # cloud.shared.AgentSuggestion (avoid hard import)
    expected: GroundTruth,
) -> EvalRow:
    """Compute metrics for one scenario."""
    suggestion_atc = [d.atc_code for d in suggestion.drugs]
    suggestion_pairs = [
        (d.atc_code, d.reimbursement.value if hasattr(d.reimbursement, "value") else str(d.reimbursement))
        for d in suggestion.drugs
    ]
    return EvalRow(
        scenario_id=scenario_id,
        drug_recall=drug_recall(suggestion_atc, expected),
        drug_precision=drug_precision(suggestion_atc, expected),
        nfz_accuracy=nfz_refund_accuracy(suggestion_pairs, expected=expected),
        empty_list_correct=empty_list_correctness(len(suggestion.drugs), expected),
        provenance_valid=provenance_validity([list(d.provenance) for d in suggestion.drugs]),
        duration_s=float(suggestion.metadata.duration_s) if hasattr(suggestion, "metadata") else 0.0,
        num_drugs_suggested=len(suggestion.drugs),
    )


def summarize(rows: list[EvalRow]) -> EvalSummary:
    return EvalSummary(rows=list(rows))
