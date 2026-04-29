"""Metric tests for `rag_eval.suggestion_accuracy` + `benchmark_rag`."""

from __future__ import annotations

from evaluation.rag_eval.benchmark_rag import EvalRow, summarize
from evaluation.rag_eval.suggestion_accuracy import (
    ExpectedDrug,
    GroundTruth,
    drug_precision,
    drug_recall,
    empty_list_correctness,
    nfz_refund_accuracy,
    provenance_validity,
)


def _gt(drugs=(), empty=False, diagnostic=False) -> GroundTruth:
    return GroundTruth(
        scenario_id="x",
        diagnostic_only=diagnostic,
        expected_empty_drug_list=empty,
        expected_drugs=tuple(drugs),
        expected_referrals=(),
        expected_tests=(),
    )


def test_recall_full_match() -> None:
    gt = _gt(
        [ExpectedDrug(name="A", atc="C07AA05"), ExpectedDrug(name="B", atc="H03BB02")]
    )
    assert drug_recall(["C07AA05", "H03BB02"], gt) == 1.0


def test_recall_partial() -> None:
    gt = _gt(
        [ExpectedDrug(name="A", atc="C07AA05"), ExpectedDrug(name="B", atc="H03BB02")]
    )
    assert drug_recall(["C07AA05"], gt) == 0.5


def test_recall_empty_expected_returns_one() -> None:
    gt = _gt([], empty=True)
    assert drug_recall([], gt) == 1.0


def test_precision_no_suggestion_returns_one() -> None:
    gt = _gt([ExpectedDrug(name="A", atc="C07AA05")])
    assert drug_precision([], gt) == 1.0


def test_precision_over_suggestion() -> None:
    gt = _gt([ExpectedDrug(name="A", atc="C07AA05")])
    assert drug_precision(["C07AA05", "X99XX99"], gt) == 0.5


def test_precision_against_empty_expected_is_zero() -> None:
    gt = _gt([], empty=True)
    assert drug_precision(["C07AA05"], gt) == 0.0


def test_nfz_refund_accuracy_all_refundable() -> None:
    assert nfz_refund_accuracy([("C07AA05", "70%"), ("H03BB02", "100%")]) is True


def test_nfz_refund_accuracy_one_unrefundable() -> None:
    assert nfz_refund_accuracy([("C07AA05", "70%"), ("X99XX99", "nierefundowane")]) is False


def test_nfz_refund_accuracy_empty_is_true() -> None:
    assert nfz_refund_accuracy([]) is True


def test_empty_list_correctness_diagnostic() -> None:
    gt = _gt([], empty=True)
    assert empty_list_correctness(0, gt) is True
    assert empty_list_correctness(1, gt) is False


def test_empty_list_correctness_treatable() -> None:
    gt = _gt([ExpectedDrug(name="A", atc="C07AA05")], empty=False)
    assert empty_list_correctness(2, gt) is True
    assert empty_list_correctness(0, gt) is False


def test_provenance_validity_all_present() -> None:
    assert provenance_validity([["a"], ["b", "c"]]) is True


def test_provenance_validity_one_missing() -> None:
    assert provenance_validity([["a"], []]) is False


def test_provenance_validity_empty_list() -> None:
    assert provenance_validity([]) is True


def test_summarize_aggregates() -> None:
    rows = [
        EvalRow(
            scenario_id="a",
            drug_recall=1.0,
            drug_precision=1.0,
            nfz_accuracy=True,
            empty_list_correct=True,
            provenance_valid=True,
            duration_s=1.0,
            num_drugs_suggested=2,
        ),
        EvalRow(
            scenario_id="b",
            drug_recall=0.5,
            drug_precision=1.0,
            nfz_accuracy=False,
            empty_list_correct=True,
            provenance_valid=True,
            duration_s=2.0,
            num_drugs_suggested=1,
        ),
    ]
    summary = summarize(rows)
    assert summary.n == 2
    assert summary.mean_recall() == 0.75
    assert summary.mean_precision() == 1.0
    assert summary.nfz_accuracy_rate() == 0.5
    assert summary.empty_list_correctness_rate() == 1.0
    assert summary.mean_duration_s() == 1.5


def test_ground_truth_from_dict() -> None:
    raw = {
        "scenario_id": "endo-gra",
        "diagnostic_only": False,
        "expected_empty_drug_list": False,
        "expected_drugs": [
            {"name": "Propranolol", "atc": "C07AA05", "must_be_refundable": True},
        ],
        "expected_referrals": ["endokrynologia"],
        "expected_tests": ["TSH"],
    }
    gt = GroundTruth.from_dict(raw)
    assert gt.scenario_id == "endo-gra"
    assert len(gt.expected_drugs) == 1
    assert gt.expected_drugs[0].atc == "C07AA05"
