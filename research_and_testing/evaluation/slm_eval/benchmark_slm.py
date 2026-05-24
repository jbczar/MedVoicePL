"""SLM eval orchestrator: runs `transcript_to_soap` over a corpus.

Per-scenario row: (scenario_id, attempts, coverage_count, cc_recall,
diag_match, icd10_match, plan_nonempty, duration_s). Aggregated by
the CLI layer (`research_and_testing/scripts/eval_slm_on_transcripts.py`);
this module just does the per-row work.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import ValidationError

from evaluation.slm_eval.soap_quality import (
    chief_complaint_recall,
    diagnosis_match,
    diagnosis_match_semantic,
    field_coverage,
    icd10_match,
    plan_nonempty,
)
from on_premise.structuring.pipeline import transcript_to_soap
from on_premise.structuring.soap_schema import SOAPNote

if TYPE_CHECKING:
    from pathlib import Path

    from dataset_pipeline.scenario_generator.schema import ScenarioModel
    from on_premise.structuring.slm_engine import StructuredExtractor
    from on_premise.stt.schema import Transcript

log = logging.getLogger(__name__)


@dataclass
class EvalRow:
    scenario_id: str
    variant: str  # clean | noisy
    schema_ok: bool
    cc_recall: float
    diag_match: bool  # legacy lexical (substring + stem-set)
    diag_match_semantic: bool  # Phase 4: embedding cosine >= threshold
    diag_cos_sim: float  # raw cosine similarity for inspection
    icd10_match: bool
    plan_nonempty: bool
    coverage_filled: int
    coverage_total: int
    duration_s: float
    error: str | None = None


def evaluate_one(
    scenario: ScenarioModel,
    transcript: Transcript,
    *,
    extractor: StructuredExtractor,
    variant: str,
    embedder: object | None = None,
    soap_persist_path: Path | None = None,
) -> EvalRow:
    """Run one (scenario, transcript) pair and return a row of metrics.

    If `embedder` is provided, computes `diag_match_semantic` via cosine
    similarity (Phase 4 Tier A1 metric upgrade). If `soap_persist_path` is
    provided, writes the SOAP JSON there for downstream stages.
    """
    t0 = time.monotonic()
    try:
        soap = transcript_to_soap(transcript, extractor=extractor)
    except ValidationError as exc:
        # The pipeline already retried; treat as schema failure row.
        return EvalRow(
            scenario_id=scenario.id,
            variant=variant,
            schema_ok=False,
            cc_recall=0.0,
            diag_match=False,
            diag_match_semantic=False,
            diag_cos_sim=0.0,
            icd10_match=False,
            plan_nonempty=False,
            coverage_filled=0,
            coverage_total=0,
            duration_s=time.monotonic() - t0,
            error=f"ValidationError: {exc.error_count()} errors",
        )
    except Exception as exc:
        return EvalRow(
            scenario_id=scenario.id,
            variant=variant,
            schema_ok=False,
            cc_recall=0.0,
            diag_match=False,
            diag_match_semantic=False,
            diag_cos_sim=0.0,
            icd10_match=False,
            plan_nonempty=False,
            coverage_filled=0,
            coverage_total=0,
            duration_s=time.monotonic() - t0,
            error=f"{type(exc).__name__}: {exc}",
        )

    assert isinstance(soap, SOAPNote)

    # Phase 4: persist SOAP to disk for downstream pipeline (Step 6.3).
    if soap_persist_path is not None:
        soap_persist_path.parent.mkdir(parents=True, exist_ok=True)
        soap_persist_path.write_text(
            soap.model_dump_json(indent=2), encoding="utf-8"
        )

    # Phase 4 Tier A1: semantic dx match alongside legacy lexical match.
    if embedder is not None:
        dx_sem, cos_sim = diagnosis_match_semantic(soap, scenario, embedder=embedder)
    else:
        dx_sem, cos_sim = False, 0.0

    coverage = field_coverage(soap)
    filled = sum(1 for v in coverage.values() if v)
    return EvalRow(
        scenario_id=scenario.id,
        variant=variant,
        schema_ok=True,
        cc_recall=chief_complaint_recall(soap, scenario),
        diag_match=diagnosis_match(soap, scenario),
        diag_match_semantic=dx_sem,
        diag_cos_sim=cos_sim,
        icd10_match=icd10_match(soap, scenario),
        plan_nonempty=plan_nonempty(soap),
        coverage_filled=filled,
        coverage_total=len(coverage),
        duration_s=time.monotonic() - t0,
        error=None,
    )


@dataclass
class EvalSummary:
    """Aggregated metrics across the eval run."""

    rows: list[EvalRow]

    @property
    def n(self) -> int:
        return len(self.rows)

    @property
    def schema_compliance_rate(self) -> float:
        if not self.rows:
            return 0.0
        return sum(1 for r in self.rows if r.schema_ok) / self.n

    @property
    def mean_cc_recall(self) -> float:
        valid = [r for r in self.rows if r.schema_ok]
        return sum(r.cc_recall for r in valid) / len(valid) if valid else 0.0

    @property
    def diagnosis_match_rate(self) -> float:
        valid = [r for r in self.rows if r.schema_ok]
        return sum(1 for r in valid if r.diag_match) / len(valid) if valid else 0.0

    @property
    def diagnosis_match_semantic_rate(self) -> float:
        valid = [r for r in self.rows if r.schema_ok]
        return (
            sum(1 for r in valid if r.diag_match_semantic) / len(valid)
            if valid
            else 0.0
        )

    @property
    def mean_diag_cos_sim(self) -> float:
        valid = [r for r in self.rows if r.schema_ok]
        return sum(r.diag_cos_sim for r in valid) / len(valid) if valid else 0.0

    @property
    def icd10_match_rate(self) -> float:
        valid = [r for r in self.rows if r.schema_ok]
        return sum(1 for r in valid if r.icd10_match) / len(valid) if valid else 0.0

    @property
    def plan_nonempty_rate(self) -> float:
        valid = [r for r in self.rows if r.schema_ok]
        return sum(1 for r in valid if r.plan_nonempty) / len(valid) if valid else 0.0

    @property
    def mean_duration_s(self) -> float:
        if not self.rows:
            return 0.0
        return sum(r.duration_s for r in self.rows) / self.n


def summarize(rows: list[EvalRow]) -> EvalSummary:
    return EvalSummary(rows=list(rows))
