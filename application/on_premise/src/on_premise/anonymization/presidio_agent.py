"""Anonymization agent: `SOAPNoteSubmission` → `AnonymizedSOAPNote`.

Pipeline:

1. **Analyze**: every string leaf of the SOAP submission goes through
   the Presidio `AnalyzerEngine` (spaCy `pl_core_news_lg` + our custom
   Polish recognizers). Returns `RecognizerResult` spans per leaf.
2. **Resolve overlaps**: spans sorted by `(start, -score)`; later spans
   contained in earlier ones are dropped (Presidio's
   `MERGE_SIMILAR_OR_CONTAINED` semantic, simplified).
3. **Splice + tokenize**: walks each leaf left-to-right, replacing
   each surviving span with a placeholder token assigned by
   per-session counters. Same `(entity_type, original_text)` within a
   session reuses the same token (e.g. all mentions of "Kowalski" map
   to `<PERSON_3>`).
4. **Rebuild**: the redacted leaves are stitched back into a dict
   matching `AnonymizedSOAPNote`'s shape and validated.

Why we **don't** use Presidio's `AnonymizerEngine`: it validates custom
operator lambdas by calling them with the literal `'PII'` as a
preflight, which pollutes our session counter (every entity type would
burn `<TYPE_1>` on the validation probe before the first real match
even runs). Splicing strings directly is also simpler — we want full
control over the token grammar (`<{TYPE}_{N}>`) and the audit trail
(`Replacement` records track originals → tokens for the on-prem
audit log).

Trust boundary: `Replacement` and `_TokenAssigner` are listed in
`FORBIDDEN_NAMES` of `test_trust_boundary_imports.py` — they hold
references to original PII via `original_sha256`/`token_map` and must
never cross into `shared/`/`cloud/`/`gateway/`.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Protocol

from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider

from on_premise.anonymization.audit_logger import Replacement, hash_pii
from on_premise.anonymization.pl_recognizers import (
    build_polish_recognizers,
    load_scenario_names,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from datetime import date
    from pathlib import Path
    from uuid import UUID

log = logging.getLogger(__name__)


# Default analyzer threshold. Recognizers emit scores 0.3-1.0; entity
# results below this drop. PESEL/NIP regex-only score is 0.4 (below
# threshold by themselves) but bumped to ~0.85 by checksum or context.
_DEFAULT_THRESHOLD = 0.5

# spaCy pl_core_news_lg uses Polish-specific entity labels. Map to
# Presidio's standard PII entity types.
_PL_ENTITY_MAPPING: dict[str, str] = {
    "persName": "PERSON",
    "placeName": "LOCATION",
    "geogName": "LOCATION",
    "orgName": "ORGANIZATION",
    "date": "DATE_TIME",
    "time": "DATE_TIME",
}

# Entity types we anonymize (everything else passes through). Order
# matches DD-10 (most-specific first) for tie-breaking when spans
# overlap.
_DEFAULT_ENTITIES = (
    "PL_PESEL",
    "PL_NIP",
    "PL_ID_CARD",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "PERSON",
    "LOCATION",
)


# ---------------------------------------------------------------------------
# RecognizerRegistry Protocol — for test injection.
# ---------------------------------------------------------------------------


class RecognizerRegistry(Protocol):
    """Mirrors the bit of Presidio's API we depend on. Test fakes
    implement this so unit tests don't need to load `pl_core_news_lg`.
    """

    def analyze(self, text: str, language: str) -> list[RecognizerResult]: ...


@dataclass
class _RealRegistry:
    """Wraps Presidio's `AnalyzerEngine`. Built lazily on first
    `analyze` call so simply importing this module doesn't trigger a
    spaCy model load (~2-3 s, ~600 MB).
    """

    threshold: float = _DEFAULT_THRESHOLD
    name_booster_names: tuple[str, ...] = ()

    def analyze(self, text: str, language: str = "pl") -> list[RecognizerResult]:
        analyzer = _get_analyzer(
            threshold=self.threshold,
            name_booster_names=tuple(self.name_booster_names),
        )
        return analyzer.analyze(text=text, language=language, entities=list(_DEFAULT_ENTITIES))


@lru_cache(maxsize=4)
def _get_analyzer(
    *,
    threshold: float,
    name_booster_names: tuple[str, ...],
) -> AnalyzerEngine:
    """Build the analyzer once per (threshold, name-list) pair.

    The analyzer construction is the expensive bit (loads spaCy
    `pl_core_news_lg`). Cached so the FastAPI startup hook can warm it
    once and every subsequent request reuses the same object.

    `lru_cache` with the small tuple of names as cache key means we
    have one cached engine per running clinic in practice (the names
    stay constant).
    """
    log.info(
        "building Presidio analyzer (threshold=%s, %d boost names)",
        threshold,
        len(name_booster_names),
    )
    nlp_config = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "pl", "model_name": "pl_core_news_lg"}],
        "ner_model_configuration": {
            "model_to_presidio_entity_mapping": _PL_ENTITY_MAPPING,
            "low_confidence_score_multiplier": 0.4,
            "low_score_entity_names": [],
        },
    }
    nlp_engine = NlpEngineProvider(nlp_configuration=nlp_config).create_engine()
    analyzer = AnalyzerEngine(
        nlp_engine=nlp_engine,
        supported_languages=["pl"],
        default_score_threshold=threshold,
    )
    pl_recognizers = build_polish_recognizers(name_booster_names=name_booster_names)
    for recognizer in pl_recognizers:
        analyzer.registry.add_recognizer(recognizer)
    log.info("Presidio analyzer ready (added %d Polish recognizers)", len(pl_recognizers))
    return analyzer


def warm_analyzer(*, name_booster_names: Iterable[str] = ()) -> None:
    """Pre-build the cached analyzer so the first request is fast.

    Called by the FastAPI `startup` event. Idempotent (the `lru_cache`
    short-circuits the second time).
    """
    _ = _get_analyzer(
        threshold=_DEFAULT_THRESHOLD,
        name_booster_names=tuple(name_booster_names),
    )


# ---------------------------------------------------------------------------
# Token assigner — manages per-session counters and the originals map.
# ---------------------------------------------------------------------------


@dataclass
class _TokenAssigner:
    """Stateful per-session: same original → same token within a session.

    Cross-session unmappable: the assigner is constructed fresh per
    `anonymize_soap` call.
    """

    counters: Counter[str] = field(default_factory=Counter)
    """Next-counter map keyed by entity type. `<PERSON_1>` is issued
    when `counters["PERSON"]` is 1, etc.
    """

    seen: dict[tuple[str, str], str] = field(default_factory=dict)
    """`(entity_type, original_text)` → assigned token. Lookup-only."""

    token_map: dict[str, str] = field(default_factory=dict)
    """`token → original` for the on-prem token-map JSON. Reverse of
    `seen` (with the `(entity_type, ...)` key flattened).
    """

    def assign(self, entity_type: str, original: str) -> str:
        key = (entity_type, original)
        if key in self.seen:
            return self.seen[key]
        self.counters[entity_type] += 1
        token = f"<{entity_type}_{self.counters[entity_type]}>"
        self.seen[key] = token
        self.token_map[token] = original
        return token


# ---------------------------------------------------------------------------
# String-leaf splicing helpers.
# ---------------------------------------------------------------------------


def _resolve_overlaps(
    results: list[RecognizerResult],
) -> list[RecognizerResult]:
    """Sort results by (start asc, score desc) and drop any later span
    whose range falls entirely inside an earlier span.

    Mirrors Presidio's `MERGE_SIMILAR_OR_CONTAINED` strategy at the
    granularity we actually need: overlaps within the same entity get
    the longer/higher-score one.
    """
    if not results:
        return []
    sorted_results = sorted(results, key=lambda r: (r.start, -r.score))
    out: list[RecognizerResult] = []
    for r in sorted_results:
        if any(kept.start <= r.start and kept.end >= r.end and kept is not r for kept in out):
            continue  # contained in something already kept
        out.append(r)
    return out


def _splice(
    text: str,
    results: list[RecognizerResult],
    assigner: _TokenAssigner,
    *,
    location_path: str,
) -> tuple[str, list[Replacement]]:
    """Replace each span in `text` with a placeholder token from `assigner`.

    Returns the redacted text plus the `Replacement` records (one per
    surviving span) for the audit logger.
    """
    if not results:
        return text, []
    resolved = _resolve_overlaps(results)
    out_parts: list[str] = []
    replacements: list[Replacement] = []
    cursor = 0
    for r in resolved:
        out_parts.append(text[cursor : r.start])
        original = text[r.start : r.end]
        token = assigner.assign(r.entity_type, original)
        out_parts.append(token)
        replacements.append(
            Replacement(
                entity_type=r.entity_type,
                token=token,
                original_sha256=hash_pii(original),
                location_path=location_path,
                location_offset_start=r.start,
                location_offset_end=r.end,
            )
        )
        cursor = r.end
    out_parts.append(text[cursor:])
    return "".join(out_parts), replacements


# ---------------------------------------------------------------------------
# PresidioAnonymizer — public entry point.
# ---------------------------------------------------------------------------


@dataclass
class AnonymizationResult:
    """Bundle of return values from `PresidioAnonymizer.anonymize_soap`.

    Defined as a dataclass (not a tuple) so call sites read clearly:
    `result.note`, `result.replacements`, `result.token_map`. The
    `n_entities_per_type` summary is what the public
    `AnonymizedMetadata` sees; the rest stays on-prem.
    """

    note: Any
    """Redacted `AnonymizedSOAPNote`. Typed `Any` so this module
    avoids importing from `shared.schemas`; the FastAPI layer narrows
    the type.
    """

    replacements: list[Replacement]
    """All replacement records across every redacted leaf. Goes into
    the on-prem audit log; never crosses the trust boundary.
    """

    token_map: dict[str, str]
    """`{token: original}` map for the doctor-side reverse view.
    Stays on-prem.
    """

    n_entities_per_type: dict[str, int]
    """Aggregate `{entity_type: count}` for `AnonymizedMetadata`."""


@dataclass
class PresidioAnonymizer:
    """Anonymizes a `SOAPNoteSubmission` → `AnonymizedSOAPNote`.

    Stateless across calls — every `anonymize_soap` invocation is an
    independent session with its own counters.

    Args:
        registry: Anything implementing `RecognizerRegistry`. Defaults
            to `_RealRegistry()` (real spaCy + Presidio). Tests inject
            a `_FakeRegistry` to avoid loading `pl_core_news_lg`.
    """

    registry: RecognizerRegistry = field(default_factory=_RealRegistry)
    eponym_filter: Any = None  # MedicalEponymFilter | None — see _walk

    # Field paths whose value is schema-constrained clinical content
    # (NOT free narrative text) and therefore should bypass the
    # analyzer. spaCy `pl_core_news_lg` PERSON has been observed
    # mis-detecting tokens like "I10" or "J44" inside ICD-10 codes,
    # producing `<PERSON_n>.9` strings that then fail the regex
    # constraint on `AnonymizedAssessment.icd10_code`. We skip these
    # explicitly — the schema's regex/Literal constraints already
    # prove they can't carry PII.
    _SKIP_PATHS = frozenset({"assessment.icd10_code", "objective.vitals.name"})

    def anonymize_soap(
        self,
        submission: Any,
        *,
        session_id: UUID,
        visit_date: date,
        specialization: str,
        anonymizer_version: str,
    ) -> AnonymizationResult:
        """Run the full anonymization pipeline.

        `submission` is duck-typed as a Pydantic `BaseModel` with the
        SOAPNoteSubmission shape. Imported from the local schema by
        the caller; we accept `Any` to avoid coupling this module to
        the 2b on-prem types.
        """
        # Local import — `shared.schemas` is the only place that
        # holds the anonymized types. Importing here keeps module
        # load light when the API is configured but not handling
        # requests.
        from shared.schemas import AnonymizedSOAPNote

        assigner = _TokenAssigner()
        all_replacements: list[Replacement] = []

        # Walk the submission tree as a dict. Pydantic's `model_dump`
        # gives us a plain dict that round-trips cleanly into the
        # anonymized type (the field shapes match — see
        # `shared.schemas.anonymized` docstring).
        data: dict[str, Any] = submission.model_dump()
        redacted = self._walk(data, path="", assigner=assigner, replacements=all_replacements)

        # Aggregate `{entity_type: count}` for the metadata block.
        counts: dict[str, int] = dict(Counter(r.entity_type for r in all_replacements))

        # Build the metadata (gateway-supplied bits + our own counts).
        redacted["metadata"] = {
            "anonymizer_version": anonymizer_version,
            "n_entities_per_type": counts,
            "visit_date": visit_date.isoformat(),
            "specialization": specialization,
            "session_id": str(session_id),
        }

        note = AnonymizedSOAPNote.model_validate(redacted)
        return AnonymizationResult(
            note=note,
            replacements=all_replacements,
            token_map=dict(assigner.token_map),
            n_entities_per_type=counts,
        )

    # ----- internal: tree walk ---------------------------------------------

    def _walk(
        self,
        node: Any,
        *,
        path: str,
        assigner: _TokenAssigner,
        replacements: list[Replacement],
    ) -> Any:
        """Recursive walk. String leaves are analyzed + spliced; dicts
        and lists are rebuilt; everything else passes through.
        """
        if isinstance(node, str):
            if not node.strip():
                return node  # skip empty / whitespace-only — nothing to redact
            # Schema-constrained fields (ICD-10 codes, vital-name
            # whitelist) bypass the analyzer — see _SKIP_PATHS rationale.
            if path in self._SKIP_PATHS:
                return node
            results = self.registry.analyze(node, language="pl")
            # Veto medical-eponym PERSON detections in clinical context
            # (e.g. "choroba Gravesa-Basedowa" → don't redact "Gravesa").
            if self.eponym_filter is not None:
                results = self.eponym_filter.filter(results, text=node)
            redacted, leaf_repls = _splice(node, results, assigner, location_path=path)
            replacements.extend(leaf_repls)
            return redacted
        if isinstance(node, dict):
            return {
                key: self._walk(
                    value,
                    path=f"{path}.{key}" if path else key,
                    assigner=assigner,
                    replacements=replacements,
                )
                for key, value in node.items()
            }
        if isinstance(node, list):
            return [
                self._walk(
                    item,
                    path=f"{path}[{i}]",
                    assigner=assigner,
                    replacements=replacements,
                )
                for i, item in enumerate(node)
            ]
        return node


# ---------------------------------------------------------------------------
# Convenience: build the default registry seeded from on-disk scenarios.
# ---------------------------------------------------------------------------


def default_registry(*, scenarios_dir: Path | None = None) -> RecognizerRegistry:
    """Returns a `_RealRegistry` seeded with names from
    `dataset_pipeline/output/scenarios/` if available.

    The seeding gives us 100% recall on Phase 2c's test corpus
    (closing spaCy's ~15% miss rate on Polish surnames) without
    requiring any code changes when names update — just regenerate
    scenarios.
    """
    names: tuple[str, ...] = ()
    if scenarios_dir is not None:
        names = tuple(load_scenario_names(scenarios_dir))
    return _RealRegistry(name_booster_names=names)


# Make the dataclass itself hashable for use as `lru_cache` key params
# elsewhere — it isn't normally because `dict` fields are unhashable.
# Not needed yet; left as a future hook.


def reset_analyzer_cache() -> None:
    """Clear the cached `_get_analyzer` instance — used by tests."""
    _get_analyzer.cache_clear()


# Re-export `Replacement` for convenience (callers want a single
# import surface).
__all__ = [
    "AnonymizationResult",
    "PresidioAnonymizer",
    "RecognizerRegistry",
    "Replacement",
    "default_registry",
    "reset_analyzer_cache",
    "warm_analyzer",
]
