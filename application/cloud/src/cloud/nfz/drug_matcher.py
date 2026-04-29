"""Match an LLM-proposed drug candidate to its NFZ refundacja entry.

The agent's `propose_candidates` node emits `DrugCandidate` records
with whatever name + (claimed) ATC code Gemini produced. Several
failure modes:

1. **Name + matching ATC**: easy case — ATC lookup wins, name is sanity check.
2. **Name only, ATC hallucinated or missing**: substance lookup finds it.
3. **Trade name** (``"Eliquis"``): trade-name lookup resolves to apixaban.
4. **Polish inflection** (``"propranololem"`` instead of ``"propranolol"``):
   stem-prefix fallback (Polish casefold + 5-char prefix, same idiom as
   Phase 2b's `diagnosis_match`).
5. **Total miss**: returns ``None``, agent flags in `notes`.

Resolution order is deliberate: most-specific to least-specific.
Once a match wins, we stop — over-matching beats under-matching here
(false-positive NFZ entries are a finance signal the doctor sees;
false-negative NFZ misses look like "drug not refundable" and may
push the doctor toward more expensive choices).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from cloud.nfz.formulary_search import NFZEntry, NFZIndex, normalize_drug_name

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MatchResult:
    """Outcome of `match_candidate`. Carries the strategy that won so
    the agent can surface it in `notes` (e.g. "matched on trade name —
    confirm substance with the doctor").
    """

    entry: NFZEntry
    strategy: str  # "atc" | "substance" | "trade_name" | "stem"


def match_candidate(
    *,
    name: str,
    atc_code: str | None,
    index: NFZIndex,
) -> MatchResult | None:
    """Resolve a candidate drug to its NFZ entry.

    Args:
        name: Free-form drug name from the LLM. Trade or substance,
            possibly Polish-inflected.
        atc_code: LLM-claimed ATC code. May be hallucinated. Tried
            first as a hard ATC lookup; on miss, ignored.
        index: An `NFZIndex`. Production passes `_RealNFZIndex`;
            tests pass `_FakeNFZIndex`.

    Returns the first matching `NFZEntry` with the strategy that won,
    or ``None`` when nothing matched.
    """
    name = name.strip()
    if not name:
        return None

    # Strategy 1 — ATC exact match. Cheapest and most specific.
    if atc_code:
        hit = index.lookup_by_atc(atc_code)
        if hit is not None:
            return MatchResult(entry=hit, strategy="atc")
        log.debug("ATC %r not in NFZ index; falling back to name match", atc_code)

    # Strategy 2 — exact substance match (after diacritic strip + casefold).
    substance_hits = index.lookup_by_substance(name)
    if substance_hits:
        if len(substance_hits) > 1:
            log.debug("multiple substance matches for %r; using first", name)
        return MatchResult(entry=substance_hits[0], strategy="substance")

    # Strategy 3 — trade-name lookup. ``"Eliquis"`` → apixaban entry.
    trade_hits = index.lookup_by_trade_name(name)
    if trade_hits:
        if len(trade_hits) > 1:
            log.debug("multiple trade-name matches for %r; using first", name)
        return MatchResult(entry=trade_hits[0], strategy="trade_name")

    # Strategy 4 — stem-prefix fallback for Polish inflection. Same
    # 5-char-prefix idiom as Phase 2b's `_content_stems`. We match the
    # candidate's first 5 chars (after normalization) against every
    # entry's substance + trade-name stems. Drops "propranololem" →
    # "propranolol" mismatches.
    stem_match = _match_by_stem(name, index)
    if stem_match is not None:
        return MatchResult(entry=stem_match, strategy="stem")

    return None


def _match_by_stem(name: str, index: NFZIndex) -> NFZEntry | None:
    """Try a 5-char stem-prefix match against every entry.

    O(n) over the index, but the index is small (~3000 entries); even
    full sweep is < 5 ms. Worth the simplicity; if the index grows we
    can pre-build a trie.
    """
    needle = normalize_drug_name(name)
    if len(needle) < 4:  # too short for a meaningful stem
        return None
    stem = needle[:5] if len(needle) >= 5 else needle

    for entry in index:
        for candidate in (entry.substance, *entry.trade_names):
            normalized = normalize_drug_name(candidate)
            if normalized.startswith(stem):
                return entry
    return None
