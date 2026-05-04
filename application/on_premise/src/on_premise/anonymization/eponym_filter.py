"""Veto filter for Presidio PERSON detections that are medical eponyms.

The Phase 2c anonymizer aggressively redacts every spaCy PERSON
detection. spaCy's `pl_core_news_lg` (rightly) tags `Gravesa`,
`Parkinsona`, `Hashimoto`, etc. as PERSON because they ARE surnames.
But in clinical context — "choroba Gravesa-Basedowa", "podejrzenie
choroby Parkinsona" — these are disease names, not patient identifiers.
Redacting them produces clinically meaningless `choroba <PERSON_2>-Basedowa`
output that breaks the cloud agent's downstream reasoning.

This module is the **veto layer**: a context-aware filter that drops
PERSON detections when:

1. The detected span (after Polish casefold + diacritic strip) is in
   our curated eponym dictionary, AND
2. The 30 characters preceding the span contain a medical-context
   trigger word (`choroba`, `zespół`, `objaw`, `test`, `skala`,
   `tetralogia`, `manewr`, `operacja`, `reakcja`, `podejrzenie X`,
   `rozpoznanie X` where X is a disease keyword).

Why both conditions: a patient surnamed "Parkinson" must still be
anonymized. The context check is what separates "Parkinson — pacjent"
from "choroba Parkinsona — diagnoza". If either condition fails, we
fall back to the default behavior (redact).

Cross-references:
* DD-3 in Phase 2c plan: `PolishNameListBooster` adds names to
  PERSON detection. This filter is the symmetric subtraction.
* `PresidioAnonymizer._walk` calls `_filter_eponyms` between
  `analyze()` and `_resolve_overlaps()`.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from presidio_analyzer import RecognizerResult

log = logging.getLogger(__name__)


# Polish diacritic-stripping table (same idiom as Phase 2c's
# `PolishNameListBooster._normalize`).
_POLISH_DIACRITICS = str.maketrans(
    "ąćęłńóśźżĄĆĘŁŃÓŚŹŻ", "acelnoszzACELNOSZZ"
)


# Medical-context trigger words. If any of these appears within the
# 30-character window preceding a PERSON detection, the detection is
# eligible for veto (when the span text is in the eponym dictionary).
# Polish forms include common inflections.
_CONTEXT_TRIGGERS = (
    "choroba", "choroby", "chorobie", "chorobę", "chorobą",
    "zespół", "zespołu", "zespole", "zespołem",
    "objaw", "objawu", "objawy", "objawów",
    "test", "testu", "testy",
    "skala", "skali", "skalę", "skalą",
    "tetralogia", "tetralogii",
    "manewr", "manewru",
    "operacja", "operacji", "operację",
    "reakcja", "reakcji", "reakcję",
    "rozpoznanie", "rozpoznania",
    "podejrzenie", "podejrzenia",
    "diagnoza", "diagnozy",
    "stwardnienie",  # for "stwardnienie zanikowe boczne (Lou Gehriga)"
)


# Surrounding context window (chars to look back from the span start).
_CONTEXT_WINDOW = 30


@dataclass
class MedicalEponymFilter:
    """Filters PERSON detections that are clinical eponyms in context.

    Build via `load_default()` to read the bundled JSON dictionary.
    Tests can construct directly with custom `eponyms` for fixtures.
    """

    eponyms: frozenset[str] = field(default_factory=frozenset)
    """Normalized (lowercase + diacritic-strip) eponym tokens. Match
    against the normalized span text.
    """

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> MedicalEponymFilter:
        """Build from the JSON shape used by `medical_eponyms.json` —
        a top-level dict whose values are lists of eponym strings.
        """
        terms: set[str] = set()
        for key, value in raw.items():
            if key.startswith("_"):
                continue
            if not isinstance(value, list):
                continue
            for entry in value:
                if isinstance(entry, str):
                    terms.add(_normalize(entry))
        return cls(eponyms=frozenset(t for t in terms if t))

    @classmethod
    def from_path(cls, path: Path) -> MedicalEponymFilter:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    @classmethod
    def empty(cls) -> MedicalEponymFilter:
        return cls(eponyms=frozenset())

    def filter(
        self,
        results: list[RecognizerResult],
        *,
        text: str,
    ) -> list[RecognizerResult]:
        """Return results with eponym-vetoed PERSON detections dropped."""
        if not results or not self.eponyms:
            return results
        out: list[RecognizerResult] = []
        for r in results:
            if self._should_veto(r, text=text):
                log.debug(
                    "vetoed eponym PERSON span %r at [%d:%d]",
                    text[r.start : r.end], r.start, r.end,
                )
                continue
            out.append(r)
        return out

    def _should_veto(self, result: RecognizerResult, *, text: str) -> bool:
        if result.entity_type != "PERSON":
            return False
        span_text = text[result.start : result.end]
        if not _is_eponym(span_text, self.eponyms):
            return False
        # Look back up to _CONTEXT_WINDOW chars for a trigger word.
        window_start = max(0, result.start - _CONTEXT_WINDOW)
        window = text[window_start : result.start].lower()
        return any(trigger in window for trigger in _CONTEXT_TRIGGERS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    """Diacritic + apostrophe + case strip for eponym match.

    Uses Unicode NFKD decomposition + combining-mark removal to handle
    Polish (ąćęłńóśźż), French (éèêç in "Ménière", "Behçet"), German
    (ö, ü), Spanish (ñ), etc. — any eponym source language. Then a
    Polish-only fallback table catches the few stroke-letters NFKD
    doesn't decompose (Ł→L on macOS NFKD, but ł→l does decompose, so
    we apply the table after for safety).
    """
    cleaned = re.sub(r"['’ʼ]", "", text)
    decomposed = unicodedata.normalize("NFKD", cleaned)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return stripped.translate(_POLISH_DIACRITICS).casefold().strip()


def _is_eponym(span_text: str, eponyms: frozenset[str]) -> bool:
    """Match the span (or its hyphen-split parts) against the dictionary.

    Examples that should match:
    * "Gravesa-Basedowa"   → both halves OR full string in dict
    * "Parkinsona"         → "parkinsona" in dict
    * "Wolffa-Parkinsona-White'a" → full string OR each part
    """
    norm = _normalize(span_text)
    if not norm:
        return False
    if norm in eponyms:
        return True
    # Hyphenated multi-name eponym — accept if ANY part matches.
    parts = [p for p in norm.split("-") if p]
    return any(p in eponyms for p in parts)


_DEFAULT_EPONYMS_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "medical_eponyms.json"
)


def load_default(path: Path | None = None) -> MedicalEponymFilter:
    """Load the bundled eponym dictionary. Falls back to empty filter
    when the file is missing (so that a misconfigured deployment fails
    open — anonymizes everything — rather than silently dropping the
    feature).
    """
    p = path or _DEFAULT_EPONYMS_PATH
    if not p.is_file():
        log.warning("eponym dictionary missing at %s — running without filter", p)
        return MedicalEponymFilter.empty()
    return MedicalEponymFilter.from_path(p)


__all__ = [
    "MedicalEponymFilter",
    "load_default",
]
