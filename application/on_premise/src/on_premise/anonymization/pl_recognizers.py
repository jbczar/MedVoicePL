"""Custom Presidio recognizers for Polish PII formats.

These recognizers cover the gap between Presidio's English defaults and
Polish-specific identifier formats observed in `dataset_pipeline` output:

* `PolishPESELRecognizer` — 11-digit national ID with checksum (8/8 in
  scenario metadata; 1/8 spoken in dialogue).
* `PolishNIPRecognizer` — 10-digit tax ID with checksum (0/8 in current
  data, but easily produced by hallucinating SLM).
* `PolishPhoneRecognizer` — mobile (`XXX-XXX-XXX` / `XXX XXX XXX`) and
  landline (`XX XXX XX XX`) with optional `+48`/`48` prefix (6/8 in
  dialogue).
* `PolishIDCardRecognizer` — `LLL NNNNNN` dowód osobisty (3 letters +
  6 digits; not in current data but plausible).
* `PolishAddressRecognizer` — `ul.`/`ulica`/`al.`/`aleja`/`os.`/`osiedle`
  prefix + word (4/8 in dialogue).
* `PolishNameListBooster` — `PatternRecognizer` whose patterns are fed
  by `dataset_pipeline/output/scenarios/*.json` `patient_full_name` at
  construction time. Catches the ~15% recall gap that spaCy
  `pl_core_news_lg` PERSON has on Polish surnames within our test
  corpus. For out-of-distribution names, spaCy alone is the fallback.

Design notes:

* All recognizers subclass `PatternRecognizer`. The `validate_result`
  hook is overridden on PESEL/NIP to apply checksums — Presidio uses
  the return value to enhance score (`True` → ~1.0) or invalidate
  (`False` → drop), which is exactly what we want for "regex matches
  but doesn't pass checksum".
* Context-word boosting (`context=[...]`) raises confidence when the
  trigger appears within 3 tokens of the match. Without context the
  bare regex score is 0.4 (below the analyzer's 0.5 default threshold)
  for low-specificity patterns, so unrelated digit runs don't get
  flagged unless something in the surrounding text suggests "this
  IS PII".
* The plan called for a `PolishStreetContextBooster` that boosts other
  recognizers' LOCATION entities. Presidio's native API only boosts
  each recognizer's own matches, so we instead emit our own
  `LOCATION` entity directly from `PolishAddressRecognizer` — same
  outcome, simpler implementation.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

import regex as _regex
from presidio_analyzer import Pattern, PatternRecognizer

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

# Regex flags WITHOUT `IGNORECASE`. The address recognizer needs
# case-sensitive matching for "uppercase street name" detection.
# (Presidio's default `global_regex_flags` includes `IGNORECASE`,
# which would let "przy wchodzeniu" match the address pattern.)
_CASE_SENSITIVE_FLAGS = _regex.M | _regex.S

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checksum helpers — pure functions, easy to unit-test.
# ---------------------------------------------------------------------------


def _pesel_checksum_ok(digits: str) -> bool:
    """PESEL official checksum: weights `1,3,7,9,1,3,7,9,1,3`, sum mod 10.

    `digits` is exactly 11 ASCII digits. The 11th digit is the check digit;
    `(10 - sum_mod_10) % 10` should equal the 11th digit.
    """
    if len(digits) != 11 or not digits.isdigit():
        return False
    weights = (1, 3, 7, 9, 1, 3, 7, 9, 1, 3)
    s = sum(int(d) * w for d, w in zip(digits[:10], weights, strict=True))
    expected = (10 - (s % 10)) % 10
    return expected == int(digits[10])


def _nip_checksum_ok(digits: str) -> bool:
    """NIP official checksum: weights `6,5,7,2,3,4,5,6,7`, sum mod 11.

    `digits` is exactly 10 ASCII digits. The check digit (10th) is the
    sum-mod-11 of the first 9 weighted digits. NIPs whose checksum lands
    on 10 are by official rule invalid (no number is issued).
    """
    if len(digits) != 10 or not digits.isdigit():
        return False
    weights = (6, 5, 7, 2, 3, 4, 5, 6, 7)
    s = sum(int(d) * w for d, w in zip(digits[:9], weights, strict=True))
    expected = s % 11
    if expected == 10:
        return False  # invalid per rule
    return expected == int(digits[9])


# ---------------------------------------------------------------------------
# Polish PESEL recognizer — 11 digits + checksum.
# ---------------------------------------------------------------------------


class PolishPESELRecognizer(PatternRecognizer):
    """Detects Polish PESEL numbers — 11 digits with checksum validation.

    Bare regex score is 0.4 (below default threshold of 0.5) so random
    11-digit runs don't fire alone. Checksum-pass via `validate_result`
    bumps confidence to ~1.0. Checksum-fail returns `None` (NOT `False`)
    — preserves the regex score so context-word boost (`PESEL` nearby)
    can still push it over threshold. This is on purpose: the
    dataset_pipeline generates synthetic PESELs that intentionally don't
    pass the official checksum (so they can never collide with a real
    person), and our anonymizer must still catch those when the patient
    speaks one in dialogue.
    """

    PATTERNS = (
        Pattern(
            name="pesel_11_digit",
            regex=r"\b\d{11}\b",
            score=0.4,
        ),
    )
    CONTEXT = ("pesel",)

    def __init__(
        self,
        patterns: list[Pattern] | None = None,
        context: list[str] | None = None,
        supported_language: str = "pl",
        supported_entity: str = "PL_PESEL",
    ) -> None:
        super().__init__(
            supported_entity=supported_entity,
            patterns=patterns or list(self.PATTERNS),
            context=context or list(self.CONTEXT),
            supported_language=supported_language,
        )

    def validate_result(self, pattern_text: str) -> bool | None:
        """`True` if checksum passes (bump to ~1.0), `None` if it fails
        (preserve regex score; context-word boost can still push it
        through the analyzer threshold).
        """
        # `pattern_text` is exactly the regex match — already 11 digits.
        return True if _pesel_checksum_ok(pattern_text) else None


# ---------------------------------------------------------------------------
# Polish NIP recognizer — 10 digits + checksum.
# ---------------------------------------------------------------------------


class PolishNIPRecognizer(PatternRecognizer):
    """Detects Polish NIP (tax ID) — 10 digits with checksum.

    Same posture as PESEL: low bare-regex score, bumped by checksum +
    context words.
    """

    PATTERNS = (
        Pattern(
            name="nip_10_digit",
            regex=r"\b\d{10}\b",
            score=0.4,
        ),
    )
    CONTEXT = ("nip",)

    def __init__(
        self,
        patterns: list[Pattern] | None = None,
        context: list[str] | None = None,
        supported_language: str = "pl",
        supported_entity: str = "PL_NIP",
    ) -> None:
        super().__init__(
            supported_entity=supported_entity,
            patterns=patterns or list(self.PATTERNS),
            context=context or list(self.CONTEXT),
            supported_language=supported_language,
        )

    def validate_result(self, pattern_text: str) -> bool | None:
        """Same fail-soft semantics as PESEL: `True` on checksum pass,
        `None` on fail (preserves regex score for context-driven boost).
        """
        return True if _nip_checksum_ok(pattern_text) else None


# ---------------------------------------------------------------------------
# Polish phone recognizer — mobile (9 digits) and landline.
# ---------------------------------------------------------------------------


class PolishPhoneRecognizer(PatternRecognizer):
    """Detects Polish phone numbers in mobile and landline formats.

    Covers, with optional `+48`/`48` prefix:

    * Mobile: `501-234-567`, `501 234 567`, `501234567` (9 contiguous digits).
    * Landline: `71 333 44 55`, `22 555 66 77` (2-3-2-2 grouping).

    Score 0.6 because the patterns are reasonably specific (9 digits in
    a phone-y arrangement). Context words give an additional bump for
    e.g. medical-history "pacjent zostawił numer 501-234-567".
    """

    PATTERNS = (
        # Mobile: 9 digits with separator-flexible grouping. Optional +48.
        # Alternation handles both `\b` (no prefix) and the prefix path
        # — between `+48` and the first digit there's no word boundary
        # (digit-to-digit is not a `\b` position), so we need an
        # explicit branch.
        Pattern(
            name="pl_phone_mobile",
            regex=(
                r"(?:\+?48[\s-]?\d{3}[\s-]?\d{3}[\s-]?\d{3}"
                r"|\b\d{3}[\s-]?\d{3}[\s-]?\d{3}\b)"
            ),
            score=0.6,
        ),
        # Landline: 2-3-2-2 grouping (separator required between groups
        # to avoid colliding with the mobile pattern's 3-3-3 form).
        Pattern(
            name="pl_phone_landline",
            regex=(
                r"(?:\+?48[\s-]?\d{2}[\s-]\d{3}[\s-]\d{2}[\s-]\d{2}"
                r"|\b\d{2}[\s-]\d{3}[\s-]\d{2}[\s-]\d{2}\b)"
            ),
            score=0.6,
        ),
    )
    CONTEXT = ("telefon", "tel", "kontakt", "numer", "komórka", "komórkowy")

    def __init__(
        self,
        patterns: list[Pattern] | None = None,
        context: list[str] | None = None,
        supported_language: str = "pl",
        supported_entity: str = "PHONE_NUMBER",
    ) -> None:
        super().__init__(
            supported_entity=supported_entity,
            patterns=patterns or list(self.PATTERNS),
            context=context or list(self.CONTEXT),
            supported_language=supported_language,
        )


# ---------------------------------------------------------------------------
# Polish ID card recognizer — 3 letters + 6 digits.
# ---------------------------------------------------------------------------


class PolishIDCardRecognizer(PatternRecognizer):
    """Detects Polish dowód osobisty numbers (3 uppercase letters + 6 digits).

    Bare-regex score is low (`0.3`); context words `dowód`, `dowodu`,
    `osobisty`, `osobistego`, `DO` boost confidence. Without context the
    pattern doesn't fire (would over-match arbitrary three-letter codes
    + numbers in clinical text, e.g. lab reagent IDs).
    """

    PATTERNS = (
        Pattern(
            name="pl_id_card",
            regex=r"\b[A-Z]{3}\s?\d{6}\b",
            score=0.3,
        ),
    )
    CONTEXT = ("dowód", "dowodu", "dowody", "osobisty", "osobistego", "do")

    def __init__(
        self,
        patterns: list[Pattern] | None = None,
        context: list[str] | None = None,
        supported_language: str = "pl",
        supported_entity: str = "PL_ID_CARD",
    ) -> None:
        super().__init__(
            supported_entity=supported_entity,
            patterns=patterns or list(self.PATTERNS),
            context=context or list(self.CONTEXT),
            supported_language=supported_language,
        )


# ---------------------------------------------------------------------------
# Polish address recognizer — street-prefix + word(s) + optional number.
# ---------------------------------------------------------------------------


class PolishAddressRecognizer(PatternRecognizer):
    """Detects Polish street addresses by prefix word.

    Triggers on `ul.`/`ulica`/`al.`/`aleja`/`os.`/`osiedle`/`przy`
    followed by the street name (one or two capitalized words) and an
    optional number with apartment suffix (`12`, `12/3`, `12a`).

    Emits `LOCATION` entity (Presidio's standard category) so downstream
    consumers don't need to learn a new entity type.

    Score 0.6 — pattern is specific enough to not need a checksum, but
    a real "ulica Piaskowa była zamknięta" use *will* fire as a false
    positive. Acceptable in 2c — the cost of a false positive is
    over-redaction, not data leakage.
    """

    PATTERNS = (
        Pattern(
            name="pl_street_address",
            regex=(
                # Prefix uses scoped case-insensitive matching `(?i:...)`
                # so "Ulica" / "ulica" / "ULICA" all match. Street name
                # below stays case-SENSITIVE (the global flags exclude
                # IGNORECASE) so capital-letter detection is real.
                # Polish prefix words cover all common case forms:
                # nominative, genitive, dative, locative.
                r"\b(?i:ul\.?|ulica|ulicy|ulicę|"
                r"al\.?|aleja|alei|aleję|"
                r"os\.?|osiedle|osiedlu|osiedla|"
                r"przy)\s+"
                # Street name MUST start with uppercase (Polish proper
                # noun). Without this, "przy" + lowercase word matches
                # things like "przy wchodzeniu" (a prepositional phrase
                # meaning "when going up") — false positive on
                # clinical text. Requiring uppercase keeps the
                # recognizer scoped to actual street names.
                r"[A-ZŁŚŻŹĆĄĘÓŃ]\w*"
                r"(?:\s+[A-ZŁŚŻŹĆĄĘÓŃa-złśżźćąęóń]\w*)?"
                # Optional number + optional /apt or letter suffix.
                r"(?:\s+\d+(?:[/-]\d+|[a-zA-Z])?)?"
            ),
            score=0.6,
        ),
    )
    CONTEXT = ("adres", "mieszka", "mieszkam", "zameldowany", "ulica")

    def __init__(
        self,
        patterns: list[Pattern] | None = None,
        context: list[str] | None = None,
        supported_language: str = "pl",
        supported_entity: str = "LOCATION",
    ) -> None:
        super().__init__(
            supported_entity=supported_entity,
            patterns=patterns or list(self.PATTERNS),
            context=context or list(self.CONTEXT),
            supported_language=supported_language,
            # Address recognizer specifically needs case-sensitive
            # matching: street names start with capital letter and that
            # uppercase requirement is what keeps "przy wchodzeniu po
            # schodach" from matching as a fake address.
            global_regex_flags=_CASE_SENSITIVE_FLAGS,
        )


# ---------------------------------------------------------------------------
# Polish PERSON name list booster — supplements spaCy's NER.
# ---------------------------------------------------------------------------


class PolishNameListBooster(PatternRecognizer):
    """Catches Polish names that spaCy `pl_core_news_lg` PERSON misses.

    spaCy NER on Polish surnames recalls ~85%; this is the gap-closer
    for our test corpus. Built from `patient_full_name` strings in
    `dataset_pipeline/output/scenarios/*.json` plus an optional
    additional list passed in.

    Implementation: each name is split on whitespace into one or more
    tokens; both the full name and individual tokens of length ≥3 are
    added as patterns. Same-original-same-token semantics in the
    anonymizer means "Kowalski" alone and "Jan Kowalski" both map to
    the same `<PERSON_n>` slot if seen in one session.

    Empty-list construction is a no-op fallback (used in tests where we
    don't want to read disk). Presidio rejects empty `patterns`, so we
    insert a single never-matching placeholder.
    """

    def __init__(
        self,
        names: Iterable[str] | None = None,
        supported_language: str = "pl",
        supported_entity: str = "PERSON",
    ) -> None:
        names_list = sorted({n.strip() for n in (names or []) if n.strip()})
        patterns: list[Pattern] = []
        for full in names_list:
            patterns.append(
                Pattern(
                    name=f"pl_name_full__{full}",
                    regex=r"\b" + re.escape(full) + r"\b",
                    score=1.0,
                )
            )
            for token in full.split():
                if len(token) >= 3:
                    patterns.append(
                        Pattern(
                            name=f"pl_name_token__{token}",
                            regex=r"\b" + re.escape(token) + r"\b",
                            score=0.85,
                        )
                    )
        if not patterns:
            patterns = [Pattern(name="pl_name_noop", regex=r"(?!x)x", score=0.0)]
        super().__init__(
            supported_entity=supported_entity,
            patterns=patterns,
            supported_language=supported_language,
        )


def load_scenario_names(scenarios_dir: Path) -> list[str]:
    """Read every scenario JSON and return the `patient_full_name` list.

    Tolerates missing directory / unreadable files — returns whatever it
    could parse. Used by the anonymizer factory to seed the name booster.
    """
    if not scenarios_dir.is_dir():
        return []
    out: list[str] = []
    for path in sorted(scenarios_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:  # pragma: no cover — defensive
            log.warning("could not read scenario %s: %s", path, exc)
            continue
        name = data.get("patient_full_name")
        if isinstance(name, str) and name.strip():
            out.append(name.strip())
    return out


def build_polish_recognizers(
    *,
    name_booster_names: Iterable[str] | None = None,
) -> list[PatternRecognizer]:
    """Factory — returns all Polish recognizers in priority order.

    Priority order matches operator-priority from DD-10: PESEL → NIP →
    ID_CARD → PHONE → ADDRESS → name list. PERSON via spaCy is added by
    the analyzer engine, not here.
    """
    return [
        PolishPESELRecognizer(),
        PolishNIPRecognizer(),
        PolishIDCardRecognizer(),
        PolishPhoneRecognizer(),
        PolishAddressRecognizer(),
        PolishNameListBooster(names=name_booster_names),
    ]
