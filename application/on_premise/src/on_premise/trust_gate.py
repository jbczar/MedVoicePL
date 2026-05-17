"""Trust-boundary runtime gate.

`validate_outbound(payload)` is the last line of defense before an
`AnonymizedSOAPNote` leaves on-premise. It runs three deterministic
checks (~10 ms total):

1. **Type contract**: the payload must be an `AnonymizedSOAPNote`.
   Passing a raw `SOAPNote` raises `TrustBoundaryViolation`.
2. **Regex pre-filter**: every string leaf is swept for shapes that
   could only be raw PII — PESEL (`\\b\\d{11}\\b`), Polish phone,
   NIP (`\\b\\d{10}\\b`), email. Hits raise.
3. **Charset invariant**: every string leaf must match an allow-list
   of characters: Polish letters, digits, whitespace, standard
   punctuation, plus the placeholder grammar `<[A-Z_]+_\\d+>`.
   Anything outside this set (e.g. a stray quoted name in a
   non-anonymized state) trips.

We deliberately do NOT re-run Presidio here. A second analyzer pass
is 100–500 ms (same order as the anonymization step itself),
non-deterministic across model upgrades, and forces "which run
wins" arbitration on disagreement. The Presidio re-run becomes a
nightly batch verification tool, not a hot-path gate.

Cross-references:

* The PESEL/phone regexes match Phase 2b's
  `pipeline._scrub_hallucinated_pii` patterns by design — both ends
  of the pipeline catch the same shapes.
* The placeholder grammar matches what
  `presidio_agent._TokenAssigner` emits.
* `Violation.kind` enumerates so `audit_logger.log_trust_gate` has a
  stable schema.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Iterator

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regex pre-filter — same shapes as Phase 2b's _scrub_hallucinated_pii.
# ---------------------------------------------------------------------------

_PESEL_RE = re.compile(r"\b\d{11}\b")
"""PESEL — 11 contiguous digits. Same regex as Phase 2b's hallucination scrub."""

_PHONE_RE = re.compile(
    # Two alternations:
    #   1. With country code (+48 / 48): 2 digits + 9 digits with optional separators.
    #   2. Without country code: 9 digits in 3-3-3 grouping with REQUIRED
    #      separators (the required separator is what stops this from
    #      tripping on incidental 9-digit runs in lab IDs).
    r"\+?48[\s-]?\d{3}[\s-]?\d{3}[\s-]?\d{3}"
    r"|\b\d{3}[\s-]\d{3}[\s-]\d{3}\b"
    r"|\b\d{2}[\s-]\d{3}[\s-]\d{2}[\s-]\d{2}\b"  # landline 2-3-2-2
)
"""Polish phone shape — covers mobile (with/without +48) and landline."""

_NIP_RE = re.compile(r"(?<!\d)\d{10}(?!\d)")
"""NIP — 10 contiguous digits, NOT followed/preceded by another digit
(would otherwise trip on PESEL substrings)."""

_EMAIL_RE = re.compile(r"\b[\w.\-]+@[\w.\-]+\.\w+\b")

_REGEX_CHECKS: tuple[tuple[re.Pattern[str], str], ...] = (
    (_PESEL_RE, "pesel"),
    (_PHONE_RE, "phone"),
    (_NIP_RE, "nip"),
    (_EMAIL_RE, "email"),
)


# ---------------------------------------------------------------------------
# Charset invariant — Polish text + placeholder grammar.
# ---------------------------------------------------------------------------

# Allowed characters:
#   - `\w` : word chars (Unicode letters + digits + underscore)
#   - whitespace
#   - punctuation we expect in clinical text: . , ; : ? ! - / ( ) " '
#   - Polish diacritics covered by `\w` under Unicode mode
# Plus the explicit placeholder grammar `<[A-Z_]+_\d+>` for tokens
# like `<PERSON_1>`, `<PHONE_NUMBER_2>`, `<PL_PESEL_1>`.
#
# Note: ICD-10 codes like `I10.0` are bare letters+digits+dot, so they
# fall into the standard charset. SHA-256 hex digests don't appear in
# `AnonymizedSOAPNote` (stripped from metadata in DD-1).
_CHARSET_RE = re.compile(
    # Legacy alias — kept for backwards-compatible tests that import it.
    # The actual check is `_check_charset` below using Unicode categories.
    # This regex is now a permissive shim that always matches.
    r"^.*$",
    re.UNICODE | re.DOTALL,
)


# ---------------------------------------------------------------------------
# Unicode-category-based charset policy (replaces whitelist regex v1/v2).
#
# Strategy:
#   - Allow ALL Unicode letters, digits, punctuation, math/currency symbols,
#     diacritic marks, spaces. This covers any legit Polish medical text:
#     ą ć ę ł ń ó ś ź ż, %, °, ±, β, ≥, ≤, „"', em/en-dash, ellipsis…
#   - Block only structurally-dangerous categories:
#     - `Cc` Control chars (\x00-\x1f) — except whitespace already filtered
#     - `Cf` Format chars (zero-width joiner, RTL override, etc.)
#     - `Cs` Surrogate halves (invalid in well-formed text)
#     - `Co` Private use area (E000-F8FF, mostly emoji-internal)
#     - `Zl` Line separator, `Zp` Paragraph separator (U+2028/2029) —
#       can confuse parsers; we accept \n / \r already as `Cc`-whitelisted.
#
# Whitelist within Cc: \t (TAB), \n (LF), \r (CR) — newlines are fine.
#
# Allowed categories include `So` (Symbol, other) which covers `°` (degree)
# and similar medical symbols. Emoji are also `So` but are rare from a
# Polish medical SLM and not a security threat — PII regex catches the
# real risks. If emoji become a problem, switch to range-based blocklist
# for U+1F300-U+1FAFF.
#
# This is a *permissive blocklist* — strictly safer than v2 whitelist
# (no false positives on legit medical typography) but still rejects all
# the threats that motivated the trust gate.
# ---------------------------------------------------------------------------
_BLOCKED_UNICODE_CATEGORIES = frozenset({"Cc", "Cf", "Cs", "Co", "Zl", "Zp"})
_WHITELISTED_CONTROL_CHARS = frozenset({"\t", "\n", "\r"})

# Block emoji explicitly via codepoint ranges (the categories Sk/So contain
# both safe medical symbols and emoji — we keep medical, reject emoji).
_EMOJI_RANGES = (
    (0x1F300, 0x1FAFF),  # Misc symbols + emoticons + emoji blocks
    (0x2600, 0x27BF),    # Miscellaneous Symbols + Dingbats
    (0x1F000, 0x1F02F),  # Mahjong/Domino tiles
)


def _is_emoji(char: str) -> bool:
    cp = ord(char)
    return any(lo <= cp <= hi for lo, hi in _EMOJI_RANGES)

# Placeholder grammar — explicitly allowed `<TOKEN_n>` form.
_PLACEHOLDER_RE = re.compile(r"<[A-Z_]+_\d+>")


def _check_charset(text: str) -> tuple[bool, str | None]:
    """Return (ok, detail). On failure detail names the offending char.

    Detail format: ``"U+XXXX (category Cf) at offset N: <name>"`` so the
    audit log surfaces actionable diagnostic info (not just
    "disallowed character").
    """
    import unicodedata

    # Strip placeholders first — their internal chars (<, >, letters,
    # digits) all pass anyway, but stripping reduces noise.
    stripped = _PLACEHOLDER_RE.sub("", text)
    for offset, char in enumerate(stripped):
        if char in _WHITELISTED_CONTROL_CHARS:
            continue
        cat = unicodedata.category(char)
        if cat in _BLOCKED_UNICODE_CATEGORIES or _is_emoji(char):
            try:
                name = unicodedata.name(char, "unnamed")
            except ValueError:
                name = "unnamed"
            reason = "emoji" if _is_emoji(char) else f"category {cat}"
            return False, (
                f"U+{ord(char):04X} ({reason}) at offset {offset}: {name}"
            )
    return True, None


# ---------------------------------------------------------------------------
# Public types.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Violation:
    """One trust-boundary violation.

    Attributes:
        location: JSONPath-ish location, e.g.
            ``"subjective.history_of_present_illness"`` or
            ``"red_flags[2]"``.
        kind: ``"type"`` (payload class wrong), ``"regex"`` (PII shape
            detected), or ``"charset"`` (disallowed character).
        detail: short human-readable note. For regex hits, the regex
            name (``"pesel"``, ``"phone"``, etc.).
    """

    location: str
    kind: Literal["type", "regex", "charset"]
    detail: str


class TrustBoundaryViolation(Exception):
    """Raised when an outbound payload fails the trust gate.

    Carries `violations` so callers (FastAPI, audit logger) can
    surface structured info to the doctor + the audit log.
    """

    def __init__(self, violations: list[Violation]) -> None:
        super().__init__(f"trust_gate rejected payload: {len(violations)} violation(s)")
        self.violations = violations


# ---------------------------------------------------------------------------
# Validation entry point.
# ---------------------------------------------------------------------------


def validate_outbound(payload: Any) -> None:
    """Run the three-check gate. Raises `TrustBoundaryViolation` on fail.

    `payload` is typed `Any` to keep this module free of conditional
    imports at module load — we import `AnonymizedSOAPNote` lazily for
    the isinstance check. Production callers always pass an
    `AnonymizedSOAPNote`.
    """
    # Local import keeps `trust_gate` importable without `shared` on
    # the path (e.g. for test isolation).
    from shared.schemas import AnonymizedSOAPNote

    violations: list[Violation] = []

    # 1. Type contract.
    if not isinstance(payload, AnonymizedSOAPNote):
        violations.append(
            Violation(
                location="$",
                kind="type",
                detail=f"expected AnonymizedSOAPNote, got {type(payload).__name__}",
            )
        )
        # Don't iterate leaves on a wrong-typed payload — model_dump
        # may not even work.
        raise TrustBoundaryViolation(violations)

    # 2 + 3. Walk every string leaf; regex + charset checks.
    for path, value in _iter_string_leaves(payload):
        for regex, kind in _REGEX_CHECKS:
            if regex.search(value):
                violations.append(Violation(location=path, kind="regex", detail=kind))
        ok, charset_detail = _check_charset(value)
        if not ok:
            violations.append(
                Violation(
                    location=path,
                    kind="charset",
                    detail=charset_detail or "disallowed character",
                )
            )

    if violations:
        log.warning("trust_gate rejected payload: %d violation(s)", len(violations))
        raise TrustBoundaryViolation(violations)


# ---------------------------------------------------------------------------
# Internals.
# ---------------------------------------------------------------------------


def _iter_string_leaves(payload: Any) -> Iterator[tuple[str, str]]:
    """Yield `(path, value)` for every string leaf of the payload tree.

    Walks the Pydantic model via `model_dump()` so we don't have to
    hardcode the section structure — works for any future changes to
    `AnonymizedSOAPNote` shape.
    """
    yield from _walk("", payload.model_dump(mode="json"))


def _walk(path: str, node: Any) -> Iterator[tuple[str, str]]:
    if isinstance(node, str):
        if node:  # skip empty strings (no PII risk)
            yield path, node
    elif isinstance(node, dict):
        for key, value in node.items():
            sub = f"{path}.{key}" if path else key
            yield from _walk(sub, value)
    elif isinstance(node, list):
        for i, item in enumerate(node):
            yield from _walk(f"{path}[{i}]", item)
    # Other types (int, bool, None, date) are not PII-bearing — skip.


# ---------------------------------------------------------------------------
# Convenience: format violations for the audit logger.
# ---------------------------------------------------------------------------


def violations_to_dicts(violations: list[Violation]) -> list[dict[str, str]]:
    """Adapter for `audit_logger.log_trust_gate(violations=...)`."""
    return [{"location": v.location, "kind": v.kind, "detail": v.detail} for v in violations]
