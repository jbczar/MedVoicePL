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
    # `À-ſ` covers Latin-1 supplement + Latin Extended-A —
    # safety net for any accented Latin glyphs that escape `\w`.
    # `<` and `>` are allowed because clinical text routinely uses
    # them as comparison operators (`>5 kg`, `<100 mmHg`); the
    # placeholder pattern `<[A-Z_]+_\d+>` is still preferred via
    # alternation, just no longer the only legal use.
    # `*` and `+` allowed for "BMI 24+", lab results.
    "^(?:[\\w\\s\\.,;:?!\\-/()\"'<>+*=]|[À-ſ]|<[A-Z_]+_\\d+>)*$",
    re.UNICODE,
)


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
        if not _CHARSET_RE.match(value):
            violations.append(
                Violation(location=path, kind="charset", detail="disallowed character")
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
