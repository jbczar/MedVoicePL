"""Word- and character-error-rate for Polish transcriptions.

Pure-Python Levenshtein on tokens (WER) and characters (CER). Kept here,
not pulled from `jiwer`, so the eval module stays dep-free: importing
this file does NOT require torch, faster-whisper, or any ML stack.

Normalization is deliberately simple:
  - casefold (Polish-aware via `str.casefold`)
  - strip basic punctuation
  - collapse whitespace

Medical abbreviations are NOT expanded — that's an upstream model's
responsibility. We measure what the recognizer produced verbatim.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Casefold, strip all Unicode punctuation, collapse whitespace."""
    # NFKC first so fancy quotes / ligatures normalize before stripping.
    text = unicodedata.normalize("NFKC", text)
    # Drop anything classified as punctuation (P*) — covers ASCII and
    # Polish/Unicode quotes like „ " « » — … etc.
    text = "".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in text)
    text = _WS_RE.sub(" ", text).strip()
    return text.casefold()


def _levenshtein(ref: list[str], hyp: list[str]) -> int:
    """Classic DP edit distance on arbitrary sequences."""
    n, m = len(ref), len(hyp)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,       # deletion
                curr[j - 1] + 1,   # insertion
                prev[j - 1] + cost,  # substitution
            )
        prev = curr
    return prev[m]


@dataclass(frozen=True)
class ErrorRate:
    """Edit distance + reference length + derived rate."""

    edits: int
    length: int

    @property
    def rate(self) -> float:
        return self.edits / self.length if self.length else 0.0


def wer(reference: str, hypothesis: str) -> ErrorRate:
    """Word Error Rate after normalization."""
    ref = normalize(reference).split()
    hyp = normalize(hypothesis).split()
    return ErrorRate(edits=_levenshtein(ref, hyp), length=len(ref))


def cer(reference: str, hypothesis: str) -> ErrorRate:
    """Character Error Rate after normalization (whitespace preserved)."""
    ref = list(normalize(reference))
    hyp = list(normalize(hypothesis))
    return ErrorRate(edits=_levenshtein(ref, hyp), length=len(ref))
