"""Unit tests for the pure-Python WER/CER calculator."""

from __future__ import annotations

from evaluation.stt_eval.wer_calculator import cer, normalize, wer


def test_normalize_strips_punct_and_casefolds() -> None:
    assert normalize("Dzień DOBRY, Panie!") == "dzień dobry panie"


def test_normalize_handles_unicode_quotes() -> None:
    assert normalize("„Test\u2014OK\u201d") == "test ok"


def test_wer_identical_is_zero() -> None:
    r = wer("pan doktor mówi powoli", "pan doktor mówi powoli")
    assert r.edits == 0
    assert r.rate == 0.0


def test_wer_one_substitution() -> None:
    r = wer("pan doktor mówi powoli", "pan doktor mówi szybko")
    assert r.edits == 1
    assert r.length == 4
    assert r.rate == 0.25


def test_wer_ignores_punctuation() -> None:
    r = wer("Tak, oczywiście.", "Tak oczywiście")
    assert r.edits == 0


def test_cer_counts_character_edits() -> None:
    r = cer("tak", "taky")
    # one insertion
    assert r.edits == 1
    assert r.length == 3


def test_wer_empty_reference_returns_zero_length() -> None:
    r = wer("", "anything here")
    assert r.length == 0
    assert r.rate == 0.0  # degenerate case — no reference to score against
