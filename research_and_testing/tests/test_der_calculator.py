"""Unit tests for the pure-Python Diarization Error Rate calculator."""

from __future__ import annotations

import pytest

from evaluation.stt_eval.der_calculator import (
    diarization_error_rate,
    reference_from_turn_durations,
)


def test_identity_case_is_zero() -> None:
    """Hypothesis == reference → DER 0.0."""
    ref = [
        (0.0, 2.0, "doctor"),
        (2.0, 5.0, "patient"),
        (5.0, 6.0, "doctor"),
    ]
    hyp = list(ref)
    result = diarization_error_rate(ref, hyp)
    assert result.rate == 0.0
    assert result.miss_s == 0.0
    assert result.false_alarm_s == 0.0
    assert result.confusion_s == 0.0
    assert result.reference_speech_s == 6.0


def test_pure_confusion_entire_timeline_swapped() -> None:
    """Swap speakers → all speech time is confusion."""
    ref = [
        (0.0, 3.0, "doctor"),
        (3.0, 6.0, "patient"),
    ]
    hyp = [
        (0.0, 3.0, "patient"),
        (3.0, 6.0, "doctor"),
    ]
    result = diarization_error_rate(ref, hyp)
    assert result.confusion_s == 6.0
    assert result.miss_s == 0.0
    assert result.false_alarm_s == 0.0
    assert result.rate == 1.0


def test_pure_missed_detection() -> None:
    """Reference says speech, hypothesis says silence."""
    ref = [
        (0.0, 4.0, "doctor"),
    ]
    hyp: list[tuple[float, float, str]] = []
    result = diarization_error_rate(ref, hyp)
    assert result.miss_s == 4.0
    assert result.false_alarm_s == 0.0
    assert result.confusion_s == 0.0
    assert result.rate == 1.0


def test_pure_false_alarm() -> None:
    """Hypothesis fabricates speech on a silent reference."""
    ref: list[tuple[float, float, str]] = []
    hyp = [(0.0, 2.5, "doctor")]
    result = diarization_error_rate(ref, hyp)
    assert result.false_alarm_s == 2.5
    assert result.miss_s == 0.0
    assert result.confusion_s == 0.0
    # No reference speech → rate degenerates to 0.0 (matches
    # pyannote.metrics behavior: DER undefined; we return 0).
    assert result.rate == 0.0


def test_partial_overlap_mix_of_errors() -> None:
    """Overlap creates one correct region, one miss, one false alarm.

    DER denominator is reference-speech, so miss 2s + fa 2s over 4s of
    reference speech = rate 1.0 (matches pyannote.metrics formulation).
    """
    # Ref:    [D 0-4]
    # Hyp:        [D 2-6]
    # Correct: 2..4 (2s), Miss: 0..2 (2s), FA: 4..6 (2s)
    ref = [(0.0, 4.0, "doctor")]
    hyp = [(2.0, 6.0, "doctor")]
    result = diarization_error_rate(ref, hyp)
    assert result.miss_s == pytest.approx(2.0)
    assert result.false_alarm_s == pytest.approx(2.0)
    assert result.confusion_s == 0.0
    assert result.reference_speech_s == pytest.approx(4.0)
    assert result.rate == pytest.approx(1.0)


def test_label_mapping_applied_to_hypothesis() -> None:
    """SPEAKER_00 → doctor mapping turns confusion into correct."""
    ref = [(0.0, 3.0, "doctor")]
    hyp = [(0.0, 3.0, "SPEAKER_00")]
    unmapped = diarization_error_rate(ref, hyp)
    assert unmapped.confusion_s == 3.0
    mapped = diarization_error_rate(
        ref, hyp, label_mapping={"SPEAKER_00": "doctor"}
    )
    assert mapped.rate == 0.0


def test_reference_from_turn_durations_lays_out_sequentially() -> None:
    segs = reference_from_turn_durations(
        [1.0, 2.0, 0.5],
        ["doctor", "patient", "doctor"],
        pause_between_turns_s=0.25,
    )
    assert segs == [
        (0.0, 1.0, "doctor"),
        (1.25, 3.25, "patient"),
        (3.5, 4.0, "doctor"),
    ]


def test_reference_from_turn_durations_length_mismatch() -> None:
    with pytest.raises(ValueError, match="equal length"):
        reference_from_turn_durations([1.0, 2.0], ["doctor"])
