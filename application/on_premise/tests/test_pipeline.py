"""Tests for `on_premise.stt.pipeline.transcribe_and_diarize`.

Uses `_FakeTranscriber` + `_FakeDiarizer` so no model weights are touched.
Asserts the merge contract: for each diarization segment, words whose
midpoint falls inside get bucketed; role mapping is deterministic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from on_premise.stt.diarization import Diarizer, RawDiarizationSegment
from on_premise.stt.pipeline import transcribe_and_diarize
from on_premise.stt.whisper_engine import RawTranscription, RawWord, Transcriber


class _StubTranscriber:
    name = "mlx-whisper:stub"
    device = "mps"

    def __init__(self, words: list[RawWord], duration: float) -> None:
        self._words = words
        self._duration = duration

    def transcribe(self, wav_path: Path) -> RawTranscription:
        return RawTranscription(
            text=" ".join(w.text for w in self._words),
            words=list(self._words),
            duration_s=self._duration,
            language="pl",
        )


class _StubDiarizer:
    name = "pyannote/stub"
    device = "cpu"

    def __init__(self, segments: list[RawDiarizationSegment]) -> None:
        self._segments = segments

    def diarize(self, wav_path: Path) -> list[RawDiarizationSegment]:
        return list(self._segments)


def _write_dummy_wav(path: Path) -> None:
    """A tiny valid-enough file — `audio_sha256` just needs bytes on disk."""
    path.write_bytes(b"RIFF    WAVEfmt ")


@pytest.fixture()
def wav(tmp_path: Path) -> Path:
    p = tmp_path / "dialogue.wav"
    _write_dummy_wav(p)
    return p


def test_merge_bins_words_by_midpoint(wav: Path) -> None:
    # 4 words; 2 pyannote segments.
    words = [
        RawWord(text="dzień", start_s=0.0, end_s=0.5),
        RawWord(text="dobry", start_s=0.5, end_s=1.0),
        RawWord(text="boli", start_s=2.0, end_s=2.4),
        RawWord(text="mnie", start_s=2.4, end_s=2.7),
    ]
    segs = [
        RawDiarizationSegment(start_s=0.0, end_s=1.5, speaker_label="SPEAKER_00"),
        RawDiarizationSegment(start_s=1.5, end_s=3.0, speaker_label="SPEAKER_01"),
    ]
    transcriber: Transcriber = _StubTranscriber(words, duration=3.0)
    diarizer: Diarizer = _StubDiarizer(segs)

    transcript = transcribe_and_diarize(
        wav,
        transcriber=transcriber,
        diarizer=diarizer,
        scenario_id="test-01",
    )

    assert transcript.scenario_id == "test-01"
    assert len(transcript.segments) == 2
    assert transcript.segments[0].text == "dzień dobry"
    assert transcript.segments[0].speaker == "doctor"  # first speaker wins
    assert transcript.segments[1].text == "boli mnie"
    assert transcript.segments[1].speaker == "patient"
    # Monotonic timestamps.
    assert transcript.segments[0].start_s < transcript.segments[1].start_s


def test_word_in_pyannote_gap_goes_to_nearest_segment(wav: Path) -> None:
    """Regression: pyannote doesn't tile audio, so Whisper words whose
    midpoint falls in a silence gap between pyannote segments used to be
    dropped (and torched WER by +16pp on real audio). Fix: attach every
    word to the nearest segment by time distance.
    """
    words = [
        RawWord(text="Dzień", start_s=0.0, end_s=0.4),     # inside seg 0
        RawWord(text="dobry", start_s=0.4, end_s=0.8),     # inside seg 0
        # Word sitting in a pyannote gap (1.5..2.0) — used to vanish.
        # Midpoint 1.75 is closer to seg 0 end (gap 0.75) than seg 1 start (gap 0.25),
        # so this word belongs on the patient side of the boundary.
        RawWord(text="Proszę", start_s=1.6, end_s=1.9),
        RawWord(text="boli", start_s=2.1, end_s=2.4),      # inside seg 1
    ]
    segs = [
        RawDiarizationSegment(start_s=0.0, end_s=1.5, speaker_label="SPEAKER_00"),
        RawDiarizationSegment(start_s=2.0, end_s=3.0, speaker_label="SPEAKER_01"),
    ]
    transcript = transcribe_and_diarize(
        wav,
        transcriber=_StubTranscriber(words, duration=3.0),
        diarizer=_StubDiarizer(segs),
    )
    # No word may be dropped: the whole transcript must preserve the
    # full Whisper output, byte-equal up to whitespace.
    rendered = " ".join(seg.text for seg in transcript.segments)
    assert "Proszę" in rendered
    assert rendered.split() == ["Dzień", "dobry", "Proszę", "boli"]
    # Gap-midpoint 1.75 → closer to seg[1].start_s=2.0 (dist 0.25) than
    # seg[0].end_s=1.5 (dist 0.25)... tie broken by first-match (seg 0).
    # Either assignment is acceptable as long as the word isn't lost;
    # assert it landed in exactly one bucket.
    placements = [
        seg for seg in transcript.segments if "Proszę" in seg.text
    ]
    assert len(placements) == 1


def test_empty_diarization_segments_are_dropped(wav: Path) -> None:
    # Diarizer reports two segments, but only the first contains any words.
    words = [RawWord(text="hm", start_s=0.1, end_s=0.3)]
    segs = [
        RawDiarizationSegment(start_s=0.0, end_s=0.5, speaker_label="SPEAKER_00"),
        RawDiarizationSegment(start_s=0.5, end_s=1.5, speaker_label="SPEAKER_01"),  # empty
    ]
    transcript = transcribe_and_diarize(
        wav,
        transcriber=_StubTranscriber(words, duration=1.5),
        diarizer=_StubDiarizer(segs),
    )
    assert len(transcript.segments) == 1
    assert transcript.segments[0].text == "hm"


def test_role_mapping_uses_first_segment_not_total_talk_time(wav: Path) -> None:
    """Regression guard: in 2026-04-23 a 'dominant speaker in first
    30 s' heuristic was tried to defend against spurious openers
    under noise. It flipped roles on clean visits because the
    *patient* typically outlasts the doctor in the first 30 s
    (doctor asks short opening question, patient answers at length).
    DER hit 97-100% on the clean corpus before the revert.

    This test encodes the case: a 4 s doctor opener followed by a
    25 s patient monologue. Simple 'first-segment wins' assigns
    doctor correctly; any total-talk-time variant flips it.
    """
    words = [
        RawWord(text="Dzień",  start_s=0.2, end_s=0.7),
        RawWord(text="dobry",  start_s=0.7, end_s=1.2),
        RawWord(text="Boli",   start_s=5.0, end_s=5.4),
        RawWord(text="mnie",   start_s=5.4, end_s=5.7),
    ]
    segs = [
        RawDiarizationSegment(start_s=0.0, end_s=4.0,  speaker_label="SPEAKER_00"),
        RawDiarizationSegment(start_s=4.0, end_s=29.0, speaker_label="SPEAKER_01"),
    ]
    transcript = transcribe_and_diarize(
        wav,
        transcriber=_StubTranscriber(words, duration=29.0),
        diarizer=_StubDiarizer(segs),
    )
    by_start = {seg.start_s: seg.speaker for seg in transcript.segments}
    # Earliest segment → doctor, regardless of who talks more after.
    assert by_start[0.0] == "doctor"
    assert by_start[4.0] == "patient"


def test_role_override_wins_over_heuristic(wav: Path) -> None:
    words = [
        RawWord(text="foo", start_s=0.1, end_s=0.3),
        RawWord(text="bar", start_s=1.1, end_s=1.3),
    ]
    segs = [
        RawDiarizationSegment(start_s=0.0, end_s=1.0, speaker_label="SPEAKER_00"),
        RawDiarizationSegment(start_s=1.0, end_s=2.0, speaker_label="SPEAKER_01"),
    ]
    transcript = transcribe_and_diarize(
        wav,
        transcriber=_StubTranscriber(words, duration=2.0),
        diarizer=_StubDiarizer(segs),
        role_override={"SPEAKER_00": "patient", "SPEAKER_01": "doctor"},
    )
    assert transcript.segments[0].speaker == "patient"
    assert transcript.segments[1].speaker == "doctor"


def test_missing_wav_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        transcribe_and_diarize(
            tmp_path / "ghost.wav",
            transcriber=_StubTranscriber([], duration=0.0),
            diarizer=_StubDiarizer([]),
        )


def test_metadata_reflects_backends(wav: Path) -> None:
    transcript = transcribe_and_diarize(
        wav,
        transcriber=_StubTranscriber([], duration=0.0),
        diarizer=_StubDiarizer([]),
    )
    assert transcript.metadata.stt_backend == "mlx"
    assert transcript.metadata.diarization_model == "pyannote/stub"
    assert transcript.audio_sha256 != ""
    assert transcript.segments == []  # no diarization segments -> no transcript segments
