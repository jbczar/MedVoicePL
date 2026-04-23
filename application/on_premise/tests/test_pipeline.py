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


def test_role_survives_spurious_opening_segment(wav: Path) -> None:
    """Regression: `b3f0c1a9` noisy DER hit 76.8% because pyannote
    emitted a short spurious SPEAKER_01 segment at t=0 before the
    real doctor turn (SPEAKER_00) began. Old heuristic
    ('first segment by start_s wins') flipped the whole role map
    and every doctor line was scored against the patient reference.

    With the new dominant-speaker-in-opening-window heuristic, a
    1.5 s spurious segment can't outweigh a 13.5 s + 9.0 s doctor
    block inside the same 30 s window, so SPEAKER_00 correctly
    takes `doctor` and DER no longer blows up.
    """
    # 30-second opening window; mirrors the noisy b3f0c1a9 pattern:
    #  SPEAKER_01 [0.0, 1.5]  — spurious noise burst, 1.5 s
    #  SPEAKER_00 [1.5, 15.0] — doctor greeting + first question, 13.5 s
    #  SPEAKER_01 [15.0, 20.0] — patient replies, 5.0 s
    #  SPEAKER_00 [20.0, 29.0] — doctor follow-up, 9.0 s
    # Opening-window totals: SPEAKER_00=22.5s, SPEAKER_01=6.5s.
    words = [
        # One word per segment — enough to materialize the segment in output.
        RawWord(text="khm",        start_s=0.3,  end_s=0.9),
        RawWord(text="Dzień",      start_s=2.0,  end_s=2.5),
        RawWord(text="dobry",      start_s=2.5,  end_s=3.0),
        RawWord(text="Co",         start_s=5.0,  end_s=5.3),
        RawWord(text="Panu",       start_s=5.3,  end_s=5.7),
        RawWord(text="dolega",     start_s=5.7,  end_s=6.3),
        RawWord(text="Boli",       start_s=16.0, end_s=16.4),
        RawWord(text="mnie",       start_s=16.4, end_s=16.7),
        RawWord(text="głowa",      start_s=16.7, end_s=17.2),
        RawWord(text="Od",         start_s=22.0, end_s=22.3),
        RawWord(text="kiedy",      start_s=22.3, end_s=22.7),
    ]
    segs = [
        RawDiarizationSegment(start_s=0.0,  end_s=1.5,  speaker_label="SPEAKER_01"),
        RawDiarizationSegment(start_s=1.5,  end_s=15.0, speaker_label="SPEAKER_00"),
        RawDiarizationSegment(start_s=15.0, end_s=20.0, speaker_label="SPEAKER_01"),
        RawDiarizationSegment(start_s=20.0, end_s=29.0, speaker_label="SPEAKER_00"),
    ]
    transcript = transcribe_and_diarize(
        wav,
        transcriber=_StubTranscriber(words, duration=29.0),
        diarizer=_StubDiarizer(segs),
    )

    # SPEAKER_00 dominates the first 30 s (22.5 s vs 6.5 s), so it
    # MUST be labeled `doctor` even though SPEAKER_01 started first.
    by_label = {seg.start_s: seg.speaker for seg in transcript.segments}
    # The long greeting + follow-up segments belong to doctor…
    assert by_label[1.5] == "doctor"
    assert by_label[20.0] == "doctor"
    # …and the spurious opener + patient reply belong to patient.
    assert by_label[0.0] == "patient"
    assert by_label[15.0] == "patient"


def test_role_falls_back_when_every_segment_is_sub_threshold(wav: Path) -> None:
    """Degenerate case: no segment is >=2 s (pathological diarization
    on an ultra-fragmented clip). The heuristic can't pick a 'real
    opener', so it must fall back to 'earliest segment wins' so we
    don't return an empty role map and silently misattribute every
    word to `patient`.
    """
    words = [RawWord(text="hm", start_s=0.1, end_s=0.3)]
    segs = [
        # All four segments are <2 s — no "real opener" candidate.
        RawDiarizationSegment(start_s=0.0, end_s=1.0, speaker_label="SPEAKER_00"),
        RawDiarizationSegment(start_s=1.0, end_s=1.8, speaker_label="SPEAKER_01"),
        RawDiarizationSegment(start_s=1.8, end_s=3.3, speaker_label="SPEAKER_00"),
        RawDiarizationSegment(start_s=3.3, end_s=4.5, speaker_label="SPEAKER_01"),
    ]
    transcript = transcribe_and_diarize(
        wav,
        transcriber=_StubTranscriber(words, duration=4.5),
        diarizer=_StubDiarizer(segs),
    )
    # SPEAKER_00 started earliest → fallback picks it as doctor.
    assert transcript.segments[0].speaker == "doctor"


def test_role_picks_doctor_when_doctor_is_short_but_still_above_threshold(wav: Path) -> None:
    """Clean-scenario regression: on typical visits the doctor's
    greeting is short (~3-8 s) and the patient's opening answer is
    long (~15-20 s). A previous 'dominant speaker in first 30 s'
    heuristic flipped roles on these scenarios because the patient
    racked up more talk time in the window. The fix must still
    award doctor to the first >=2 s segment regardless of who
    dominates total talk time afterward.
    """
    # 4-second doctor greeting, then 25-second patient monologue.
    # The old dominant-speaker heuristic would mis-pick SPEAKER_01
    # (patient) as doctor because 25 s > 4 s in the 30 s window.
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
    # The 4-second opener belongs to the doctor even though the
    # patient outlasts them in the first 30 s.
    by_start = {seg.start_s: seg.speaker for seg in transcript.segments}
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
