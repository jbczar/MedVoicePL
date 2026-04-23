"""Diarization Error Rate for 2-speaker clinical dialogues.

Pure-Python DER — no `pyannote.metrics` dep. Same posture as
`wer_calculator.py`: importing this file does NOT require torch or any
ML stack. Good for CI and for being called from both the eval script
and unit tests.

DER = (false_alarm + missed_detection + speaker_confusion) / total_reference_speech

Definitions (per standard NIST RT-03 formulation, applied to our
2-speaker case):
  * `missed_detection`: reference says someone is speaking, hypothesis says silence.
  * `false_alarm`: hypothesis says someone is speaking, reference says silence.
  * `speaker_confusion`: both say speech, but disagree on who.
  * `total_reference_speech`: sum of reference segment durations (with
    overlaps counted once — we collapse).

We do NOT support the `collar` or `skip_overlap` options from
pyannote.metrics: clinic recordings are two people taking turns with
clean boundaries; a collar would just hide genuine merge errors.

Implementation strategy: union-of-endpoints time-slicing.
  1. Collect all (start, end) boundaries from both timelines.
  2. Sort, dedupe → N+1 split points → N atomic intervals.
  3. For each atomic interval, look up who (if anyone) is speaking on
     each side using bisect.
  4. Accumulate miss / false-alarm / confusion durations.
Complexity: O((R + H) log (R + H)) where R, H are segment counts —
fine for the sub-minute-per-scenario budget we have.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import TypeAlias

# A labeled interval: [start_s, end_s) with a speaker label.
# Label None means "no speech" but we never store those explicitly —
# gaps between intervals are silence by construction.
Segment: TypeAlias = tuple[float, float, str]


@dataclass(frozen=True)
class DERResult:
    """Decomposed diarization error rate."""

    miss_s: float
    false_alarm_s: float
    confusion_s: float
    reference_speech_s: float

    @property
    def rate(self) -> float:
        if self.reference_speech_s <= 0:
            return 0.0
        return (self.miss_s + self.false_alarm_s + self.confusion_s) / self.reference_speech_s

    @property
    def total_error_s(self) -> float:
        return self.miss_s + self.false_alarm_s + self.confusion_s


def _normalize(segments: list[Segment]) -> list[Segment]:
    """Sort by start, drop zero-length, clip overlaps within the SAME speaker.

    We do NOT merge adjacent same-speaker gaps — they carry info about
    diarizer turn-taking we want preserved. We DO collapse overlapping
    spans from the same speaker into one (happens with reference
    timelines where a turn got double-counted).
    """
    cleaned = [(s, e, label) for s, e, label in segments if e > s]
    cleaned.sort(key=lambda x: (x[0], x[1]))
    out: list[Segment] = []
    for s, e, label in cleaned:
        if out and out[-1][2] == label and s <= out[-1][1]:
            prev_s, prev_e, _ = out[-1]
            out[-1] = (prev_s, max(prev_e, e), label)
        else:
            out.append((s, e, label))
    return out


def _speaker_at(segments: list[Segment], starts: list[float], t: float) -> str | None:
    """Return the speaker label active at time `t`, or None if silence.

    `starts` must be the sorted list of segment start times (cached).
    Assumes `segments` is sorted by start and non-overlapping per
    speaker after `_normalize`. For the 2-speaker clinical case
    reference and hypothesis timelines are single-track (no overlapping
    speakers), so one span can cover any instant.
    """
    if not segments:
        return None
    idx = bisect_right(starts, t) - 1
    if idx < 0:
        return None
    s, e, label = segments[idx]
    if s <= t < e:
        return label
    return None


def diarization_error_rate(
    reference: list[Segment],
    hypothesis: list[Segment],
    *,
    label_mapping: dict[str, str] | None = None,
) -> DERResult:
    """Compute DER from reference vs. hypothesis segment lists.

    `reference` and `hypothesis` are lists of `(start_s, end_s, label)`.
    Labels are compared verbatim after optional `label_mapping`
    (applied to HYPOTHESIS labels only — reference is ground truth).

    Typical call from the eval script::

        ref = [(0.0, 3.2, "doctor"), (3.2, 7.1, "patient"), ...]
        hyp = [(seg.start_s, seg.end_s, seg.speaker) for seg in transcript.segments]
        result = diarization_error_rate(ref, hyp)
        print(f"DER: {result.rate:.2%}")
    """
    ref = _normalize(reference)
    hyp = _normalize(hypothesis)

    if label_mapping:
        hyp = [(s, e, label_mapping.get(lbl, lbl)) for s, e, lbl in hyp]

    # Reference speech time — what DER is normalized against.
    ref_speech_s = sum(e - s for s, e, _ in ref)

    # Build sorted boundary list → atomic intervals.
    boundaries: set[float] = set()
    for s, e, _ in ref:
        boundaries.add(s)
        boundaries.add(e)
    for s, e, _ in hyp:
        boundaries.add(s)
        boundaries.add(e)
    if not boundaries:
        return DERResult(0.0, 0.0, 0.0, 0.0)

    sorted_boundaries = sorted(boundaries)
    ref_starts = [s for s, _, _ in ref]
    hyp_starts = [s for s, _, _ in hyp]

    miss = 0.0
    fa = 0.0
    conf = 0.0

    for i in range(len(sorted_boundaries) - 1):
        t0 = sorted_boundaries[i]
        t1 = sorted_boundaries[i + 1]
        dur = t1 - t0
        if dur <= 0:
            continue
        # Probe the midpoint — atomic intervals have constant speaker
        # on each side by construction.
        mid = (t0 + t1) / 2
        r = _speaker_at(ref, ref_starts, mid)
        h = _speaker_at(hyp, hyp_starts, mid)

        if r is None and h is None:
            continue                            # both silent
        if r is None and h is not None:
            fa += dur
        elif r is not None and h is None:
            miss += dur
        elif r == h:
            continue                            # correct speech
        else:
            conf += dur

    return DERResult(
        miss_s=miss,
        false_alarm_s=fa,
        confusion_s=conf,
        reference_speech_s=ref_speech_s,
    )


def reference_from_turn_durations(
    turn_durations_s: list[float],
    turn_speakers: list[str],
    *,
    pause_between_turns_s: float = 0.0,
) -> list[Segment]:
    """Build a reference timeline from per-turn durations + speakers.

    Matches how `dataset_pipeline.postprocessing.dialog_mixer` lays out
    turns: turn N starts at `sum(prev_durations) + N * pause`. For DER
    ground truth we use actual per-turn WAV durations (from the
    `ScenarioAudioBundle`) so the reference reflects what ended up in
    `dialogue.wav` after crossfade; the small crossfade overlap is
    below the measurement noise floor.

    Returns a sorted list of `(start_s, end_s, speaker)` suitable for
    `diarization_error_rate(reference=...)`.
    """
    if len(turn_durations_s) != len(turn_speakers):
        raise ValueError("turn_durations_s and turn_speakers must have equal length")
    out: list[Segment] = []
    cursor = 0.0
    for dur, speaker in zip(turn_durations_s, turn_speakers, strict=True):
        if dur <= 0:
            cursor += pause_between_turns_s
            continue
        out.append((cursor, cursor + dur, speaker))
        cursor += dur + pause_between_turns_s
    return out
