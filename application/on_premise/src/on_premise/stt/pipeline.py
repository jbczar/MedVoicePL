"""High-level STT pipeline: audio WAV -> speaker-labeled `Transcript`.

Composition:
    whisper_engine.Transcriber  +  diarization.Diarizer  ->  schema.Transcript

Phase 2a deliverable — the function `transcribe_and_diarize()` is the entry
point Phase 2b (SLM) and Phase 2c (anonymization + trust gate) will call.

Merge strategy between Whisper (word-level text) and pyannote (speaker time
ranges): for each pyannote segment, collect the Whisper words whose midpoint
falls inside the segment. Empty pyannote segments (speech detected but no
words transcribed) are dropped.

Role mapping is deterministic and domain-specific: the speaker whose first
utterance starts earliest is labeled `doctor` (clinically true — Polish
medical visits open with a doctor greeting). Override via
`role_override={"SPEAKER_00": "patient"}` if a dataset breaks that
assumption.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from on_premise.stt.schema import (
    Transcript,
    TranscriptionMetadata,
    TranscriptSegment,
    WordTiming,
)

if TYPE_CHECKING:
    from on_premise.stt.diarization import Diarizer, RawDiarizationSegment
    from on_premise.stt.whisper_engine import RawWord, Transcriber

log = logging.getLogger(__name__)


SpeakerRole = Literal["doctor", "patient"]


def transcribe_and_diarize(
    wav_path: Path,
    *,
    transcriber: Transcriber,
    diarizer: Diarizer,
    scenario_id: str | None = None,
    role_override: dict[str, SpeakerRole] | None = None,
    speaker_classifier: object | None = None,
) -> Transcript:
    """Run Whisper + pyannote and return a typed, speaker-labeled transcript.

    `speaker_classifier` is an optional `SpeakerClassifier` (from
    `on_premise.stt.speaker_classifier`) that runs as a post-step over
    the diarized transcript. Content-based reclassification recovers
    most of the DER loss from `pyannote` clustering collapse on
    synthetic XTTSv2 voices. Default `None` preserves Phase 2a baseline.
    """
    wav_path = Path(wav_path)
    if not wav_path.exists():
        raise FileNotFoundError(wav_path)

    log.info("transcribing %s", wav_path.name)
    raw = transcriber.transcribe(wav_path)

    log.info("diarizing %s", wav_path.name)
    raw_segs = diarizer.diarize(wav_path)

    role_map = _build_role_map(raw_segs, override=role_override)
    segments = _merge_words_into_segments(raw.words, raw_segs, role_map)

    # `audio_sha256` is computed once so the transcript is reproducible across
    # reruns on unchanged input audio (mirrors dataset_pipeline's SHA anchors).
    audio_bytes = wav_path.read_bytes()
    audio_sha = hashlib.sha256(audio_bytes).hexdigest()

    metadata = TranscriptionMetadata(
        stt_backend="mlx" if transcriber.name.startswith("mlx-whisper") else "faster-whisper",
        stt_model=transcriber.name,
        stt_device=transcriber.device,
        stt_compute_type=getattr(transcriber, "compute_type", None),
        diarization_model=diarizer.name,
        diarization_device=diarizer.device,
        transcribed_at=datetime.now().astimezone(),
    )
    transcript = Transcript(
        scenario_id=scenario_id,
        audio_path=str(wav_path),
        audio_sha256=audio_sha,
        duration_s=max(raw.duration_s, _max_end(raw_segs)),
        language=raw.language,
        segments=segments,
        metadata=metadata,
    )

    if speaker_classifier is not None:
        log.info("running SLM speaker reclassifier on %s", wav_path.name)
        transcript = speaker_classifier.reclassify(transcript)  # type: ignore[attr-defined]

    return transcript


# ---------------------------------------------------------------------------
# Merge + role mapping
# ---------------------------------------------------------------------------


def _build_role_map(
    raw_segs: list[RawDiarizationSegment],
    *,
    override: dict[str, SpeakerRole] | None,
) -> dict[str, SpeakerRole]:
    """Assign `doctor` to whichever opaque label speaks first.

    Clinical visits open with the doctor greeting the patient — so
    the earliest pyannote segment is virtually always on the doctor
    side. Two fancier variants were tried in 2026-04-23 and both
    regressed:

    * "dominant speaker in first 30 s" — flipped roles on 6/8 clean
      scenarios because the patient typically talks longer than the
      doctor in the opening (doctor asks short questions, patient
      answers at length). DER hit 97-100%.
    * "first segment >=2 s wins" — flipped roles when pyannote's
      segmentation on TTS audio starts with a cascade of sub-second
      fragments and the first >=2 s segment lands mid-conversation
      on whichever speaker pyannote's clustering arbitrarily assigned
      to the "long" label. Same catastrophic DER.

    Both failed because on our synthetic TTS dataset pyannote's
    embedder (wespeaker) doesn't separate the two cloned voices
    reliably — the same physical speaker is relabeled between
    SPEAKER_00 and SPEAKER_01 within a single dialogue. Any
    heuristic that tries to identify "the speaker" from a later
    segment is sampling from that unstable labeling. The earliest
    segment is the one point where we've seen pyannote consistently
    put the doctor in whatever cluster it started with, so
    `min(start_s)` wins.

    If the heuristic misfires on a specific recording (e.g., real
    clinic audio with a noise burst at t=0), callers pass
    `override={"SPEAKER_00": "patient", ...}`. That's the escape
    hatch — don't bake more domain logic into the default path.
    """
    if override:
        return dict(override)
    if not raw_segs:
        return {}
    first_label = min(raw_segs, key=lambda s: s.start_s).speaker_label
    role_map: dict[str, SpeakerRole] = {}
    for seg in raw_segs:
        if seg.speaker_label == first_label:
            role_map[seg.speaker_label] = "doctor"
        else:
            role_map[seg.speaker_label] = "patient"
    return role_map


def _merge_words_into_segments(
    words: list[RawWord],
    raw_segs: list[RawDiarizationSegment],
    role_map: dict[str, SpeakerRole],
) -> list[TranscriptSegment]:
    """Assign every Whisper word to its nearest pyannote segment.

    Strict "midpoint inside segment" bucketing drops any word whose
    midpoint lands in a pyannote gap (silence between detected-speech
    spans). In practice ~15-25% of words fall in those gaps because
    pyannote's speech activity detector is tighter than Whisper's, so
    dropping them inflates WER by double digits vs. the bare-MLX
    baseline on the same audio.

    Strategy: for each word, pick the pyannote segment whose interval
    is time-closest to the word's midpoint — if the midpoint is inside
    a segment, distance is 0; otherwise it's the gap to the nearest
    edge. Words still attach to at most one segment, so DER and
    speaker-attribution stay unchanged — we only recover text that
    would have been dropped.

    Empty segments (pyannote detected speech but no Whisper words
    landed closest) are still dropped: those are usually inhale/cough
    noises with no transcribable content, and keeping them would emit
    empty `TranscriptSegment`s that break downstream consumers.
    """
    if not raw_segs:
        return []

    buckets: list[list[RawWord]] = [[] for _ in raw_segs]
    for w in words:
        midpoint = 0.5 * (w.start_s + w.end_s)
        best_idx = 0
        best_dist = float("inf")
        for i, seg in enumerate(raw_segs):
            if seg.start_s <= midpoint < seg.end_s:
                dist = 0.0
            elif midpoint < seg.start_s:
                dist = seg.start_s - midpoint
            else:
                dist = midpoint - seg.end_s
            if dist < best_dist:
                best_dist = dist
                best_idx = i
                if dist == 0.0:
                    break
        buckets[best_idx].append(w)

    out: list[TranscriptSegment] = []
    for seg, bucket in zip(raw_segs, buckets, strict=True):
        if not bucket:
            continue
        text = " ".join(w.text for w in bucket if w.text).strip()
        if not text:
            continue
        role = role_map.get(seg.speaker_label, "patient")
        out.append(
            TranscriptSegment(
                start_s=seg.start_s,
                end_s=seg.end_s,
                speaker=role,
                text=text,
                words=[
                    WordTiming(
                        text=w.text,
                        start_s=w.start_s,
                        end_s=w.end_s,
                        probability=w.probability,
                    )
                    for w in bucket
                ],
                diarization_confidence=seg.confidence,
            )
        )
    return out


def _max_end(raw_segs: list[RawDiarizationSegment]) -> float:
    return max((s.end_s for s in raw_segs), default=0.0)
