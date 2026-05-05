"""SLM-based post-classifier for speaker labels in a `Transcript`.

Phase 2a baseline DER (mean clean 32.96% / noisy 43.36%) is dominated
by `pyannote.audio` clustering collapse on synthetic XTTSv2 voices —
the wespeaker embedding doesn't separate cloned voices reliably, so
the same physical speaker gets relabeled SPEAKER_00↔SPEAKER_01 mid-
conversation. This is an artifact of the synthetic dataset; on real
clinical recordings pyannote performs much better.

We can recover most of this loss by **content-based reclassification**:
doctors and patients use distinctive language patterns. The SLM
(Bielik-11B-v3.0 from Phase 2b) reads the segmented transcript and
emits a (segment_idx, role, confidence) triple per turn. Hybrid
override policy: SLM wins when its confidence ≥ threshold; pyannote
wins otherwise.

Why post-classifier (not replacement):
- pyannote handles VAD + segment boundaries (timing) — that's audio-
  native and accurate.
- pyannote handles cluster count detection (we hard-code 2 here, but
  the architecture is sound).
- Only the role-assignment step suffers — and that's exactly what the
  SLM can do better, because role is a CONTENT property.

Module trust posture: lives under `on_premise.stt`, never imported
from `cloud/`. Reuses `StructuredExtractor` from Phase 2b.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from on_premise.stt.schema import Transcript, TranscriptSegment

log = logging.getLogger(__name__)


# Override threshold — SLM disagreement beats pyannote when confidence ≥ this.
# Tuned conservatively: 0.7 means "SLM is fairly sure"; below that we trust
# the audio-native label.
DEFAULT_OVERRIDE_THRESHOLD = 0.7

# Chunk size for SLM batching. Bielik-11B-v3.0-4bit on M5 24GB has ~1.0 RTF
# on long structured generations: a 60-segment transcript serialised as one
# call produces ~5K JSON tokens = 60-120 s. Chunking into 12-segment batches
# caps each call at ~1K tokens (~5-10 s) and avoids token-budget truncation
# entirely. Total wall time scales linearly with segment count.
DEFAULT_CHUNK_SIZE = 12


# ---------------------------------------------------------------------------
# Pydantic schemas — what the SLM emits.
# ---------------------------------------------------------------------------


class TurnClassification(BaseModel):
    """One per-segment classification from the SLM."""

    model_config = ConfigDict(extra="forbid")

    segment_idx: int = Field(ge=0)
    role: Literal["doctor", "patient"]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""


class TranscriptClassification(BaseModel):
    """SLM response shape — one classification per segment."""

    model_config = ConfigDict(extra="forbid")

    classifications: list[TurnClassification] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Protocol + production impl + fake.
# ---------------------------------------------------------------------------


class SpeakerClassifier(Protocol):
    """Minimum surface — `reclassify` consumes a Transcript, returns a new one."""

    def reclassify(self, transcript: Transcript) -> Transcript: ...


SYSTEM_PROMPT_PL = """\
Jesteś asystentem klasyfikującym wypowiedzi w transkrypcie polskiej wizyty \
lekarskiej. Każda wypowiedź należy do jednej z dwóch ról: LEKARZ lub PACJENT.

SYGNAŁY ROZRÓŻNIAJĄCE:
- LEKARZ: zadaje pytania otwierające ("Co Pana/Panią sprowadza?", "Od kiedy?", \
  "Czy boli?"), używa terminologii medycznej (terapia, rozpoznanie, badanie, \
  jednostki SI, ICD), formułuje diagnozy, wystawia zalecenia, przepisuje leki, \
  proponuje skierowania, podsumowuje wizytę.
- PACJENT: opisuje swoje objawy w pierwszej osobie ("boli mnie", "od dwóch \
  tygodni mam", "schudłem", "czuję się"), zgłasza dolegliwości, odpowiada na \
  pytania, prosi o wyjaśnienie, używa potocznego języka.

ZASADY:
1. Każdej wypowiedzi przyporządkuj `role` ("doctor" albo "patient") + \
   `confidence` (0.0-1.0) + krótkie `rationale` po polsku.
2. Confidence 0.9+ tylko gdy treść jest jednoznaczna (np. "wystawię receptę" \
   = lekarz, "boli mnie głowa od dwóch dni" = pacjent).
3. Confidence 0.5-0.8 dla treści dwuznacznych (np. "tak", "nie wiem", "rozumiem").
4. Confidence < 0.5 gdy nie da się sklasyfikować z samego tekstu.
5. Zwróć `classifications` z wpisem dla KAŻDEGO segmentu wejściowego, w tej \
   samej kolejności (segment_idx zgodne z wejściem).
6. NIE generuj nowych segmentów ani nie pomijaj — same liczność co na wejściu."""


def render_user_prompt(transcript: Transcript) -> str:
    """Serialize the transcript as numbered turns for the SLM."""
    lines = ["TRANSKRYPT:"]
    for i, seg in enumerate(transcript.segments):
        # Strip whitespace + collapse newlines so each turn is one line.
        text = " ".join(seg.text.split())
        lines.append(f"[{i}] {seg.speaker}: {text}")
    lines.append("")
    lines.append(
        "ZADANIE: zwróć `classifications` — listę obiektów "
        "{segment_idx, role, confidence, rationale} po jednym na każdy "
        "segment, w tej samej kolejności co wejście."
    )
    return "\n".join(lines)


@dataclass
class _BielikReclassifier:
    """Production reclassifier — uses Phase 2b's `StructuredExtractor`.

    The extractor is duck-typed (any object with `.extract(prompt, schema=...)`
    method works). Default factory builds it from `slm.yaml` + env, mirroring
    the Phase 2b CLI bootstrap path.

    **Chunking**: long transcripts are split into `chunk_size`-segment
    batches, one SLM call per batch. Bielik-11B-v3.0-4bit on M5 24GB
    bottlenecks on token-by-token generation, so 5× short calls beat
    1× long call by a 5-10× wall-time factor.
    """

    extractor: object  # StructuredExtractor — avoid circular import
    override_threshold: float = DEFAULT_OVERRIDE_THRESHOLD
    chunk_size: int = DEFAULT_CHUNK_SIZE

    def reclassify(self, transcript: Transcript) -> Transcript:
        if not transcript.segments:
            return transcript

        all_classifications: list[TurnClassification] = []
        n_segments = len(transcript.segments)
        chunk_size = max(1, self.chunk_size)

        for chunk_start in range(0, n_segments, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_segments)
            chunk_segs = transcript.segments[chunk_start:chunk_end]

            prompt = (
                SYSTEM_PROMPT_PL
                + "\n\n"
                + _render_chunk_prompt(chunk_segs, offset=chunk_start)
            )
            schema = TranscriptClassification.model_json_schema()
            try:
                raw = self.extractor.extract(prompt, schema=schema)  # type: ignore[attr-defined]
            except Exception as exc:
                log.warning(
                    "speaker reclassifier chunk [%d:%d] failed: %s — keeping pyannote labels",
                    chunk_start, chunk_end, exc,
                )
                continue

            try:
                chunk_class = TranscriptClassification.model_validate_json(raw)
            except Exception as exc:
                log.warning(
                    "speaker reclassifier chunk [%d:%d] parse failed: %s — keeping pyannote labels",
                    chunk_start, chunk_end, exc,
                )
                continue

            all_classifications.extend(chunk_class.classifications)

        merged = TranscriptClassification(classifications=all_classifications)
        return _apply_classification(
            transcript, merged, override_threshold=self.override_threshold
        )


def _render_chunk_prompt(
    segments: list[TranscriptSegment], *, offset: int = 0
) -> str:
    """Render a CHUNK of segments. `offset` is the global index of the
    first segment in the chunk — `segment_idx` returned by the SLM is
    expected to match the GLOBAL index, so the agent always sees the
    real position even when chunked.
    """
    lines = ["TRANSKRYPT (fragment):"]
    for i, seg in enumerate(segments):
        text = " ".join(seg.text.split())
        lines.append(f"[{offset + i}] {seg.speaker}: {text}")
    lines.append("")
    lines.append(
        "ZADANIE: zwróć `classifications` — listę obiektów "
        "{segment_idx, role, confidence, rationale} po jednym na każdy "
        "segment z fragmentu, z `segment_idx` zgodnym z numerami w nawiasach "
        "kwadratowych (np. [42] → segment_idx=42)."
    )
    return "\n".join(lines)


def _apply_classification(
    transcript: Transcript,
    classification: TranscriptClassification,
    *,
    override_threshold: float,
) -> Transcript:
    """Merge SLM classifications into the transcript by index."""
    by_idx: dict[int, TurnClassification] = {
        c.segment_idx: c for c in classification.classifications
    }
    new_segments: list[TranscriptSegment] = []
    overrides = 0
    kept = 0
    skipped_low_conf = 0
    for i, seg in enumerate(transcript.segments):
        c = by_idx.get(i)
        if c is None:
            new_segments.append(seg)
            continue
        if c.confidence < override_threshold:
            skipped_low_conf += 1
            new_segments.append(seg)
            continue
        if c.role == seg.speaker:
            kept += 1
            new_segments.append(seg)
            continue
        # SLM is confident AND disagrees → override.
        overrides += 1
        new_segments.append(seg.model_copy(update={"speaker": c.role}))

    log.info(
        "speaker reclassifier: %d/%d overridden (kept %d agreeing, "
        "skipped %d low-confidence)",
        overrides, len(transcript.segments), kept, skipped_low_conf,
    )
    return transcript.model_copy(update={"segments": new_segments})


# ---------------------------------------------------------------------------
# Test fixture — never calls a real SLM.
# ---------------------------------------------------------------------------


@dataclass
class _FakeSpeakerClassifier:
    """Returns a canned `TranscriptClassification` regardless of input."""

    canned: TranscriptClassification
    override_threshold: float = DEFAULT_OVERRIDE_THRESHOLD
    calls: int = 0

    def reclassify(self, transcript: Transcript) -> Transcript:
        self.calls += 1
        return _apply_classification(
            transcript, self.canned, override_threshold=self.override_threshold
        )


@dataclass
class _NullClassifier:
    """No-op — always returns the transcript unchanged. Used when SLM is
    unavailable or feature is disabled.
    """

    def reclassify(self, transcript: Transcript) -> Transcript:
        return transcript


# ---------------------------------------------------------------------------
# Factory.
# ---------------------------------------------------------------------------


def build_classifier(*, extractor: object | None = None) -> SpeakerClassifier:
    """Returns `_BielikReclassifier(extractor)` when extractor is provided,
    else `_NullClassifier`. Tests inject `_FakeSpeakerClassifier` directly.
    """
    if extractor is None:
        return _NullClassifier()
    return _BielikReclassifier(extractor=extractor)


__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_OVERRIDE_THRESHOLD",
    "SYSTEM_PROMPT_PL",
    "SpeakerClassifier",
    "TranscriptClassification",
    "TurnClassification",
    "_BielikReclassifier",
    "_FakeSpeakerClassifier",
    "_NullClassifier",
    "_render_chunk_prompt",
    "build_classifier",
    "render_user_prompt",
]
