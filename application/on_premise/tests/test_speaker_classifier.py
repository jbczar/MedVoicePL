"""Tests for `cloud-side`/`on-prem-side` speaker reclassifier (Phase 2a.1)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from on_premise.stt.schema import (
    Transcript,
    TranscriptionMetadata,
    TranscriptSegment,
)
from on_premise.stt.speaker_classifier import (
    DEFAULT_OVERRIDE_THRESHOLD,
    SYSTEM_PROMPT_PL,
    TranscriptClassification,
    TurnClassification,
    _apply_classification,
    _BielikReclassifier,
    _FakeSpeakerClassifier,
    _NullClassifier,
    build_classifier,
    render_user_prompt,
)


def _seg(start: float, end: float, speaker: str, text: str) -> TranscriptSegment:
    return TranscriptSegment(
        start_s=start, end_s=end, speaker=speaker, text=text  # type: ignore[arg-type]
    )


def _transcript(*segments: TranscriptSegment) -> Transcript:
    return Transcript(
        scenario_id="test",
        audio_path="/tmp/x.wav",
        audio_sha256="0" * 64,
        duration_s=max(s.end_s for s in segments) if segments else 0.0,
        language="pl",
        segments=list(segments),
        metadata=TranscriptionMetadata(
            stt_backend="mlx",
            stt_model="test",
            stt_device="cpu",
            stt_compute_type=None,
            diarization_model="test",
            diarization_device="cpu",
            transcribed_at=datetime.now(UTC),
        ),
    )


# ---------------------------------------------------------------------------
# Fake & Null
# ---------------------------------------------------------------------------


class TestNullClassifier:
    def test_returns_input_unchanged(self) -> None:
        t = _transcript(_seg(0.0, 1.0, "doctor", "co Pana sprowadza"))
        out = _NullClassifier().reclassify(t)
        assert out.segments == t.segments


class TestFakeClassifier:
    def test_overrides_when_high_confidence(self) -> None:
        t = _transcript(
            _seg(0.0, 1.0, "doctor", "boli mnie głowa"),
            _seg(1.0, 2.0, "patient", "rozumiem co Pana"),
        )
        canned = TranscriptClassification(
            classifications=[
                TurnClassification(segment_idx=0, role="patient", confidence=0.95),
                TurnClassification(segment_idx=1, role="doctor", confidence=0.9),
            ]
        )
        f = _FakeSpeakerClassifier(canned=canned)
        out = f.reclassify(t)
        assert [s.speaker for s in out.segments] == ["patient", "doctor"]

    def test_keeps_pyannote_when_low_confidence(self) -> None:
        t = _transcript(_seg(0.0, 1.0, "doctor", "tak"))
        canned = TranscriptClassification(
            classifications=[
                TurnClassification(segment_idx=0, role="patient", confidence=0.4)
            ]
        )
        out = _FakeSpeakerClassifier(canned=canned).reclassify(t)
        assert out.segments[0].speaker == "doctor"  # pyannote label preserved

    def test_keeps_pyannote_when_agreeing(self) -> None:
        t = _transcript(_seg(0.0, 1.0, "doctor", "co Pana boli"))
        canned = TranscriptClassification(
            classifications=[
                TurnClassification(segment_idx=0, role="doctor", confidence=0.95)
            ]
        )
        out = _FakeSpeakerClassifier(canned=canned).reclassify(t)
        assert out.segments[0].speaker == "doctor"


class TestApplyClassification:
    def test_threshold_boundary_inclusive(self) -> None:
        t = _transcript(_seg(0.0, 1.0, "doctor", "x"))
        c = TranscriptClassification(
            classifications=[
                TurnClassification(
                    segment_idx=0, role="patient",
                    confidence=DEFAULT_OVERRIDE_THRESHOLD,
                )
            ]
        )
        out = _apply_classification(t, c, override_threshold=DEFAULT_OVERRIDE_THRESHOLD)
        assert out.segments[0].speaker == "patient"

    def test_missing_classification_keeps_segment(self) -> None:
        t = _transcript(
            _seg(0.0, 1.0, "doctor", "a"),
            _seg(1.0, 2.0, "patient", "b"),
        )
        # Only classify segment 0; segment 1 should be preserved.
        c = TranscriptClassification(
            classifications=[
                TurnClassification(segment_idx=0, role="patient", confidence=0.95)
            ]
        )
        out = _apply_classification(t, c, override_threshold=0.7)
        assert out.segments[0].speaker == "patient"
        assert out.segments[1].speaker == "patient"  # original

    def test_empty_segments_passthrough(self) -> None:
        t = _transcript()
        c = TranscriptClassification()
        out = _apply_classification(t, c, override_threshold=0.7)
        assert out.segments == []


class TestRenderPrompt:
    def test_includes_segment_indices(self) -> None:
        t = _transcript(
            _seg(0.0, 1.0, "doctor", "Co Pana sprowadza?"),
            _seg(1.0, 2.0, "patient", "Boli mnie głowa od dwóch dni."),
        )
        prompt = render_user_prompt(t)
        assert "[0]" in prompt
        assert "[1]" in prompt
        assert "Co Pana sprowadza?" in prompt
        assert "Boli mnie głowa" in prompt

    def test_collapses_whitespace(self) -> None:
        t = _transcript(_seg(0.0, 1.0, "doctor", "Co\n\n  Pana\tsprowadza?"))
        prompt = render_user_prompt(t)
        assert "Co Pana sprowadza?" in prompt


class TestSystemPrompt:
    def test_contains_role_signals(self) -> None:
        text = SYSTEM_PROMPT_PL.lower()
        assert "lekarz" in text
        assert "pacjent" in text
        assert "confidence" in text
        assert "rationale" in text

    def test_explicit_format_anchor(self) -> None:
        # Should mention required JSON fields.
        assert "segment_idx" in SYSTEM_PROMPT_PL
        assert "role" in SYSTEM_PROMPT_PL


class TestBielikReclassifier:
    def test_extractor_failure_falls_back_to_pyannote(self) -> None:
        class _BoomExtractor:
            def extract(self, prompt: str, *, schema):
                raise RuntimeError("boom")

        t = _transcript(_seg(0.0, 1.0, "doctor", "x"))
        r = _BielikReclassifier(extractor=_BoomExtractor())
        out = r.reclassify(t)
        assert out.segments[0].speaker == "doctor"  # unchanged

    def test_parse_failure_falls_back_to_pyannote(self) -> None:
        class _GarbageExtractor:
            def extract(self, prompt: str, *, schema):
                return "definitely not json"

        t = _transcript(_seg(0.0, 1.0, "doctor", "x"))
        r = _BielikReclassifier(extractor=_GarbageExtractor())
        out = r.reclassify(t)
        assert out.segments[0].speaker == "doctor"

    def test_valid_json_response_applied(self) -> None:
        canned_json = TranscriptClassification(
            classifications=[
                TurnClassification(segment_idx=0, role="patient", confidence=0.9)
            ]
        ).model_dump_json()

        class _GoodExtractor:
            def extract(self, prompt: str, *, schema):
                return canned_json

        t = _transcript(_seg(0.0, 1.0, "doctor", "x"))
        r = _BielikReclassifier(extractor=_GoodExtractor())
        out = r.reclassify(t)
        assert out.segments[0].speaker == "patient"

    def test_chunking_calls_extractor_multiple_times(self) -> None:
        """30 segments / chunk_size 12 → 3 calls (12+12+6)."""

        class _ChunkedExtractor:
            def __init__(self) -> None:
                self.calls = 0

            def extract(self, prompt: str, *, schema):
                self.calls += 1
                # Echo back same shape with empty classifications — chunking
                # behavior is what we're testing, not classification quality.
                return TranscriptClassification(classifications=[]).model_dump_json()

        ext = _ChunkedExtractor()
        segments = [_seg(i * 1.0, (i + 1) * 1.0, "doctor", f"seg {i}") for i in range(30)]
        t = _transcript(*segments)
        r = _BielikReclassifier(extractor=ext, chunk_size=12)
        r.reclassify(t)
        assert ext.calls == 3  # 12 + 12 + 6

    def test_chunk_indices_are_global(self) -> None:
        """When chunk 2 (segments 12-23) is classified, returned segment_idx
        should match GLOBAL position, so override targets the right segment.
        """
        # Each chunk gets its own canned response — segment_idx mirrors
        # GLOBAL index so the merge applies to the right segment.
        sent_prompts: list[str] = []

        class _Echo:
            def extract(self, prompt: str, *, schema):
                sent_prompts.append(prompt)
                # Pretend SLM correctly returns global indices.
                # We over-classify with high-confidence override on segment 13.
                return TranscriptClassification(
                    classifications=[
                        TurnClassification(
                            segment_idx=13, role="patient", confidence=0.95
                        )
                    ]
                ).model_dump_json()

        # 24 segments, all initially "doctor".
        segs = [_seg(i * 1.0, (i + 1) * 1.0, "doctor", f"x{i}") for i in range(24)]
        t = _transcript(*segs)
        r = _BielikReclassifier(extractor=_Echo(), chunk_size=12)
        out = r.reclassify(t)

        # Segment 13 should be flipped exactly once, and chunk-2 prompt
        # should have included "[12]"…"[23]" (global indices in brackets).
        assert out.segments[13].speaker == "patient"
        assert all(out.segments[i].speaker == "doctor" for i in range(24) if i != 13)
        assert any("[12]" in p and "[23]" in p for p in sent_prompts)

    def test_chunk_failure_does_not_abort_other_chunks(self) -> None:
        """If one chunk's SLM call raises, other chunks still apply."""

        class _FlakyExtractor:
            def __init__(self) -> None:
                self.calls = 0

            def extract(self, prompt: str, *, schema):
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError("transient failure")
                return TranscriptClassification(
                    classifications=[
                        TurnClassification(
                            segment_idx=0, role="patient", confidence=0.95
                        )
                    ]
                ).model_dump_json()

        # 36 segments → 3 chunks of 12. Chunk 2 fails, chunks 1+3 succeed.
        segs = [_seg(i * 1.0, (i + 1) * 1.0, "doctor", f"x{i}") for i in range(36)]
        t = _transcript(*segs)
        ext = _FlakyExtractor()
        r = _BielikReclassifier(extractor=ext, chunk_size=12)
        out = r.reclassify(t)
        assert ext.calls == 3
        # Segment 0 was overridden (chunk 1 succeeded).
        assert out.segments[0].speaker == "patient"

    def test_empty_transcript_skips_call(self) -> None:
        class _CountingExtractor:
            calls = 0

            def extract(self, prompt: str, *, schema):
                self.calls += 1
                return "{}"

        ext = _CountingExtractor()
        r = _BielikReclassifier(extractor=ext)
        t = _transcript()
        r.reclassify(t)
        assert ext.calls == 0


class TestBuildClassifier:
    def test_null_when_no_extractor(self) -> None:
        c = build_classifier(extractor=None)
        assert isinstance(c, _NullClassifier)

    def test_bielik_when_extractor_given(self) -> None:
        class _Stub:
            def extract(self, prompt, *, schema):
                return "{}"

        c = build_classifier(extractor=_Stub())
        assert isinstance(c, _BielikReclassifier)


class TestSchemas:
    def test_turn_classification_validates(self) -> None:
        t = TurnClassification(segment_idx=0, role="doctor", confidence=0.85)
        assert t.role == "doctor"
        assert t.confidence == 0.85

    def test_turn_classification_rejects_bad_role(self) -> None:
        with pytest.raises(Exception):
            TurnClassification(
                segment_idx=0, role="nurse", confidence=0.9  # type: ignore[arg-type]
            )

    def test_turn_classification_rejects_out_of_range_confidence(self) -> None:
        with pytest.raises(Exception):
            TurnClassification(segment_idx=0, role="doctor", confidence=1.5)
