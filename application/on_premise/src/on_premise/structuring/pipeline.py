"""High-level SOAP-structuring pipeline: `Transcript` -> validated `SOAPNote`.

Composition:
    structuring.slm_engine.StructuredExtractor  +  Transcript  ->  SOAPNote

Phase 2b deliverable — the function `transcript_to_soap()` is the entry
point that Phase 2c (trust gate + Presidio + API) will call.

Pipeline stages:

1. **Render prompt**: `prompts.render_user_prompt(transcript)` serializes
   the diarized transcript as `[N] speaker: text` lines so the SLM can
   cite segment indices in `provenance`.
2. **Extract**: `extractor.extract(prompt, schema=...)` returns a raw
   JSON string. Under `structured_mode="grammar"` it's always
   syntactically valid JSON with the right keys; under `"raw"` anything
   can come back.
3. **Validate with retry**: `SOAPNote.model_validate_json(raw)`. On
   `ValidationError`, append `render_correction_hint(exc)` to the
   prompt and try again (bounded by `max_validation_retries`). Mirrors
   the Phase 1a `ScenarioGenerator._run_one` pattern.
4. **PII hallucination post-check**: walk every string leaf in the
   validated SOAP; drop any PESEL-shaped (`\\d{11}`) or Polish-phone
   shaped (`\\+?48\\d{9}`) substring that isn't present verbatim in the
   transcript text. Cheap guard — the SLM occasionally invents
   plausible Polish numbers, and we'd rather surface that as a dropped
   token + warning than let it ride into Phase 2c.
5. **Build metadata**: SHA-anchor the source transcript (computed here
   — Phase 2a's `Transcript` stays frozen), stamp backend + timing info.
6. **Return** the validated, PII-scrubbed `SOAPNote`.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, ValidationError

from on_premise.structuring.prompts import (
    SYSTEM_PROMPT_PL,
    render_correction_hint,
    render_user_prompt,
)
from on_premise.structuring.soap_schema import (
    SOAPNote,
    SOAPNoteMetadata,
    SOAPNoteSubmission,
)

if TYPE_CHECKING:
    from on_premise.structuring.slm_engine import StructuredExtractor
    from on_premise.stt.schema import Transcript

log = logging.getLogger(__name__)


# PESEL = 11 digits. Polish mobile phone = optional + optional 48 + 9 digits.
# Both patterns intentionally loose — we're pre-filtering the SLM output
# for anything that *looks* like PII but doesn't appear in the source.
# Real anonymization (detecting PII that IS in source) is Phase 2c.
_PESEL_RE = re.compile(r"\b\d{11}\b")
_PHONE_RE = re.compile(r"\+?\d{2}[\s-]?\d{3}[\s-]?\d{3}[\s-]?\d{3}")


class SOAPExtractionError(RuntimeError):
    """Raised when the pipeline can't produce a valid SOAPNote.

    Always wraps the last ValidationError / extractor error as `__cause__`
    so CLI callers can surface the actionable hint.
    """


def transcript_to_soap(
    transcript: Transcript,
    *,
    extractor: StructuredExtractor,
    max_validation_retries: int = 3,
    scenario_id_override: str | None = None,
) -> SOAPNote:
    """Drive the SLM to a validated `SOAPNote` for the given transcript."""
    user_prompt = render_user_prompt(transcript)
    # Phase 2b's system+user concatenation mirrors Phase 1a's rendering —
    # the SLM backends apply the chat template internally so we hand them
    # a single user-role string. Bielik's chat template embeds BOS / EOS
    # tokens; doing it ourselves would double-bracket them.
    full_prompt = f"{SYSTEM_PROMPT_PL}\n\n{user_prompt}"

    # The model emits `SOAPNoteSubmission` (clinical fields only). The
    # pipeline composes the full `SOAPNote` by attaching metadata it built
    # itself — the model can't truthfully produce `source_transcript_sha256`
    # / `generated_at` / backend strings.
    schema = SOAPNoteSubmission.model_json_schema()
    last_error: Exception | None = None
    submission: SOAPNoteSubmission | None = None

    t0 = time.monotonic()
    for attempt in range(max_validation_retries + 1):
        try:
            raw_json = extractor.extract(full_prompt, schema=schema)
        except Exception as exc:
            # Extractor errors propagate — retrying a bad prompt won't fix
            # "model not found" or "grammar compile failed".
            raise SOAPExtractionError(f"extractor failed on attempt {attempt + 1}: {exc}") from exc

        try:
            submission = SOAPNoteSubmission.model_validate_json(raw_json)
            break
        except ValidationError as exc:
            last_error = exc
            log.warning(
                "SOAP validation failed (attempt %d/%d): %s",
                attempt + 1,
                max_validation_retries + 1,
                _summarize(exc),
            )
            if attempt == max_validation_retries:
                break
            full_prompt = full_prompt + "\n\n" + render_correction_hint(exc)

    if submission is None:
        raise SOAPExtractionError(
            f"SOAP extraction failed after {max_validation_retries + 1} attempts"
        ) from last_error

    generation_duration_s = time.monotonic() - t0

    # PII hallucination post-check. The SLM can synthesize plausible
    # Polish numbers (doctors' license numbers, invented phones) that
    # never appeared in the source; pre-2c we strip them rather than
    # letting them cross into the trust gate.
    transcript_text = transcript.text
    submission_scrubbed = _scrub_hallucinated_pii(submission, transcript_text)

    # Compose the full SOAPNote — pipeline owns SHA anchoring + timing so
    # Phase 2a's `Transcript` stays frozen.
    metadata = _build_metadata(
        transcript=transcript,
        extractor=extractor,
        generation_duration_s=generation_duration_s,
        scenario_id_override=scenario_id_override,
    )
    return SOAPNote(**submission_scrubbed.model_dump(), metadata=metadata)


# ---------------------------------------------------------------------------
# Metadata + PII scrub helpers
# ---------------------------------------------------------------------------


def _build_metadata(
    *,
    transcript: Transcript,
    extractor: StructuredExtractor,
    generation_duration_s: float,
    scenario_id_override: str | None,
) -> SOAPNoteMetadata:
    """Anchor the SOAP to transcript + audio bytes and stamp backend info."""
    transcript_bytes = transcript.model_dump_json().encode("utf-8")
    transcript_sha = hashlib.sha256(transcript_bytes).hexdigest()

    # `extractor.name` is "mlx-lm:<repo>" or "llama-cpp:<file>". Split
    # once on ':' to recover the backend key that SOAPNoteMetadata's
    # Literal expects.
    backend_raw, _, model_raw = extractor.name.partition(":")
    if backend_raw not in {"mlx-lm", "llama-cpp"}:
        raise SOAPExtractionError(f"unexpected extractor.name prefix: {extractor.name!r}")
    mode_raw = getattr(extractor, "structured_mode", "grammar")
    if mode_raw not in {"grammar", "raw"}:
        raise SOAPExtractionError(f"unexpected extractor.structured_mode: {mode_raw!r}")

    return SOAPNoteMetadata(
        source_transcript_sha256=transcript_sha,
        source_audio_sha256=transcript.audio_sha256,
        scenario_id=scenario_id_override or transcript.scenario_id,
        slm_backend=cast("Any", backend_raw),
        slm_model=model_raw or extractor.name,
        slm_device=extractor.device,
        slm_structured_mode=cast("Any", mode_raw),
        generated_at=datetime.now().astimezone(),
        generation_duration_s=generation_duration_s,
    )


def _scrub_hallucinated_pii(
    submission: SOAPNoteSubmission, transcript_text: str
) -> SOAPNoteSubmission:
    """Drop PII-shaped substrings that are NOT in `transcript_text`.

    Walks every string leaf in the submission tree. A PESEL-shaped /
    phone-shaped substring that appears in the SLM output but not in
    the source transcript is replaced with an empty string in scalars
    and filtered out of lists of strings. Logs a warning per drop so
    the CLI / eval can surface rate.
    """
    dropped: list[str] = []

    def _clean_str(value: str, path: str) -> str:
        if not value:
            return value
        original = value
        for regex in (_PESEL_RE, _PHONE_RE):
            for match in regex.finditer(original):
                token = match.group(0)
                if token and token not in transcript_text:
                    value = value.replace(token, "")
                    dropped.append(f"{path}:{token!r}")
        return value

    def _walk(node: Any, path: str) -> Any:
        if isinstance(node, str):
            return _clean_str(node, path)
        if isinstance(node, list):
            out: list[Any] = []
            for i, item in enumerate(node):
                cleaned = _walk(item, f"{path}[{i}]")
                # Drop list entries that became empty strings after scrubbing —
                # "" in `medications_prescribed` is worse than silence.
                if isinstance(cleaned, str) and not cleaned.strip():
                    continue
                out.append(cleaned)
            return out
        if isinstance(node, dict):
            return {k: _walk(v, f"{path}.{k}") for k, v in node.items()}
        if isinstance(node, BaseModel):
            as_dict = node.model_dump()
            cleaned = _walk(as_dict, path)
            return type(node).model_validate(cleaned)
        return node

    cleaned_submission = _walk(submission, "soap")
    if dropped:
        log.warning(
            "dropped %d hallucinated PII-shaped token(s): %s",
            len(dropped),
            ", ".join(dropped[:10]),
        )
    # `_walk` returns a fresh SOAPNoteSubmission when the root is a BaseModel.
    assert isinstance(cleaned_submission, SOAPNoteSubmission)
    return cleaned_submission


def _summarize(exc: ValidationError) -> str:
    parts: list[str] = []
    for err in exc.errors()[:3]:
        loc = ".".join(str(x) for x in err.get("loc", ()))
        parts.append(f"{loc}: {err.get('msg', '')}")
    return "; ".join(parts)
