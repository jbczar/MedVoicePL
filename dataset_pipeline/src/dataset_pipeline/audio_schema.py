"""Pydantic models for Phase 1b audio manifest entries.

Kept separate from `ScenarioModel` (in `scenario_generator/schema.py`) so the
text dialogue contract stays frozen. Audio is metadata *about* scenarios; it
belongs in the manifest, not in the scenario JSON itself.

Contract consumed by:
  - `dataset_pipeline/scripts/export_manifest.py` (writer)
  - `application/on_premise/stt` evaluator (reader, future)
  - `research_and_testing` benchmarks (reader, future)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AudioArtifact(BaseModel):
    """One per-turn WAV file on disk."""

    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    turn_index: int = Field(ge=0)
    speaker: Literal["doctor", "patient"]
    path: str  # relative to output/ root, forward slashes
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    duration_s: float = Field(ge=0.0)
    sample_rate: int = Field(ge=1)
    voice_profile: str  # "doctor_male" | "doctor_female" | "patient_male" | "patient_female"


class ScenarioAudioBundle(BaseModel):
    """All audio artifacts for a single scenario, including the mixed track."""

    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    dialogue_path: str
    dialogue_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    dialogue_duration_s: float = Field(ge=0.0)
    dialogue_sample_rate: int = Field(ge=1)
    turns: list[AudioArtifact] = Field(min_length=1)
    synthesized_at: datetime
    tts_model: str
    tts_device: str

    # ------------------------------------------------------------------
    # Phase 1b.1 — noisy variant (optional). Null on old Phase 1b bundles
    # and on runs with `--clean-only`.
    # ------------------------------------------------------------------
    dialogue_noisy_path: str | None = None
    dialogue_noisy_sha256: str | None = Field(
        default=None, pattern=r"^[a-f0-9]{64}$"
    )
    dialogue_noisy_duration_s: float | None = Field(default=None, ge=0.0)
    noise_snr_db: float | None = None
    noise_clip: str | None = None
    reverb_room: str | None = None


def bundle_from_dir(
    bundle_dir: Path,
    *,
    scenario_id: str,
    tts_model: str,
    tts_device: str,
) -> ScenarioAudioBundle:
    """Reconstruct a `ScenarioAudioBundle` from on-disk WAVs + sidecar JSON.

    Not used by `run_audio` directly (it builds the bundle in-memory), but used
    by `export_manifest.py` to re-validate entries that were written earlier.
    """
    # Deferred import keeps schema module import-cheap for callers that only
    # need the Pydantic classes.
    import hashlib
    import json

    from dataset_pipeline.tts_engine.audio_utils import duration_seconds, load_wav

    sidecar = bundle_dir / "bundle.json"
    if not sidecar.exists():
        raise FileNotFoundError(f"missing bundle sidecar: {sidecar}")
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    # Refresh SHAs/durations from disk so on-disk edits are caught.
    turns: list[AudioArtifact] = []
    for entry in data["turns"]:
        wav = bundle_dir / Path(entry["path"]).name
        audio, sr = load_wav(wav)
        turns.append(
            AudioArtifact(
                scenario_id=scenario_id,
                turn_index=entry["turn_index"],
                speaker=entry["speaker"],
                path=entry["path"],
                sha256=hashlib.sha256(wav.read_bytes()).hexdigest(),
                duration_s=duration_seconds(audio, sr),
                sample_rate=sr,
                voice_profile=entry["voice_profile"],
            )
        )
    dialogue = bundle_dir / "dialogue.wav"
    d_audio, d_sr = load_wav(dialogue)
    return ScenarioAudioBundle(
        scenario_id=scenario_id,
        dialogue_path=data["dialogue_path"],
        dialogue_sha256=hashlib.sha256(dialogue.read_bytes()).hexdigest(),
        dialogue_duration_s=duration_seconds(d_audio, d_sr),
        dialogue_sample_rate=d_sr,
        turns=turns,
        synthesized_at=datetime.fromisoformat(data["synthesized_at"]),
        tts_model=tts_model,
        tts_device=tts_device,
    )
