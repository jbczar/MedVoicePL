"""Voice profile and reference sample management.

Reads `configs/tts_profiles.yaml` and maps each `(speaker, patient_sex, age)`
tuple to one of the 12 voice profiles introduced in Phase 4:

* Doctor profiles (4): `doctor_{male,female}_{young,senior}` — age bucket
  derived deterministically from `scenario.id` for stability.
* Patient profiles (8): `patient_{male,female}_{young,middle,senior}` and
  `patient_child_{male,female}`. Bucket derived from `scenario.patient_age`.

Pre-Phase 4 the catalog had only 4 generic profiles. Tests and legacy
configs that still use those keys (`doctor_male`, `patient_female`, etc.)
are supported via a fallback in `profile_for`.

Pediatric note: for `patient_age < 6` the "patient" speaker is often a
parent rather than the child themselves. Voice mapping currently uses
the child-voice profile for ages 0-17 regardless; the scenario generator
emits dialogue text that matches (parent speech for very young patients).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import yaml

if TYPE_CHECKING:
    from pathlib import Path

    from dataset_pipeline.scenario_generator.schema import SpeakerRole


PatientSex = Literal["M", "F"]
AgeBucket = Literal["child", "young", "middle", "senior"]


@dataclass(frozen=True)
class VoiceProfile:
    name: str
    reference_audio: Path
    language: str
    speed: float


@dataclass(frozen=True)
class VoiceCatalog:
    """All voice profiles + synthesis-level config loaded from YAML."""

    profiles: dict[str, VoiceProfile]
    sample_rate: int
    output_format: str


def load_profiles(config_path: Path, *, repo_root: Path | None = None) -> VoiceCatalog:
    """Parse `tts_profiles.yaml` and return a `VoiceCatalog`.

    `reference_audio` paths in the YAML are resolved relative to `repo_root`
    (defaults to the parent of `config_path`'s parent — i.e., the workspace
    member root `dataset_pipeline/`). This keeps configs portable while letting
    tests point at a fixture tree.
    """
    if repo_root is None:
        repo_root = config_path.parent.parent  # configs/ -> dataset_pipeline/

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    profiles_raw = raw.get("profiles") or {}
    if not profiles_raw:
        raise ValueError(f"no profiles defined in {config_path}")

    profiles: dict[str, VoiceProfile] = {}
    for name, spec in profiles_raw.items():
        ref = repo_root / str(spec["reference_audio"])
        profiles[name] = VoiceProfile(
            name=name,
            reference_audio=ref,
            language=str(spec.get("language", "pl")),
            speed=float(spec.get("speed", 1.0)),
        )

    synthesis = raw.get("synthesis") or {}
    return VoiceCatalog(
        profiles=profiles,
        sample_rate=int(synthesis.get("sample_rate", 22050)),
        output_format=str(synthesis.get("output_format", "wav")),
    )


def patient_age_bucket(age: int) -> AgeBucket:
    """Map a numeric age to a discrete bucket for voice selection.

    Buckets are coarse on purpose — Mozilla Common Voice provides
    coarse-grained `age` labels (teens / twenties / thirties / ...),
    and XTTSv2 voice cloning blends timbre across a wide range, so
    finer-grained buckets buy nothing.
    """
    if age < 0:
        raise ValueError(f"patient_age must be non-negative, got {age}")
    if age < 18:
        return "child"
    if age < 40:
        return "young"
    if age < 60:
        return "middle"
    return "senior"


def _doctor_sex_for_scenario(scenario_id: str) -> PatientSex:
    """Pick a deterministic doctor sex for a scenario."""
    digest = hashlib.sha256(scenario_id.encode("utf-8")).digest()
    return "M" if digest[0] % 2 == 0 else "F"


def _doctor_age_bucket_for_scenario(scenario_id: str) -> AgeBucket:
    """Pick a deterministic doctor age bucket (young or senior, no middle).

    Doctors get only two age buckets (vs three for patients) so the audible
    voice difference is bigger — a senior doctor sounds distinctly older
    than a young doctor. With 'middle' the bucket boundary blurs and the
    generated audio doesn't add diversity.
    """
    digest = hashlib.sha256(scenario_id.encode("utf-8")).digest()
    return "young" if digest[1] % 2 == 0 else "senior"


def _resolve_with_fallback(
    catalog: VoiceCatalog, primary: str, fallback: str
) -> VoiceProfile:
    """Try the Phase 4 key first, fall back to the pre-Phase 4 short name.

    Lets old YAML configs (`doctor_male` etc.) keep working alongside the
    expanded 12-profile catalog.
    """
    if primary in catalog.profiles:
        return catalog.profiles[primary]
    if fallback in catalog.profiles:
        return catalog.profiles[fallback]
    raise KeyError(
        f"voice profile {primary!r} (or fallback {fallback!r}) missing from "
        f"tts_profiles.yaml (available: {sorted(catalog.profiles)})"
    )


def profile_for(
    catalog: VoiceCatalog,
    *,
    speaker: SpeakerRole,
    patient_sex: PatientSex,
    scenario_id: str,
    patient_age: int | None = None,
) -> VoiceProfile:
    """Resolve the `VoiceProfile` for a given turn.

    `patient_age` is optional for backward compatibility. When omitted the
    selector defaults to bucket="middle" for adults (legacy behavior).
    """
    from dataset_pipeline.scenario_generator.schema import SpeakerRole

    if speaker is SpeakerRole.patient:
        sex_key = "male" if patient_sex == "M" else "female"
        bucket: AgeBucket = (
            patient_age_bucket(patient_age) if patient_age is not None else "middle"
        )
        primary = (
            f"patient_child_{sex_key}"
            if bucket == "child"
            else f"patient_{sex_key}_{bucket}"
        )
        # Fallback to the pre-Phase 4 generic key (`patient_male` / `patient_female`)
        fallback = f"patient_{sex_key}"
        return _resolve_with_fallback(catalog, primary, fallback)

    if speaker is SpeakerRole.doctor:
        doctor_sex = _doctor_sex_for_scenario(scenario_id)
        doctor_bucket = _doctor_age_bucket_for_scenario(scenario_id)
        sex_key = "male" if doctor_sex == "M" else "female"
        primary = f"doctor_{sex_key}_{doctor_bucket}"
        fallback = f"doctor_{sex_key}"
        return _resolve_with_fallback(catalog, primary, fallback)

    raise ValueError(f"unknown speaker role: {speaker!r}")  # pragma: no cover
