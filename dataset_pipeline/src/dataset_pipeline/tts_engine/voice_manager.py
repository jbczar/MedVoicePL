"""Voice profile and reference sample management.

Reads `configs/tts_profiles.yaml` and maps each `(speaker, patient_sex)` pair
to one of the four profiles (`doctor_male`, `doctor_female`, `patient_male`,
`patient_female`). For the doctor side the choice between male/female is
derived deterministically from `scenario.id` so a given scenario always gets
the same doctor voice while different scenarios vary.
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


def _doctor_sex_for_scenario(scenario_id: str) -> PatientSex:
    """Pick a deterministic doctor sex for a scenario.

    Hashes the scenario id so a given scenario is stable across reruns but
    different scenarios get different doctor voices.
    """
    digest = hashlib.sha256(scenario_id.encode("utf-8")).digest()
    return "M" if digest[0] % 2 == 0 else "F"


def profile_for(
    catalog: VoiceCatalog,
    *,
    speaker: SpeakerRole,
    patient_sex: PatientSex,
    scenario_id: str,
) -> VoiceProfile:
    """Resolve the `VoiceProfile` for a given turn."""
    from dataset_pipeline.scenario_generator.schema import SpeakerRole

    if speaker is SpeakerRole.patient:
        key = "patient_male" if patient_sex == "M" else "patient_female"
    elif speaker is SpeakerRole.doctor:
        doctor_sex = _doctor_sex_for_scenario(scenario_id)
        key = "doctor_male" if doctor_sex == "M" else "doctor_female"
    else:  # pragma: no cover — StrEnum is exhaustive
        raise ValueError(f"unknown speaker role: {speaker!r}")

    try:
        return catalog.profiles[key]
    except KeyError as exc:
        raise KeyError(
            f"voice profile {key!r} missing from tts_profiles.yaml "
            f"(available: {sorted(catalog.profiles)})"
        ) from exc
