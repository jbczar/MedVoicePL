"""Unit tests for tts_engine.voice_manager.

Covers:
  - YAML loading: 4 profiles surface with the right language/speed/reference.
  - Reference paths resolve relative to the workspace member root.
  - `profile_for` picks patient voice by patient_sex.
  - Doctor voice is stable per scenario_id but varies across scenarios.
  - Missing profile key raises KeyError with useful message.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dataset_pipeline.scenario_generator.schema import SpeakerRole
from dataset_pipeline.tts_engine.voice_manager import (
    VoiceCatalog,
    load_profiles,
    profile_for,
)

_YAML_TEMPLATE = """\
profiles:
  doctor_male:
    reference_audio: "reference_voices/doctor_male_01.wav"
    language: "pl"
    speed: 1.0
  doctor_female:
    reference_audio: "reference_voices/doctor_female_01.wav"
    language: "pl"
    speed: 1.0
  patient_male:
    reference_audio: "reference_voices/patient_male_01.wav"
    language: "pl"
    speed: 0.95
  patient_female:
    reference_audio: "reference_voices/patient_female_01.wav"
    language: "pl"
    speed: 0.95

synthesis:
  sample_rate: 22050
  output_format: "wav"
"""


def _write_catalog_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Create a minimal `<member>/configs/tts_profiles.yaml` layout."""
    member_root = tmp_path / "dataset_pipeline"
    configs_dir = member_root / "configs"
    ref_dir = member_root / "reference_voices"
    configs_dir.mkdir(parents=True)
    ref_dir.mkdir(parents=True)
    for name in (
        "doctor_male_01.wav",
        "doctor_female_01.wav",
        "patient_male_01.wav",
        "patient_female_01.wav",
    ):
        (ref_dir / name).write_bytes(b"RIFF\x00\x00\x00\x00WAVEfake")
    config_path = configs_dir / "tts_profiles.yaml"
    config_path.write_text(_YAML_TEMPLATE, encoding="utf-8")
    return config_path, member_root


def test_load_profiles_parses_four_entries(tmp_path: Path) -> None:
    config_path, member_root = _write_catalog_fixture(tmp_path)

    catalog = load_profiles(config_path)

    assert isinstance(catalog, VoiceCatalog)
    assert catalog.sample_rate == 22050
    assert catalog.output_format == "wav"
    assert set(catalog.profiles) == {
        "doctor_male",
        "doctor_female",
        "patient_male",
        "patient_female",
    }
    doctor_m = catalog.profiles["doctor_male"]
    assert doctor_m.language == "pl"
    assert doctor_m.speed == 1.0
    # Reference path resolved relative to member root (parent of configs/).
    assert doctor_m.reference_audio == member_root / "reference_voices" / "doctor_male_01.wav"
    assert doctor_m.reference_audio.exists()


def test_load_profiles_respects_custom_repo_root(tmp_path: Path) -> None:
    config_path, member_root = _write_catalog_fixture(tmp_path)

    catalog = load_profiles(config_path, repo_root=member_root)
    assert catalog.profiles["patient_female"].speed == 0.95


def test_load_profiles_rejects_empty_yaml(tmp_path: Path) -> None:
    bad = tmp_path / "empty.yaml"
    bad.write_text("synthesis:\n  sample_rate: 16000\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no profiles"):
        load_profiles(bad)


def test_profile_for_patient_picks_by_sex(tmp_path: Path) -> None:
    config_path, _ = _write_catalog_fixture(tmp_path)
    catalog = load_profiles(config_path)

    male = profile_for(
        catalog,
        speaker=SpeakerRole.patient,
        patient_sex="M",
        scenario_id="scenario-a",
    )
    female = profile_for(
        catalog,
        speaker=SpeakerRole.patient,
        patient_sex="F",
        scenario_id="scenario-a",
    )
    assert male.name == "patient_male"
    assert female.name == "patient_female"


def test_profile_for_doctor_is_stable_per_scenario(tmp_path: Path) -> None:
    config_path, _ = _write_catalog_fixture(tmp_path)
    catalog = load_profiles(config_path)

    # Same scenario id → same doctor profile regardless of patient_sex.
    scenario = "11111111-2222-3333-4444-555555555555"
    a = profile_for(
        catalog, speaker=SpeakerRole.doctor, patient_sex="M", scenario_id=scenario
    )
    b = profile_for(
        catalog, speaker=SpeakerRole.doctor, patient_sex="F", scenario_id=scenario
    )
    c = profile_for(
        catalog, speaker=SpeakerRole.doctor, patient_sex="M", scenario_id=scenario
    )
    assert a.name == b.name == c.name
    assert a.name in {"doctor_male", "doctor_female"}


def test_profile_for_doctor_varies_across_scenarios(tmp_path: Path) -> None:
    config_path, _ = _write_catalog_fixture(tmp_path)
    catalog = load_profiles(config_path)

    # Across many distinct ids, both doctor profiles should show up. If the
    # hash is skewed we want to see it; pick 64 UUID-like inputs.
    seen: set[str] = set()
    for i in range(64):
        profile = profile_for(
            catalog,
            speaker=SpeakerRole.doctor,
            patient_sex="M",
            scenario_id=f"scenario-{i:04d}-aaaa-bbbb-cccc-dddddddddddd",
        )
        seen.add(profile.name)
    assert seen == {"doctor_male", "doctor_female"}


def test_profile_for_missing_key_raises(tmp_path: Path) -> None:
    config_path, _ = _write_catalog_fixture(tmp_path)
    catalog = load_profiles(config_path)
    # Drop a profile to simulate a bad config.
    broken = VoiceCatalog(
        profiles={k: v for k, v in catalog.profiles.items() if k != "patient_male"},
        sample_rate=catalog.sample_rate,
        output_format=catalog.output_format,
    )
    with pytest.raises(KeyError, match="patient_male"):
        profile_for(
            broken,
            speaker=SpeakerRole.patient,
            patient_sex="M",
            scenario_id="any",
        )
