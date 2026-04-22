"""End-to-end test for `pipeline.run_audio` using a fake synthesizer.

The test wires real `voice_manager`, `dialog_mixer`, `audio_utils`, and
`audio_schema` modules together — only the XTTSv2 model is swapped for a
`_FakeSynthesizer` that returns deterministic short silence. No network
traffic, no 2 GB model download, no torch-only GPU path.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from dataset_pipeline.audio_schema import ScenarioAudioBundle
from dataset_pipeline.pipeline import run_audio
from dataset_pipeline.tts_engine.voice_manager import VoiceProfile

FIXTURE_SCENARIO = Path(__file__).parent / "fixtures" / "tiny_scenario.json"


TTS_YAML = """\
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

POSTPROCESS_YAML = """\
noise:
  enabled: false

crossfade:
  duration_ms: 200
  pause_between_turns_ms:
    min: 300
    max: 1500

output:
  sample_rate: 16000
  format: "wav"
  normalize: true
"""


class _FakeSynthesizer:
    """Returns 0.5 s of silence per turn; no torch, no model load."""

    sample_rate: int = 22050
    model_name: str = "fake-xtts"
    device: str = "cpu"

    def __init__(self, duration_s: float = 0.5) -> None:
        self._duration_s = duration_s
        self.calls: list[tuple[str, str]] = []

    def synthesize(
        self, text: str, profile: VoiceProfile, *, seed: int | None = None
    ) -> np.ndarray:
        self.calls.append((text, profile.name))
        del seed  # fake is already deterministic; accept and ignore.
        n = int(self._duration_s * self.sample_rate)
        # Tiny ramp so the signal is non-trivial but bounded [-0.01, 0.01].
        return (np.linspace(-0.01, 0.01, n, dtype=np.float32))


def _build_env(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Lay out scenarios/, audio/, configs/, reference_voices/ under tmp_path."""
    member_root = tmp_path / "dataset_pipeline"
    scenarios_dir = member_root / "output" / "scenarios"
    audio_dir = member_root / "output" / "audio"
    configs_dir = member_root / "configs"
    ref_dir = member_root / "reference_voices"
    for d in (scenarios_dir, audio_dir, configs_dir, ref_dir):
        d.mkdir(parents=True, exist_ok=True)

    shutil.copy(FIXTURE_SCENARIO, scenarios_dir / "tiny_scenario.json")

    # Fake reference WAVs — existence is not checked by _FakeSynthesizer but
    # keeps the test realistic in case future code adds an early guard.
    for name in (
        "doctor_male_01.wav",
        "doctor_female_01.wav",
        "patient_male_01.wav",
        "patient_female_01.wav",
    ):
        (ref_dir / name).write_bytes(b"RIFF0000WAVEfake")

    tts_cfg = configs_dir / "tts_profiles.yaml"
    tts_cfg.write_text(TTS_YAML, encoding="utf-8")
    post_cfg = configs_dir / "audio_postprocess.yaml"
    post_cfg.write_text(POSTPROCESS_YAML, encoding="utf-8")

    return scenarios_dir, audio_dir, tts_cfg, post_cfg


def test_run_audio_produces_bundle(tmp_path: Path) -> None:
    scenarios_dir, audio_dir, tts_cfg, post_cfg = _build_env(tmp_path)
    fake = _FakeSynthesizer()

    report = run_audio(
        scenarios_dir=scenarios_dir,
        audio_out_dir=audio_dir,
        tts_config_path=tts_cfg,
        postprocess_config_path=post_cfg,
        synthesizer=fake,
    )

    assert report.success_count == 1
    assert report.failures == []
    assert len(fake.calls) == 6  # tiny fixture has 6 turns

    bundle = report.bundles[0]
    # scenario id starts with 11111111 — bundle dir uses first 8 chars.
    bundle_dir = audio_dir / "11111111"
    assert bundle_dir.exists()

    # Per-turn WAVs present.
    turn_wavs = sorted(bundle_dir.glob("turn_*.wav"))
    assert len(turn_wavs) == 6
    # Turn 0 should be doctor (fixture starts with doctor).
    assert turn_wavs[0].name.startswith("turn_00_doctor")

    # dialogue.wav present and non-empty.
    dialogue = bundle_dir / "dialogue.wav"
    assert dialogue.exists() and dialogue.stat().st_size > 0

    # bundle.json sidecar present and validates.
    sidecar = bundle_dir / "bundle.json"
    parsed = ScenarioAudioBundle.model_validate_json(sidecar.read_text(encoding="utf-8"))
    assert parsed.scenario_id == bundle.scenario_id
    assert len(parsed.turns) == 6
    assert parsed.dialogue_sample_rate == 16000  # output sr from postprocess yaml
    assert parsed.tts_model == "fake-xtts"
    assert parsed.tts_device == "cpu"


def test_run_audio_dialogue_sha_is_deterministic(tmp_path: Path) -> None:
    scenarios_dir, audio_dir, tts_cfg, post_cfg = _build_env(tmp_path)

    # First run.
    run_audio(
        scenarios_dir=scenarios_dir,
        audio_out_dir=audio_dir,
        tts_config_path=tts_cfg,
        postprocess_config_path=post_cfg,
        synthesizer=_FakeSynthesizer(),
    )
    dialogue_path = audio_dir / "11111111" / "dialogue.wav"
    sha_first = hashlib.sha256(dialogue_path.read_bytes()).hexdigest()

    # Wipe and re-run with a fresh fake — same scenario id → same mixer seed.
    shutil.rmtree(audio_dir)
    audio_dir.mkdir(parents=True)
    run_audio(
        scenarios_dir=scenarios_dir,
        audio_out_dir=audio_dir,
        tts_config_path=tts_cfg,
        postprocess_config_path=post_cfg,
        synthesizer=_FakeSynthesizer(),
    )
    sha_second = hashlib.sha256(dialogue_path.read_bytes()).hexdigest()

    assert sha_first == sha_second


def test_run_audio_respects_limit(tmp_path: Path) -> None:
    scenarios_dir, audio_dir, tts_cfg, post_cfg = _build_env(tmp_path)
    # Duplicate the fixture so scenarios_dir has two files; limit=1 should stop.
    shutil.copy(
        scenarios_dir / "tiny_scenario.json", scenarios_dir / "tiny_scenario_b.json"
    )
    # Need to give the copy a different id or the 8-char bundle dir collides.
    data = json.loads((scenarios_dir / "tiny_scenario_b.json").read_text(encoding="utf-8"))
    data["id"] = "22222222-aaaa-bbbb-cccc-dddddddddddd"
    (scenarios_dir / "tiny_scenario_b.json").write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8"
    )

    report = run_audio(
        scenarios_dir=scenarios_dir,
        audio_out_dir=audio_dir,
        tts_config_path=tts_cfg,
        postprocess_config_path=post_cfg,
        synthesizer=_FakeSynthesizer(),
        limit=1,
    )
    assert report.success_count == 1
    bundle_dirs = sorted(p for p in audio_dir.iterdir() if p.is_dir())
    assert len(bundle_dirs) == 1


NOISY_POSTPROCESS_YAML = """\
crossfade:
  duration_ms: 200
  pause_between_turns_ms:
    min: 300
    max: 1500

augment:
  enabled: true
  pitch_semitones: {min: -0.5, max: 0.5}
  tempo_factor: {min: 0.98, max: 1.02}

reverb:
  enabled: true
  room_sizes:
    - {name: small, size: 0.2, weight: 1.0}
  damping: 0.5
  wet_level: 0.2
  dry_level: 0.8

noise:
  enabled: true
  library_path: noise_library/
  snr_buckets:
    - {min_db: 15, max_db: 15, weight: 1.0}

output:
  sample_rate: 16000
  format: "wav"
  normalize: true
"""


def _write_noise_library(library_dir: Path) -> None:
    """Drop a couple of synthetic white-noise clips for the injector."""
    library_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(2):
        data = (rng.standard_normal(22050 * 2) * 0.2).astype(np.float32)
        sf.write(str(library_dir / f"noise_{i:02d}.wav"), data, 22050, subtype="PCM_16")


def test_run_audio_noisy_variant_end_to_end(tmp_path: Path) -> None:
    scenarios_dir, audio_dir, tts_cfg, _post_cfg = _build_env(tmp_path)
    # Override postprocess config with 1b.1-enabled variant.
    post_cfg = tts_cfg.parent / "audio_postprocess.yaml"
    post_cfg.write_text(NOISY_POSTPROCESS_YAML, encoding="utf-8")
    # Noise library sits next to the postprocess config (same resolution
    # logic as in production).
    _write_noise_library(post_cfg.parent / "noise_library")

    report = run_audio(
        scenarios_dir=scenarios_dir,
        audio_out_dir=audio_dir,
        tts_config_path=tts_cfg,
        postprocess_config_path=post_cfg,
        synthesizer=_FakeSynthesizer(duration_s=0.3),
    )
    assert report.success_count == 1
    bundle = report.bundles[0]

    # Clean + noisy both on disk.
    bundle_dir = audio_dir / "11111111"
    assert (bundle_dir / "dialogue.wav").exists()
    assert (bundle_dir / "dialogue_noisy.wav").exists()

    # Bundle metadata populated.
    assert bundle.dialogue_noisy_path == "audio/11111111/dialogue_noisy.wav"
    assert bundle.dialogue_noisy_sha256 is not None
    assert bundle.dialogue_noisy_duration_s is not None
    assert bundle.noise_snr_db == pytest.approx(15.0)
    assert bundle.noise_clip is not None and bundle.noise_clip.startswith("noise_")
    assert bundle.reverb_room == "small"

    # Clean mix SHA still stable w.r.t. clean-only runs (independent RNG).
    sidecar = ScenarioAudioBundle.model_validate_json(
        (bundle_dir / "bundle.json").read_text(encoding="utf-8")
    )
    assert sidecar.dialogue_sha256 == bundle.dialogue_sha256


def test_run_audio_clean_only_skips_noisy(tmp_path: Path) -> None:
    scenarios_dir, audio_dir, tts_cfg, _post_cfg = _build_env(tmp_path)
    post_cfg = tts_cfg.parent / "audio_postprocess.yaml"
    post_cfg.write_text(NOISY_POSTPROCESS_YAML, encoding="utf-8")
    # Intentionally skip noise library — clean_only must not touch it.

    report = run_audio(
        scenarios_dir=scenarios_dir,
        audio_out_dir=audio_dir,
        tts_config_path=tts_cfg,
        postprocess_config_path=post_cfg,
        synthesizer=_FakeSynthesizer(duration_s=0.3),
        clean_only=True,
    )
    assert report.success_count == 1
    bundle_dir = audio_dir / "11111111"
    assert (bundle_dir / "dialogue.wav").exists()
    assert not (bundle_dir / "dialogue_noisy.wav").exists()
    bundle = report.bundles[0]
    assert bundle.dialogue_noisy_path is None
    assert bundle.noise_snr_db is None


def test_run_audio_scenario_filter(tmp_path: Path) -> None:
    scenarios_dir, audio_dir, tts_cfg, post_cfg = _build_env(tmp_path)
    # Add a second scenario that the filter should exclude.
    extra = json.loads(FIXTURE_SCENARIO.read_text(encoding="utf-8"))
    extra["id"] = "99999999-aaaa-bbbb-cccc-dddddddddddd"
    (scenarios_dir / "ninety_nine.json").write_text(
        json.dumps(extra, ensure_ascii=False), encoding="utf-8"
    )

    report = run_audio(
        scenarios_dir=scenarios_dir,
        audio_out_dir=audio_dir,
        tts_config_path=tts_cfg,
        postprocess_config_path=post_cfg,
        synthesizer=_FakeSynthesizer(),
        scenario_filter=["tiny"],
    )
    assert report.success_count == 1
    assert report.bundles[0].scenario_id.startswith("11111111")
