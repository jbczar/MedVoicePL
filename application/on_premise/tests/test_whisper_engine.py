"""Tests for `on_premise.stt.whisper_engine`.

No model downloads: we only exercise the factory's dispatch logic and the
Protocol contract. Real inference is verified end-to-end by the smoke-test
CLI, not here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from on_premise.stt.whisper_engine import (
    MLXWhisperTranscriber,
    RawTranscription,
    Transcriber,
    build_transcriber,
    load_whisper_config,
)


class _FakeTranscriber:
    """Satisfies the Transcriber Protocol without loading weights."""

    name = "fake:tiny"
    device = "cpu"

    def transcribe(self, wav_path: Path) -> RawTranscription:
        return RawTranscription(text="test", words=[], duration_s=0.0)


def test_fake_transcriber_satisfies_protocol() -> None:
    t: Transcriber = _FakeTranscriber()
    out = t.transcribe(Path("/dev/null"))
    assert out.text == "test"


def test_build_transcriber_mlx_dispatch_without_loading() -> None:
    # We can't load real MLX weights in a unit test, but we CAN assert that the
    # factory returns the MLX class — construction doesn't download until the
    # first `.transcribe()` call.
    env = {
        "WHISPER_BACKEND": "mlx",
        "WHISPER_MLX_REPO": "mlx-community/whisper-tiny",
    }
    transcriber = build_transcriber({}, env)
    assert isinstance(transcriber, MLXWhisperTranscriber)
    assert transcriber.device == "mps"
    assert "mlx-community/whisper-tiny" in transcriber.name


def test_build_transcriber_unknown_backend_raises() -> None:
    with pytest.raises(ValueError, match="unknown WHISPER_BACKEND"):
        build_transcriber({}, {"WHISPER_BACKEND": "whisperx"})


def test_load_whisper_config_expands_env_templates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WHISPER_MODEL_SIZE", "tiny")
    monkeypatch.setenv("WHISPER_DEVICE", "cpu")
    monkeypatch.setenv("WHISPER_COMPUTE_TYPE", "int8")
    cfg_path = tmp_path / "whisper.yaml"
    cfg_path.write_text(
        "model:\n"
        "  size: ${WHISPER_MODEL_SIZE}\n"
        "  device: ${WHISPER_DEVICE}\n"
        "  compute_type: ${WHISPER_COMPUTE_TYPE}\n"
        "  language: pl\n",
        encoding="utf-8",
    )
    cfg = load_whisper_config(cfg_path)
    assert cfg["model"]["size"] == "tiny"  # type: ignore[index]
    assert cfg["model"]["language"] == "pl"  # type: ignore[index]


def test_build_transcriber_default_backend_is_mlx() -> None:
    # No WHISPER_BACKEND in env -> default path is MLX.
    transcriber = build_transcriber({}, {})
    assert isinstance(transcriber, MLXWhisperTranscriber)
