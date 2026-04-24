"""Tests for `on_premise.structuring.slm_engine`.

No model downloads — `build_extractor` is exercised by class dispatch
only, and the Protocol shape is verified by a local `_StubExtractor`.
Actual MLX-LM / llama-cpp loading is preflight work (see
`scripts/check_slm_models.py`).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from on_premise.structuring.slm_engine import (
    PermanentSLMError,
    build_extractor,
    load_slm_config,
)


class _StubExtractor:
    """Canned extractor satisfying the `StructuredExtractor` Protocol."""

    name = "mlx-lm:stub"
    device = "cpu"
    structured_mode = "grammar"

    def __init__(self, response: str) -> None:
        self._response = response
        self.calls: list[tuple[str, Mapping[str, Any]]] = []

    def extract(self, prompt: str, *, schema: Mapping[str, Any]) -> str:
        self.calls.append((prompt, schema))
        return self._response


def test_stub_satisfies_protocol_by_duck_type() -> None:
    """StructuredExtractor isn't @runtime_checkable; verify shape manually."""
    stub = _StubExtractor(response="{}")
    assert hasattr(stub, "name")
    assert hasattr(stub, "device")
    assert hasattr(stub, "structured_mode")
    out = stub.extract("prompt", schema={"type": "object"})
    assert out == "{}"
    assert stub.calls[0][0] == "prompt"


def test_build_extractor_unknown_backend() -> None:
    with pytest.raises(PermanentSLMError, match="unknown SLM_BACKEND"):
        build_extractor({"backend": "gpt5"}, env={"SLM_BACKEND": "gpt5"})


def test_build_extractor_mlx_lm_dispatches_to_mlx(monkeypatch: pytest.MonkeyPatch) -> None:
    """Class dispatch only — we don't actually load the model."""
    # Swap `mlx_lm` import with a dummy module so the constructor's
    # import-time presence check passes.
    import sys
    import types

    dummy = types.ModuleType("mlx_lm")
    monkeypatch.setitem(sys.modules, "mlx_lm", dummy)

    # build_extractor instantiates MLXLMExtractor, which only records
    # config at __init__ time; actual model load is lazy via `_load()`.
    extractor = build_extractor(
        {"backend": "mlx-lm", "model": {"mlx_repo": "some-repo"}},
        env={
            "SLM_BACKEND": "mlx-lm",
            "SLM_MLX_REPO": "mlx-community/foo",
            "SLM_STRUCTURED_MODE": "raw",
            "SLM_MAX_NEW_TOKENS": "512",
            "SLM_TEMPERATURE": "0.2",
        },
    )
    assert type(extractor).__name__ == "MLXLMExtractor"
    assert extractor.name == "mlx-lm:mlx-community/foo"
    assert extractor.device == "mps"
    assert extractor.structured_mode == "raw"


def test_build_extractor_llama_cpp_requires_existing_path(tmp_path: Path) -> None:
    with pytest.raises(PermanentSLMError, match="does not exist"):
        build_extractor(
            {"backend": "llama-cpp"},
            env={"SLM_BACKEND": "llama-cpp", "SLM_GGUF_PATH": str(tmp_path / "ghost.gguf")},
        )


def test_load_slm_config_expands_env_templates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_path = tmp_path / "slm.yaml"
    cfg_path.write_text(
        "backend: ${SLM_BACKEND_TEST}\n"
        "model:\n"
        "  id: ${SLM_MODEL_ID_TEST}\n"
        "  max_new_tokens: 2048\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SLM_BACKEND_TEST", "mlx-lm")
    monkeypatch.setenv("SLM_MODEL_ID_TEST", "speakleash/Bielik-11B")

    cfg = load_slm_config(cfg_path)
    assert cfg["backend"] == "mlx-lm"
    assert cfg["model"]["id"] == "speakleash/Bielik-11B"
    assert cfg["model"]["max_new_tokens"] == 2048


def test_load_slm_config_resolves_unset_templates_to_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_path = tmp_path / "slm.yaml"
    cfg_path.write_text("backend: ${UNSET_VAR_OF_DOOM}\n", encoding="utf-8")
    monkeypatch.delenv("UNSET_VAR_OF_DOOM", raising=False)

    cfg = load_slm_config(cfg_path)
    # Unset templates resolve to None so that `build_extractor`'s
    # hardcoded fallbacks kick in. This makes the pipeline forgiving of
    # stale `.env` files that don't yet have the Phase 2b block — the
    # common failure mode when the config is added after users already
    # set up their `.env` from Phase 2a.
    assert cfg["backend"] is None


def test_build_extractor_structured_mode_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    import types

    monkeypatch.setitem(sys.modules, "mlx_lm", types.ModuleType("mlx_lm"))
    with pytest.raises(PermanentSLMError, match="structured_mode"):
        build_extractor(
            {"backend": "mlx-lm"},
            env={
                "SLM_BACKEND": "mlx-lm",
                "SLM_MLX_REPO": "x",
                "SLM_STRUCTURED_MODE": "bogus",
            },
        )


# Silence unused-import warning on `os` when running `-q`
_ = os
