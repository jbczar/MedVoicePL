"""Lint-style boundary guard.

Un-anonymized `Transcript` / `TranscriptSegment` types belong to `on_premise`.
If any module under `shared/` ever imports them (or redefines symbols with
those names), the trust boundary is compromised — cloud and gateway
dependencies would start pulling raw transcript structure into their import
graph.

Phase 2c replaces this with a runtime `trust_gate.py`; until then, this test
is the cheapest way to keep the boundary honest.
"""

from __future__ import annotations

import importlib
import pkgutil


def test_shared_schemas_do_not_expose_transcript() -> None:
    shared_schemas = importlib.import_module("shared.schemas")
    forbidden = {"Transcript", "TranscriptSegment", "TranscriptionMetadata", "WordTiming"}

    # Walk every public module under shared.schemas and assert none of them
    # define/expose these symbols at module level.
    for info in pkgutil.iter_modules(shared_schemas.__path__, prefix="shared.schemas."):
        mod = importlib.import_module(info.name)
        public_names = {n for n in dir(mod) if not n.startswith("_")}
        leaked = public_names & forbidden
        assert not leaked, (
            f"{info.name} exposes trust-boundary-private symbols: {sorted(leaked)}"
        )

    # Package itself shouldn't re-export them either.
    public_names = {n for n in dir(shared_schemas) if not n.startswith("_")}
    leaked = public_names & forbidden
    assert not leaked, f"shared.schemas re-exports: {sorted(leaked)}"
