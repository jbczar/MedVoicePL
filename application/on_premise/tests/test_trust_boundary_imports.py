"""Lint-style boundary guards.

Two checks:

1. **Symbol-leak check**: `shared.schemas` must NOT expose any of the
   un-anonymized on-premise types (`Transcript`, `SOAPNote`, etc.). If
   someone later fills `shared/schemas/soap.py` with
   `from on_premise.structuring.soap_schema import SOAPNote`, the
   re-export would appear here and this test fails.

2. **Import-graph check** (AST-walk): no module under
   `application/shared/` may contain `from on_premise.<...> import ...`.
   This catches the failure mode one layer earlier — the actual import
   statement — even if the symbol isn't re-exported yet. Defense in
   depth ahead of Phase 2c's runtime `trust_gate.py`.
"""

from __future__ import annotations

import ast
import importlib
import pkgutil
from pathlib import Path

# All un-anonymized types that belong to on_premise and must not leak.
FORBIDDEN_NAMES = {
    # Phase 2a — transcripts
    "Transcript",
    "TranscriptSegment",
    "TranscriptionMetadata",
    "WordTiming",
    # Phase 2b — un-anonymized SOAP
    "SOAPNote",
    "SOAPNoteSubmission",
    "SOAPSubjective",
    "SOAPObjective",
    "SOAPAssessment",
    "SOAPPlan",
    "SOAPNoteMetadata",
    "Vital",
    "LabMention",
}


def test_shared_schemas_do_not_expose_private_symbols() -> None:
    shared_schemas = importlib.import_module("shared.schemas")

    # Walk every public module under shared.schemas and assert none of them
    # define/expose these symbols at module level.
    for info in pkgutil.iter_modules(shared_schemas.__path__, prefix="shared.schemas."):
        mod = importlib.import_module(info.name)
        public_names = {n for n in dir(mod) if not n.startswith("_")}
        leaked = public_names & FORBIDDEN_NAMES
        assert not leaked, (
            f"{info.name} exposes trust-boundary-private symbols: {sorted(leaked)}"
        )

    # Package itself shouldn't re-export them either.
    public_names = {n for n in dir(shared_schemas) if not n.startswith("_")}
    leaked = public_names & FORBIDDEN_NAMES
    assert not leaked, f"shared.schemas re-exports: {sorted(leaked)}"


def test_shared_does_not_import_from_on_premise() -> None:
    """AST-walk `application/shared/` and fail on any `on_premise.<...>` import.

    Catches the statement-level failure mode: someone fills
    `shared/schemas/soap.py` with `from on_premise.structuring.soap_schema
    import SOAPNote` — the type would bleed across the trust boundary
    even if `__init__.py` doesn't re-export it.
    """
    shared_root = Path(__file__).resolve().parents[3] / "shared" / "src" / "shared"
    # Resolve the actual shared root if the layout differs from the walk above.
    if not shared_root.exists():
        # Fallback: walk the installed shared package's file path.
        shared_pkg = importlib.import_module("shared")
        shared_root = Path(shared_pkg.__file__ or "").parent  # type: ignore[arg-type]

    offenders: list[tuple[str, int, str]] = []
    for py_path in shared_root.rglob("*.py"):
        try:
            tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
        except SyntaxError:  # pragma: no cover
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("on_premise"):
                offenders.append((str(py_path), node.lineno, node.module or ""))
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("on_premise"):
                        offenders.append((str(py_path), node.lineno, alias.name))

    assert not offenders, (
        "shared/ imports from on_premise/ — trust boundary breach:\n  "
        + "\n  ".join(f"{p}:{lineno} imports {mod}" for p, lineno, mod in offenders)
    )
