"""Lint-style boundary guards — extended in Phase 2c.

Three checks:

1. **Symbol-leak check** (negative contract): `shared.schemas` must NOT
   expose any of the un-anonymized on-premise types (`Transcript`,
   `SOAPNote`, etc.) or audit-trail types (`Replacement`,
   `PresidioAnonymizer`, etc.). If someone later fills
   `shared/schemas/soap.py` with
   `from on_premise.structuring.soap_schema import SOAPNote`, the
   re-export would appear here and this test fails.

2. **Symbol-presence check** (positive contract, new in 2c):
   `AnonymizedSOAPNote` MUST be importable from `shared.schemas`.
   Catches the inverse failure (someone refactors and the gateway can't
   import the contract).

3. **Import-graph check** (AST-walk): no module under
   `application/shared/`, `application/cloud/`, or
   `application/gateway/` may contain `from on_premise.<...> import ...`.
   Phase 2a covered `shared/`; Phase 2c extends to `cloud/` and
   `gateway/` because those modules begin to take shape during Phase 3,
   and the lint must already enforce the boundary against them.
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
    # Phase 2c — anonymizer audit types (hold references to original PII).
    "Replacement",
    "AuditLogger",
    "PresidioAnonymizer",
    "AnonymizationResult",
}

# Symbols that MUST be importable from `shared.schemas` for the
# cross-boundary contract to function. Each is the "yes, this is what
# crosses the boundary" anchor.
REQUIRED_SHARED_SYMBOLS = {
    "AnonymizedSOAPNote",
    "AnonymizedMetadata",
    "AnonymizedSubjective",
    "AnonymizedObjective",
    "AnonymizedAssessment",
    "AnonymizedPlan",
}

# Subpackages where the trust-boundary AST walk is enforced. Each path
# is relative to the repo root; we resolve to absolute below.
SCANNED_SUBPACKAGES = ("shared", "cloud", "gateway")


def test_shared_schemas_do_not_expose_private_symbols() -> None:
    """Negative contract — `shared.schemas` must not re-export forbidden types."""
    shared_schemas = importlib.import_module("shared.schemas")

    # Walk every public module under shared.schemas and assert none of them
    # define/expose these symbols at module level.
    for info in pkgutil.iter_modules(shared_schemas.__path__, prefix="shared.schemas."):
        mod = importlib.import_module(info.name)
        public_names = {n for n in dir(mod) if not n.startswith("_")}
        leaked = public_names & FORBIDDEN_NAMES
        assert not leaked, f"{info.name} exposes trust-boundary-private symbols: {sorted(leaked)}"

    # Package itself shouldn't re-export them either.
    public_names = {n for n in dir(shared_schemas) if not n.startswith("_")}
    leaked = public_names & FORBIDDEN_NAMES
    assert not leaked, f"shared.schemas re-exports: {sorted(leaked)}"


def test_shared_schemas_exposes_required_symbols() -> None:
    """Positive contract — `AnonymizedSOAPNote` (etc.) must be importable.

    The gateway and cloud modules import the trust-boundary contract
    from `shared.schemas`. If a refactor accidentally drops the
    re-export, downstream code breaks at import time. This test catches
    that at lint time.
    """
    shared_schemas = importlib.import_module("shared.schemas")
    public_names = {n for n in dir(shared_schemas) if not n.startswith("_")}
    missing = REQUIRED_SHARED_SYMBOLS - public_names
    assert not missing, f"shared.schemas does NOT re-export: {sorted(missing)}"

    # Each must round-trip via `from shared.schemas import X`.
    for symbol in REQUIRED_SHARED_SYMBOLS:
        assert hasattr(shared_schemas, symbol), f"shared.schemas.{symbol} not importable"


def _resolve_subpackage_root(name: str) -> Path | None:
    """Resolve `application/<name>/src/<name>` if it exists."""
    candidate = Path(__file__).resolve().parents[3] / name / "src" / name
    if candidate.exists():
        return candidate
    # Fallback: walk the installed package's __file__ if importable.
    try:
        pkg = importlib.import_module(name)
    except ModuleNotFoundError:
        return None
    pkg_file = pkg.__file__
    if pkg_file is None:
        return None
    return Path(pkg_file).parent


def test_no_subpackage_imports_from_on_premise() -> None:
    """AST-walk `shared/`, `cloud/`, `gateway/` and fail on any
    `from on_premise.<...> import ...` (or `import on_premise`).

    Catches the statement-level failure mode: someone fills
    `cloud/agent.py` with `from on_premise.structuring.soap_schema
    import SOAPNote` — the type would bleed across the trust boundary
    even if `__init__.py` doesn't re-export it.
    """
    offenders: list[tuple[str, int, str]] = []
    scanned_at_least_one = False
    for sub in SCANNED_SUBPACKAGES:
        root = _resolve_subpackage_root(sub)
        if root is None:
            continue  # subpackage not present yet (gateway/cloud may be stub-only)
        scanned_at_least_one = True
        for py_path in root.rglob("*.py"):
            try:
                tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
            except SyntaxError:  # pragma: no cover
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                    "on_premise"
                ):
                    offenders.append((str(py_path), node.lineno, node.module or ""))
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith("on_premise"):
                            offenders.append((str(py_path), node.lineno, alias.name))

    assert scanned_at_least_one, (
        "no subpackage roots found — does `application/shared/src/shared/` exist?"
    )
    assert not offenders, (
        "trust-boundary breach — these modules import from on_premise/:\n  "
        + "\n  ".join(f"{p}:{lineno} imports {mod}" for p, lineno, mod in offenders)
    )
