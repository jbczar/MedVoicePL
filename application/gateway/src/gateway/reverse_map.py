"""Token reverse-map for the doctor's view.

The cloud agent's `AgentSuggestion` may contain placeholders like
``<PERSON_1>``, ``<LOCATION_2>`` injected by the on-prem anonymizer.
The doctor's UI must show original names. This module reads the
on-prem token map (``application/on_premise/audit/<date>/<session_id>/token_map.json``)
and walks the suggestion tree replacing placeholders with originals.

The token map NEVER goes to the cloud — gateway holds it briefly,
applies it to the response, and discards. Token map files are chmod
0600 on disk; gateway must run as the same user as the on-prem
process to read them.

Fail-soft: any token not in the map stays as-is and is logged. Audit
records the count of unmapped tokens for ops review.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date
from pathlib import Path
from uuid import UUID

from shared.schemas import AgentSuggestion

log = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"<[A-Z_]+_\d+>")


class TokenMapNotFoundError(FileNotFoundError):
    """Raised when the expected token map file does not exist."""


def load_token_map(
    *,
    audit_root: Path,
    session_id: UUID,
    visit_date: date | None = None,
) -> dict[str, str]:
    """Load the on-prem token map for a session.

    `audit_root` is the on-prem audit dir (chmod 0600 file inside).
    `visit_date` narrows the date partition; if None, scan all date
    folders for a matching session subdirectory.
    """
    sid = str(session_id)
    candidates: list[Path] = []
    if visit_date is not None:
        path = audit_root / visit_date.isoformat() / sid / "token_map.json"
        if path.is_file():
            candidates.append(path)
    if not candidates and audit_root.is_dir():
        # Fall back to scanning all date folders.
        for child in sorted(audit_root.iterdir()):
            if not child.is_dir():
                continue
            path = child / sid / "token_map.json"
            if path.is_file():
                candidates.append(path)
    if not candidates:
        raise TokenMapNotFoundError(f"no token_map.json for session {sid} under {audit_root}")
    # Use the most recent (date-folder names sort lexicographically).
    chosen = candidates[-1]
    try:
        data = json.loads(chosen.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TokenMapNotFoundError(f"could not read token map: {chosen}: {exc}") from exc
    if not isinstance(data, dict):
        raise TokenMapNotFoundError(f"token map at {chosen} is not a JSON object")
    return {str(k): str(v) for k, v in data.items()}


def reveal_text(text: str, token_map: dict[str, str]) -> tuple[str, int]:
    """Replace `<{ENTITY}_{N}>` tokens in `text`.

    Returns the revealed text and the count of unmapped tokens that
    remained as-is.
    """
    unmapped = 0

    def _sub(match: re.Match[str]) -> str:
        nonlocal unmapped
        token = match.group(0)
        replacement = token_map.get(token)
        if replacement is None:
            unmapped += 1
            return token
        return replacement

    return _TOKEN_RE.sub(_sub, text), unmapped


def reveal_dict(
    data: object,
    *,
    token_map: dict[str, str],
) -> tuple[object, int]:
    """Generic tree-walk reveal — operates on plain dict/list/str.

    Used by SOAP and transcript endpoints which don't have a Pydantic
    type to validate against on the gateway side (they pass through
    as-is to the frontend).
    """
    total_unmapped = 0

    def _walk(node: object) -> object:
        nonlocal total_unmapped
        if isinstance(node, str):
            revealed, missing = reveal_text(node, token_map)
            total_unmapped += missing
            return revealed
        if isinstance(node, list):
            return [_walk(item) for item in node]
        if isinstance(node, dict):
            return {key: _walk(value) for key, value in node.items()}
        return node

    return _walk(data), total_unmapped


def reveal_for_doctor(
    suggestion: AgentSuggestion,
    *,
    token_map: dict[str, str],
) -> tuple[AgentSuggestion, int]:
    """Walk an `AgentSuggestion`, replace placeholders, return new instance.

    Returns the revealed suggestion and the total count of unmapped
    placeholders (zero is the happy path; non-zero is logged at WARN).
    """
    dumped = suggestion.model_dump(mode="json")
    revealed_dump, total_unmapped = reveal_dict(dumped, token_map=token_map)
    revealed = AgentSuggestion.model_validate(revealed_dump)
    if total_unmapped:
        log.warning("reverse_map: %d unmapped placeholders remained", total_unmapped)
    return revealed, total_unmapped


__all__ = [
    "TokenMapNotFoundError",
    "load_token_map",
    "reveal_dict",
    "reveal_for_doctor",
    "reveal_text",
]
