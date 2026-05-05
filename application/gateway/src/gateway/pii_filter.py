"""PII-stripping structlog processor.

Walks every `event_dict` value before serialization. Any string that
matches a PESEL/phone/email/NIP regex gets replaced with `[REDACTED]`.

Patterns mirror Phase 2c's `trust_gate.py`. We deliberately duplicate
the regexes here (gateway cannot import on_premise) — boundary walls
duplicate by design. If patterns drift, both must be updated in sync.

This is a *log* filter, not a trust gate. Gateway request bodies are
parsed before logging; this catches accidental PII leakage in
exception messages, structured fields, third-party log frames, etc.
"""

from __future__ import annotations

import re
from typing import Any

# Mirror Phase 2c trust_gate patterns; duplication is the price of
# the gateway/on-prem trust boundary.
_PESEL_RE = re.compile(r"\b\d{11}\b")
_NIP_RE = re.compile(r"\b\d{10}\b")
_PHONE_RE = re.compile(
    r"\+?\d{2}[\s-]?\d{3}[\s-]?\d{3}[\s-]?\d{3}|\b\d{9}\b"
)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")

REDACTED = "[REDACTED]"


def strip_pii_from_string(value: str) -> str:
    """Public helper: redact PII patterns from a raw string."""
    out = _EMAIL_RE.sub(REDACTED, value)
    out = _PESEL_RE.sub(REDACTED, out)
    out = _NIP_RE.sub(REDACTED, out)
    out = _PHONE_RE.sub(REDACTED, out)
    return out


def _walk(value: Any) -> Any:
    if isinstance(value, str):
        return strip_pii_from_string(value)
    if isinstance(value, dict):
        return {k: _walk(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        cls = type(value)
        return cls(_walk(item) for item in value)
    return value


def pii_stripper(_logger: Any, _name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """structlog processor signature: `(logger, method_name, event_dict)`."""
    return {key: _walk(val) for key, val in event_dict.items()}


__all__ = ["pii_stripper", "strip_pii_from_string"]
