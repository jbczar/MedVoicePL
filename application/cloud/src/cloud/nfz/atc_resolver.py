"""Lazy-resolver for unknown drug names → ATC, with on-disk cache.

Bridge between the agent's free-text drug suggestions and our finite
NFZ index. When the agent proposes a drug whose ATC isn't in the
substance→ATC map (and therefore isn't in the NFZ index), this
resolver makes a single small Gemini call to look up the canonical
ATC, caches the result for next time, and lets the orchestrator
re-attempt the NFZ lookup.

Cache location: `application/cloud/data/nfz_formulary/atc_cache.json`
(gitignored). Format mirrors `substance_atc.json` for easy promotion
of cache entries into the curated map after periodic review.

Trust posture: lives in `cloud/` only — never crosses the boundary.
The resolved ATC enters `DrugSuggestion.atc_code`, which IS a
cross-boundary field, but the resolver itself is internal.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

log = logging.getLogger(__name__)


# Pydantic shape for Gemini structured output (single-drug lookup).
class AtcLookupResult(BaseModel):
    """Response shape for `_GeminiResolver.resolve`."""

    model_config = ConfigDict(extra="forbid")
    atc_code: str = Field(
        pattern=r"^[A-Z]\d{2}[A-Z]{2}\d{2}$|^UNKNOWN$",
        description="7-char ATC code, or 'UNKNOWN' if no canonical ATC exists.",
    )
    display_pl: str = Field(default="", description="Polish display name.")


_LOOKUP_PROMPT = """\
Podaj kanoniczny 7-znakowy kod ATC (WHO ATC/DDD) dla podanego leku/substancji.

Reguły:
1. Format ATC: 7 znaków — wielka litera, 2 cyfry, 2 wielkie litery, 2 cyfry \
   (np. `C07AB07`, `B01AF02`).
2. Dla preparatów złożonych podaj ATC właściwego preparatu złożonego, \
   nie pojedynczego składnika.
3. Jeśli nie istnieje kanoniczny kod ATC, zwróć `UNKNOWN`.
4. Polska nazwa: krótka, zwyczajowa (np. "Apixaban").

LEK: {drug_name}
"""


class LazyAtcResolver(Protocol):
    """The minimum surface the orchestrator needs."""

    def resolve(self, drug_name: str) -> str | None: ...
    """Returns canonical ATC for `drug_name`, or `None` if unresolvable."""


# ---------------------------------------------------------------------------
# Production resolver — Gemini-backed with disk cache.
# ---------------------------------------------------------------------------


@dataclass
class _GeminiResolver:
    """Single-drug ATC resolver via Vertex Gemini.

    Behaviour:

    * `resolve(name)` returns ATC string (canonical) or None.
    * In-memory cache (process lifetime) + on-disk cache (persistent).
    * Disk cache mirrors `substance_atc.json` shape, so reviewers can
      promote good entries to the curated map by hand.
    * Negative results (`UNKNOWN`) cached as `None` to skip re-querying.
    """

    llm: object  # cloud.llm_integration.llm_client.LLMClient (avoid circular)
    cache_path: Path
    _mem_cache: dict[str, str | None] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _disk_loaded: bool = False

    def resolve(self, drug_name: str) -> str | None:
        key = drug_name.strip()
        if not key:
            return None

        with self._lock:
            self._maybe_load_disk()
            if key in self._mem_cache:
                return self._mem_cache[key]
            # Case-insensitive fallback hit.
            lower = key.casefold()
            for existing_key, value in self._mem_cache.items():
                if existing_key.casefold() == lower:
                    return value

        # Cache miss — query LLM (outside the lock; the LLM call is slow).
        atc = self._query_llm(key)
        with self._lock:
            self._mem_cache[key] = atc
            self._persist()
        return atc

    def _query_llm(self, drug_name: str) -> str | None:
        prompt = _LOOKUP_PROMPT.format(drug_name=drug_name)
        try:
            result = self.llm.generate(prompt, response_schema=AtcLookupResult)  # type: ignore[attr-defined]
        except Exception as exc:  # last-resort safety
            log.warning("ATC resolver LLM call failed for %r: %s", drug_name, exc)
            return None
        if not isinstance(result, AtcLookupResult):
            log.warning(
                "ATC resolver got unexpected response type for %r: %s",
                drug_name,
                type(result).__name__,
            )
            return None
        if result.atc_code == "UNKNOWN":
            log.debug("ATC resolver: %r → UNKNOWN", drug_name)
            return None
        log.info("ATC resolver: %r → %s", drug_name, result.atc_code)
        return result.atc_code

    def _maybe_load_disk(self) -> None:
        if self._disk_loaded:
            return
        if self.cache_path.is_file():
            try:
                raw = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("could not load ATC cache at %s: %s", self.cache_path, exc)
            else:
                if isinstance(raw, dict):
                    for key, value in raw.items():
                        if value is None:
                            self._mem_cache[key] = None
                        elif isinstance(value, dict) and "atc" in value:
                            self._mem_cache[key] = str(value["atc"])
                        elif isinstance(value, str):
                            self._mem_cache[key] = value
        self._disk_loaded = True

    def _persist(self) -> None:
        """Write cache atomically; mirrors substance_atc.json shape."""
        payload: dict[str, dict[str, str] | None] = {}
        for key, value in self._mem_cache.items():
            if value is None:
                payload[key] = None
            else:
                payload[key] = {"atc": value, "display": ""}
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            os.replace(tmp, self.cache_path)
        except OSError as exc:
            log.warning("could not persist ATC cache: %s", exc)


# ---------------------------------------------------------------------------
# Test fixture — never calls a real LLM.
# ---------------------------------------------------------------------------


@dataclass
class _FakeAtcResolver:
    """Dictionary-backed resolver for unit tests."""

    canned: dict[str, str | None] = field(default_factory=dict)
    calls: int = 0

    def resolve(self, drug_name: str) -> str | None:
        self.calls += 1
        key = drug_name.strip()
        # Direct hit, then case-insensitive fallback.
        if key in self.canned:
            return self.canned[key]
        for existing_key, value in self.canned.items():
            if existing_key.casefold() == key.casefold():
                return value
        return None


@dataclass
class _NullResolver:
    """No-op resolver — always returns None. Default when LLM unavailable."""

    def resolve(self, drug_name: str) -> str | None:
        return None


def build_resolver(
    *,
    llm: object | None,
    cache_path: Path,
) -> LazyAtcResolver:
    """Factory: `_GeminiResolver` when `llm` is provided, else `_NullResolver`."""
    if llm is None:
        return _NullResolver()
    return _GeminiResolver(llm=llm, cache_path=cache_path)


__all__ = [
    "AtcLookupResult",
    "LazyAtcResolver",
    "_FakeAtcResolver",
    "_GeminiResolver",
    "_NullResolver",
    "build_resolver",
]
