"""NFZ refundacja in-memory index: ATC / substance / trade-name lookup.

The agent's `enrich_from_nfz` node hits this on the hot path — we want
O(1) ATC lookup, O(log n) substance search after Polish-normalised
casefold, and tolerant trade-name matching for cases where the LLM
proposes a brand name (e.g. ``"Eliquis"``) we have to map to a
substance (apixaban → ATC ``B01AF02``).

Index source: the JSON produced by
`application/cloud/data/nfz_formulary/ingest_nfz.py` from the bimonthly
NFZ XLSX. JSON shape (mirrors the cell-by-cell structure of the official
list, with `parse_reimbursement_string` already applied):

```json
{
  "snapshot_date": "2026-04-01",
  "entries": [
    {
      "atc_code": "C07AB07",
      "substance": "Bisoprolol",
      "trade_names": ["Bisocard", "Concor"],
      "reimbursement": "70%",
      "patient_copay_pln": "4.50",
      "indication_restrictions": ["niewydolność serca"]
    }
  ]
}
```

Designed as a `Protocol` so tests inject `_FakeNFZIndex` (canned
entries, no JSON parse). Same idiom as Phase 2b's `StructuredExtractor`
and Phase 2c's `RecognizerRegistry`.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol

from shared.schemas import ReimbursementLevel

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data carrier — one row of the NFZ refundacja list, normalized.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NFZEntry:
    """One drug as it appears in the NFZ refundacja list.

    Frozen so it can serve as a dict value without aliasing surprises.
    Lives in `cloud/` (NOT `shared/`) — it carries operational metadata
    (``indication_restrictions`` is free-form Polish, sometimes
    contains drug-specific dosing notes) that should not cross the
    trust boundary into gateway/frontend code. The cross-boundary
    type is `DrugSuggestion` in `shared.schemas.drug`.
    """

    atc_code: str
    substance: str
    trade_names: tuple[str, ...] = field(default_factory=tuple)
    reimbursement: ReimbursementLevel = ReimbursementLevel.none
    patient_copay_pln: Decimal | None = None
    indication_restrictions: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Polish-normalised key — the only place we strip diacritics + casefold.
# ---------------------------------------------------------------------------

_POLISH_DIACRITICS = str.maketrans(
    "ąćęłńóśźżĄĆĘŁŃÓŚŹŻ",
    "acelnoszzACELNOSZZ",
)


def normalize_drug_name(name: str) -> str:
    """Strip diacritics + casefold + collapse whitespace.

    Used as the key for substance + trade-name dictionaries. Polish
    trade names appear in mixed forms across NFZ releases:
    ``"Bisocard"``, ``"BISOCARD"``, ``"Bisocard 5 mg"``. Normalising
    to ``"bisocard"`` makes lookup robust without a fuzzy library.

    Note: NOT a Levenshtein-based fuzzy match — that's
    `drug_matcher.match_candidate`'s job. This is just casefold +
    diacritic strip.
    """
    return name.translate(_POLISH_DIACRITICS).casefold().strip()


# ---------------------------------------------------------------------------
# NFZIndex Protocol — testable abstraction over the JSON store.
# ---------------------------------------------------------------------------


class NFZIndex(Protocol):
    """The minimum surface the agent needs from an NFZ index.

    `_RealNFZIndex` is the production impl, `_FakeNFZIndex` is the
    test fixture. Both implement all four methods; tests instantiate
    fakes via `app.dependency_overrides`.
    """

    snapshot_date: date

    def lookup_by_atc(self, atc_code: str) -> NFZEntry | None: ...

    def lookup_by_substance(self, name: str) -> list[NFZEntry]: ...

    def lookup_by_trade_name(self, name: str) -> list[NFZEntry]: ...

    def __len__(self) -> int: ...

    def __iter__(self) -> Iterator[NFZEntry]: ...


# ---------------------------------------------------------------------------
# Real implementation — loads JSON written by ingest_nfz.py
# ---------------------------------------------------------------------------


@dataclass
class _RealNFZIndex:
    """Loads `index_<YYYY-MM-DD>.json` once and serves O(1) lookups.

    Build via `load_nfz_index(path)`; never construct directly outside
    that factory (it sets up the lookup dicts).
    """

    snapshot_date: date
    _by_atc: dict[str, NFZEntry] = field(default_factory=dict)
    _by_substance: dict[str, list[NFZEntry]] = field(default_factory=dict)
    _by_trade: dict[str, list[NFZEntry]] = field(default_factory=dict)

    def lookup_by_atc(self, atc_code: str) -> NFZEntry | None:
        return self._by_atc.get(atc_code.upper().strip())

    def lookup_by_substance(self, name: str) -> list[NFZEntry]:
        return list(self._by_substance.get(normalize_drug_name(name), ()))

    def lookup_by_trade_name(self, name: str) -> list[NFZEntry]:
        return list(self._by_trade.get(normalize_drug_name(name), ()))

    def __len__(self) -> int:
        return len(self._by_atc)

    def __iter__(self) -> Iterator[NFZEntry]:
        return iter(self._by_atc.values())


# ---------------------------------------------------------------------------
# Loader factory.
# ---------------------------------------------------------------------------


def load_nfz_index(path: Path) -> _RealNFZIndex:
    """Read `index_<YYYY-MM-DD>.json` and build the lookup dicts.

    Tolerant of unknown JSON keys (forward-compat with future ingest
    schema additions). Strict on the fields it cares about — bad
    `atc_code` or missing `substance` raises `ValueError`.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    snapshot_date = date.fromisoformat(raw["snapshot_date"])
    entries_raw: list[dict[str, Any]] = raw["entries"]

    by_atc: dict[str, NFZEntry] = {}
    by_substance: dict[str, list[NFZEntry]] = {}
    by_trade: dict[str, list[NFZEntry]] = {}

    for row in entries_raw:
        entry = _entry_from_json(row)
        # Multiple rows can share an ATC (different package sizes /
        # indications); keep the first and warn — agent only ever needs
        # the substance + reimbursement tier, which are stable per ATC.
        if entry.atc_code in by_atc:
            log.debug("duplicate ATC %s in NFZ index — keeping first", entry.atc_code)
            continue
        by_atc[entry.atc_code] = entry
        sub_key = normalize_drug_name(entry.substance)
        by_substance.setdefault(sub_key, []).append(entry)
        for trade in entry.trade_names:
            by_trade.setdefault(normalize_drug_name(trade), []).append(entry)

    log.info(
        "loaded NFZ index: snapshot=%s entries=%d substances=%d trade_names=%d",
        snapshot_date.isoformat(),
        len(by_atc),
        len(by_substance),
        len(by_trade),
    )
    return _RealNFZIndex(
        snapshot_date=snapshot_date,
        _by_atc=by_atc,
        _by_substance=by_substance,
        _by_trade=by_trade,
    )


def latest_index_path(directory: Path) -> Path | None:
    """Find the newest `index_*.json` under `directory`. ``None`` if absent."""
    if not directory.is_dir():
        return None
    candidates = sorted(directory.glob("index_*.json"))
    return candidates[-1] if candidates else None


# ---------------------------------------------------------------------------
# Test fixture — canned entries, no disk.
# ---------------------------------------------------------------------------


@dataclass
class _FakeNFZIndex:
    """In-memory index built from a list of `NFZEntry` instances.

    Lives in this module (not the test file) so tests in any cloud
    submodule can build it. Trust-boundary lint marks it as a forbidden
    name in `shared/schemas` — it's an on-prem-style operational type.
    """

    entries: tuple[NFZEntry, ...] = field(default_factory=tuple)
    snapshot_date: date = field(default_factory=lambda: date(2026, 4, 1))

    def __post_init__(self) -> None:
        # Build the same lookup dicts as the real loader. Wrapped in
        # __post_init__ so the public `entries` argument stays a clean
        # tuple of structured data.
        self._by_atc: dict[str, NFZEntry] = {}
        self._by_substance: dict[str, list[NFZEntry]] = {}
        self._by_trade: dict[str, list[NFZEntry]] = {}
        for entry in self.entries:
            self._by_atc[entry.atc_code] = entry
            sub_key = normalize_drug_name(entry.substance)
            self._by_substance.setdefault(sub_key, []).append(entry)
            for trade in entry.trade_names:
                self._by_trade.setdefault(normalize_drug_name(trade), []).append(entry)

    def lookup_by_atc(self, atc_code: str) -> NFZEntry | None:
        return self._by_atc.get(atc_code.upper().strip())

    def lookup_by_substance(self, name: str) -> list[NFZEntry]:
        return list(self._by_substance.get(normalize_drug_name(name), ()))

    def lookup_by_trade_name(self, name: str) -> list[NFZEntry]:
        return list(self._by_trade.get(normalize_drug_name(name), ()))

    def __len__(self) -> int:
        return len(self._by_atc)

    def __iter__(self) -> Iterator[NFZEntry]:
        return iter(self._by_atc.values())


# ---------------------------------------------------------------------------
# JSON serialization helpers (used by ingest CLI + load_nfz_index).
# ---------------------------------------------------------------------------


def _entry_from_json(row: dict[str, Any]) -> NFZEntry:
    """Decode one row of the JSON index into a frozen `NFZEntry`."""
    copay_raw = row.get("patient_copay_pln")
    return NFZEntry(
        atc_code=str(row["atc_code"]).upper().strip(),
        substance=str(row["substance"]).strip(),
        trade_names=tuple(row.get("trade_names") or ()),
        reimbursement=ReimbursementLevel(row.get("reimbursement", "nierefundowane")),
        patient_copay_pln=Decimal(copay_raw) if copay_raw is not None else None,
        indication_restrictions=tuple(row.get("indication_restrictions") or ()),
    )


def serialize_index(snapshot_date: date, entries: Iterable[NFZEntry]) -> dict[str, Any]:
    """Inverse of `load_nfz_index` — produces the JSON-serializable shape."""
    return {
        "schema_version": "3a.0",
        "snapshot_date": snapshot_date.isoformat(),
        "entries": [
            {
                "atc_code": e.atc_code,
                "substance": e.substance,
                "trade_names": list(e.trade_names),
                "reimbursement": e.reimbursement.value,
                "patient_copay_pln": (
                    str(e.patient_copay_pln) if e.patient_copay_pln is not None else None
                ),
                "indication_restrictions": list(e.indication_restrictions),
            }
            for e in entries
        ],
    }
