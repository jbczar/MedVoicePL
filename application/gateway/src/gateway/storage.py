"""Disk-backed stores for `Session` and `Visit`.

Atomic-write idiom: write to `<path>.tmp`, then `os.replace`. Survives
crashes mid-write. Same idiom as
`on_premise.anonymization.audit_logger._atomic_write_json` —
duplicated here (6 LOC) because gateway cannot import on_premise per
trust-boundary lint.

State persists across gateway restarts. In-memory dict is the
read-through cache; every mutation re-writes the JSON sidecar
synchronously. Volume is small (one doctor, ~tens of visits/day) so
no async write queue.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID

from gateway.models import DoctorIdentity, Session, Visit
from shared.schemas import Specialization, VisitStatus

log = logging.getLogger(__name__)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _chmod_0600(path: Path) -> None:
    """Best-effort owner-only permissions; silently no-op on non-POSIX."""
    try:
        os.chmod(path, 0o600)
    except OSError as exc:
        log.warning("chmod 0600 failed for %s: %s", path, exc)


def _session_to_dict(session: Session) -> dict[str, Any]:
    return {
        "session_id": str(session.session_id),
        "doctor": {
            "username": session.doctor.username,
            "full_name": session.doctor.full_name,
        },
        "created_at": session.created_at.isoformat(),
        "expires_at": session.expires_at.isoformat(),
        "last_seen": session.last_seen.isoformat(),
    }


def _session_from_dict(data: dict[str, Any]) -> Session:
    return Session(
        session_id=UUID(data["session_id"]),
        doctor=DoctorIdentity(
            username=data["doctor"]["username"],
            full_name=data["doctor"].get("full_name", "Demo Lekarz"),
        ),
        created_at=datetime.fromisoformat(data["created_at"]),
        expires_at=datetime.fromisoformat(data["expires_at"]),
        last_seen=datetime.fromisoformat(data["last_seen"]),
    )


def _visit_to_dict(visit: Visit) -> dict[str, Any]:
    return {
        "visit_id": str(visit.visit_id),
        "session_id": str(visit.session_id),
        "status": visit.status.value,
        "specialization": visit.specialization.value if visit.specialization else None,
        "visit_date": visit.visit_date.isoformat() if visit.visit_date else None,
        "audio_path": str(visit.audio_path) if visit.audio_path else None,
        "audio_sha256": visit.audio_sha256,
        "transcript_id": visit.transcript_id,
        "error": visit.error,
        "created_at": visit.created_at.isoformat(),
        "completed_at": visit.completed_at.isoformat() if visit.completed_at else None,
    }


def _visit_from_dict(data: dict[str, Any]) -> Visit:
    return Visit(
        visit_id=UUID(data["visit_id"]),
        session_id=UUID(data["session_id"]),
        status=VisitStatus(data["status"]),
        specialization=(
            Specialization(data["specialization"]) if data["specialization"] else None
        ),
        visit_date=date.fromisoformat(data["visit_date"]) if data["visit_date"] else None,
        audio_path=Path(data["audio_path"]) if data["audio_path"] else None,
        audio_sha256=data["audio_sha256"],
        transcript_id=data["transcript_id"],
        error=data["error"],
        created_at=datetime.fromisoformat(data["created_at"]),
        completed_at=(
            datetime.fromisoformat(data["completed_at"]) if data["completed_at"] else None
        ),
    )


# ---------------------------------------------------------------------------
# Session store
# ---------------------------------------------------------------------------


class SessionStore(Protocol):
    """Read-write store for active doctor sessions."""

    def get(self, session_id: UUID) -> Session | None: ...
    def save(self, session: Session) -> None: ...
    def delete(self, session_id: UUID) -> None: ...
    def purge_expired(self, *, now: datetime | None = None) -> int: ...


class _DiskBackedSessionStore:
    """JSON-per-session under `<root>/<session_id>.json`, chmod 0600."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._cache: dict[UUID, Session] = {}
        self._loaded = False

    def _path(self, session_id: UUID) -> Path:
        return self._root / f"{session_id}.json"

    def _load_all(self) -> None:
        if self._loaded:
            return
        self._root.mkdir(parents=True, exist_ok=True)
        for path in self._root.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                session = _session_from_dict(data)
                self._cache[session.session_id] = session
            except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
                log.warning("skipping malformed session %s: %s", path.name, exc)
        self._loaded = True

    def get(self, session_id: UUID) -> Session | None:
        self._load_all()
        return self._cache.get(session_id)

    def save(self, session: Session) -> None:
        self._load_all()
        path = self._path(session.session_id)
        _atomic_write_json(path, _session_to_dict(session))
        _chmod_0600(path)
        self._cache[session.session_id] = session

    def delete(self, session_id: UUID) -> None:
        self._load_all()
        path = self._path(session_id)
        if path.exists():
            path.unlink()
        self._cache.pop(session_id, None)

    def purge_expired(self, *, now: datetime | None = None) -> int:
        self._load_all()
        now = now or datetime.now(UTC)
        expired = [sid for sid, s in self._cache.items() if s.is_expired(now=now)]
        for sid in expired:
            self.delete(sid)
        return len(expired)


def build_session_store(root: Path) -> SessionStore:
    return _DiskBackedSessionStore(root)


# ---------------------------------------------------------------------------
# Visit store
# ---------------------------------------------------------------------------


class VisitStore(Protocol):
    """Read-write store for visits, scoped per-session at the API layer."""

    def get(self, visit_id: UUID) -> Visit | None: ...
    def save(self, visit: Visit) -> None: ...
    def list_for_session(self, session_id: UUID) -> list[Visit]: ...
    def visit_dir(self, visit_id: UUID) -> Path: ...


class _DiskBackedVisitStore:
    """Per-visit directory `<root>/<visit_id>/{meta.json, audio.wav, ...}`."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._cache: dict[UUID, Visit] = {}
        self._loaded = False

    def visit_dir(self, visit_id: UUID) -> Path:
        d = self._root / str(visit_id)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _meta_path(self, visit_id: UUID) -> Path:
        return self.visit_dir(visit_id) / "meta.json"

    def _load_all(self) -> None:
        if self._loaded:
            return
        self._root.mkdir(parents=True, exist_ok=True)
        for child in self._root.iterdir():
            if not child.is_dir():
                continue
            meta = child / "meta.json"
            if not meta.exists():
                continue
            try:
                data = json.loads(meta.read_text(encoding="utf-8"))
                visit = _visit_from_dict(data)
                self._cache[visit.visit_id] = visit
            except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
                log.warning("skipping malformed visit %s: %s", child.name, exc)
        self._loaded = True

    def get(self, visit_id: UUID) -> Visit | None:
        self._load_all()
        return self._cache.get(visit_id)

    def save(self, visit: Visit) -> None:
        self._load_all()
        path = self._meta_path(visit.visit_id)
        _atomic_write_json(path, _visit_to_dict(visit))
        self._cache[visit.visit_id] = visit

    def list_for_session(self, session_id: UUID) -> list[Visit]:
        self._load_all()
        return [v for v in self._cache.values() if v.session_id == session_id]


def build_visit_store(root: Path) -> VisitStore:
    return _DiskBackedVisitStore(root)


__all__ = [
    "SessionStore",
    "VisitStore",
    "_atomic_write_json",
    "build_session_store",
    "build_visit_store",
]
