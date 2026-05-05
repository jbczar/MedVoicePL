"""Tests for disk-backed session and visit stores."""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from gateway.models import DoctorIdentity, Session, Visit
from gateway.storage import build_session_store, build_visit_store
from shared.schemas import Specialization, VisitStatus


def _session(expires_in: timedelta = timedelta(hours=8)) -> Session:
    now = datetime.now(UTC)
    return Session(
        session_id=uuid4(),
        doctor=DoctorIdentity(username="demo_doctor"),
        created_at=now,
        expires_at=now + expires_in,
        last_seen=now,
    )


def _visit(session_id=None) -> Visit:
    return Visit(
        visit_id=uuid4(),
        session_id=session_id or uuid4(),
        status=VisitStatus.draft,
        specialization=Specialization.kardiologia,
        visit_date=date(2026, 5, 5),
    )


def test_session_save_and_get_round_trip(tmp_path: Path) -> None:
    store = build_session_store(tmp_path / "sessions")
    s = _session()
    store.save(s)
    out = store.get(s.session_id)
    assert out is not None
    assert out.session_id == s.session_id
    assert out.doctor.username == "demo_doctor"


def test_session_get_unknown_returns_none(tmp_path: Path) -> None:
    store = build_session_store(tmp_path / "sessions")
    assert store.get(uuid4()) is None


def test_session_persists_across_store_instances(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    s = _session()
    build_session_store(root).save(s)
    out = build_session_store(root).get(s.session_id)
    assert out is not None
    assert out.session_id == s.session_id


def test_session_chmod_0600_on_posix(tmp_path: Path) -> None:
    if sys.platform == "win32":
        return
    store = build_session_store(tmp_path / "sessions")
    s = _session()
    store.save(s)
    path = tmp_path / "sessions" / f"{s.session_id}.json"
    mode = os.stat(path).st_mode & 0o777
    assert mode == 0o600


def test_session_delete_removes_disk_and_cache(tmp_path: Path) -> None:
    store = build_session_store(tmp_path / "sessions")
    s = _session()
    store.save(s)
    store.delete(s.session_id)
    assert store.get(s.session_id) is None
    assert not (tmp_path / "sessions" / f"{s.session_id}.json").exists()


def test_session_purge_expired(tmp_path: Path) -> None:
    store = build_session_store(tmp_path / "sessions")
    fresh = _session(expires_in=timedelta(hours=1))
    stale = _session(expires_in=timedelta(seconds=-1))
    store.save(fresh)
    store.save(stale)
    purged = store.purge_expired()
    assert purged == 1
    assert store.get(fresh.session_id) is not None
    assert store.get(stale.session_id) is None


def test_session_atomic_write_no_partial_file(tmp_path: Path) -> None:
    store = build_session_store(tmp_path / "sessions")
    s = _session()
    store.save(s)
    path = tmp_path / "sessions" / f"{s.session_id}.json"
    assert path.exists()
    assert not path.with_suffix(".json.tmp").exists()
    json.loads(path.read_text())  # parses cleanly


def test_visit_save_and_get_round_trip(tmp_path: Path) -> None:
    store = build_visit_store(tmp_path / "visits")
    v = _visit()
    store.save(v)
    out = store.get(v.visit_id)
    assert out is not None
    assert out.visit_id == v.visit_id
    assert out.specialization == Specialization.kardiologia
    assert out.visit_date == date(2026, 5, 5)


def test_visit_meta_under_per_visit_dir(tmp_path: Path) -> None:
    store = build_visit_store(tmp_path / "visits")
    v = _visit()
    store.save(v)
    assert (tmp_path / "visits" / str(v.visit_id) / "meta.json").exists()


def test_visit_persists_across_store_instances(tmp_path: Path) -> None:
    root = tmp_path / "visits"
    v = _visit()
    build_visit_store(root).save(v)
    out = build_visit_store(root).get(v.visit_id)
    assert out is not None
    assert out.visit_id == v.visit_id


def test_visit_list_for_session_isolates_per_session(tmp_path: Path) -> None:
    store = build_visit_store(tmp_path / "visits")
    sid_a = uuid4()
    sid_b = uuid4()
    store.save(_visit(session_id=sid_a))
    store.save(_visit(session_id=sid_a))
    store.save(_visit(session_id=sid_b))
    a = store.list_for_session(sid_a)
    b = store.list_for_session(sid_b)
    assert len(a) == 2
    assert len(b) == 1


def test_visit_status_transition_is_persisted(tmp_path: Path) -> None:
    root = tmp_path / "visits"
    store = build_visit_store(root)
    v = _visit()
    store.save(v)
    v.status = VisitStatus.audio_uploaded
    v.audio_sha256 = "deadbeef" * 8
    store.save(v)
    out = build_visit_store(root).get(v.visit_id)
    assert out is not None
    assert out.status == VisitStatus.audio_uploaded
    assert out.audio_sha256 == "deadbeef" * 8


def test_visit_dir_is_idempotent(tmp_path: Path) -> None:
    store = build_visit_store(tmp_path / "visits")
    vid = uuid4()
    d1 = store.visit_dir(vid)
    d2 = store.visit_dir(vid)
    assert d1 == d2
    assert d1.is_dir()
