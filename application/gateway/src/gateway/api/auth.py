"""Authentication endpoints + session resolver dependency.

Single demo doctor: credentials read from env (`GATEWAY_DOCTOR_USERNAME`
+ `GATEWAY_DOCTOR_PASSWORD_HASH`). Login mints a UUID session, sets it
as an httpOnly cookie. The same UUID is later passed downstream as
`session_id` to the on-prem anonymizer (Phase 2c contract).

The `require_session` dependency is the gate other routers use; failure
modes are:
  - missing/blank cookie  → 401
  - cookie not in store   → 401
  - session expired       → 401 + delete from store
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, ConfigDict

from gateway.audit import GatewayAuditLogger
from gateway.config import GatewayConfig
from gateway.models import DoctorIdentity, Session
from gateway.security import verify_password
from gateway.storage import SessionStore

SESSION_COOKIE_NAME = "session_id"


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    username: str
    password: str


class DoctorIdentityResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    username: str
    full_name: str


# ---------------------------------------------------------------------------
# Dependency providers (overridden in tests via app.dependency_overrides).
# ---------------------------------------------------------------------------


def get_config() -> GatewayConfig:
    raise RuntimeError(
        "config dependency not wired — gateway.api.app must register an override"
    )


def get_session_store() -> SessionStore:
    raise RuntimeError(
        "session store dependency not wired — gateway.api.app must register an override"
    )


def get_audit_logger(request: Request) -> GatewayAuditLogger | None:
    """Return the audit logger if wired (None in tests that don't need it)."""
    return getattr(request.app.state, "audit", None)


# ---------------------------------------------------------------------------
# require_session dependency.
# ---------------------------------------------------------------------------


def require_session(
    session_id: Annotated[str | None, Cookie(alias=SESSION_COOKIE_NAME)] = None,
    store: SessionStore = Depends(get_session_store),
) -> Session:
    if not session_id:
        raise HTTPException(status_code=401, detail="not_authenticated")
    try:
        sid = UUID(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail="invalid_session") from exc
    session = store.get(sid)
    if session is None:
        raise HTTPException(status_code=401, detail="session_unknown")
    if session.is_expired():
        store.delete(sid)
        raise HTTPException(status_code=401, detail="session_expired")
    session.last_seen = datetime.now(UTC)
    store.save(session)
    return session


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


router = APIRouter(prefix="/v1/auth", tags=["auth"])


@router.post("/login", status_code=status.HTTP_204_NO_CONTENT)
def login(
    body: LoginRequest,
    request: Request,
    response: Response,
    config: Annotated[GatewayConfig, Depends(get_config)],
    store: Annotated[SessionStore, Depends(get_session_store)],
    audit: Annotated[GatewayAuditLogger | None, Depends(get_audit_logger)],
) -> Response:
    client_ip = request.client.host if request.client else "unknown"
    if body.username != config.doctor_username:
        if audit is not None:
            audit.log_login_attempt(username=body.username, ip=client_ip, success=False)
        raise HTTPException(status_code=401, detail="invalid_credentials")
    if not verify_password(body.password, config.doctor_password_hash):
        if audit is not None:
            audit.log_login_attempt(username=body.username, ip=client_ip, success=False)
        raise HTTPException(status_code=401, detail="invalid_credentials")
    now = datetime.now(UTC)
    session = Session(
        session_id=uuid4(),
        doctor=DoctorIdentity(username=config.doctor_username),
        created_at=now,
        expires_at=now + timedelta(hours=config.session_ttl_hours),
        last_seen=now,
    )
    store.save(session)
    if audit is not None:
        audit.log_login_attempt(username=body.username, ip=client_ip, success=True)
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=str(session.session_id),
        httponly=True,
        samesite="lax",
        secure=False,  # localhost demo; production sets True via reverse proxy
        max_age=config.session_ttl_hours * 3600,
    )
    response.status_code = status.HTTP_204_NO_CONTENT
    return response


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(
    response: Response,
    session: Annotated[Session, Depends(require_session)],
    store: Annotated[SessionStore, Depends(get_session_store)],
) -> Response:
    store.delete(session.session_id)
    response.delete_cookie(SESSION_COOKIE_NAME)
    response.status_code = status.HTTP_204_NO_CONTENT
    return response


@router.get("/me", response_model=DoctorIdentityResponse)
def me(
    session: Annotated[Session, Depends(require_session)],
) -> DoctorIdentityResponse:
    return DoctorIdentityResponse(
        username=session.doctor.username,
        full_name=session.doctor.full_name,
    )
