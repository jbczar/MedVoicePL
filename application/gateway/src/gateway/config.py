"""Gateway configuration loaded from process environment.

Single source of truth for env-driven settings. Built once per app
lifespan and dependency-injected. Tests override by constructing
`GatewayConfig` directly with desired values.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GatewayConfig:
    host: str
    port: int
    doctor_username: str
    doctor_password_hash: str
    session_secret: str
    session_ttl_hours: int
    on_premise_api_url: str
    cloud_api_url: str
    audit_retention_days: int
    cors_origins: tuple[str, ...]
    rate_limit_enabled: bool
    rate_limit_per_min: int
    log_level: str
    output_dir: Path
    audit_dir: Path
    on_premise_audit_dir: Path

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> GatewayConfig:
        e = env if env is not None else os.environ
        cors = [
            origin.strip()
            for origin in e.get(
                "GATEWAY_CORS_ORIGINS", "http://localhost:3000,http://localhost:8080"
            ).split(",")
            if origin.strip()
        ]
        return cls(
            host=e.get("GATEWAY_HOST", "127.0.0.1"),
            port=int(e.get("GATEWAY_PORT", "8080")),
            doctor_username=e.get("GATEWAY_DOCTOR_USERNAME", "demo_doctor"),
            doctor_password_hash=e.get("GATEWAY_DOCTOR_PASSWORD_HASH", ""),
            session_secret=e.get("GATEWAY_SESSION_SECRET", ""),
            session_ttl_hours=int(e.get("GATEWAY_SESSION_TTL_HOURS", "8")),
            on_premise_api_url=e.get("ON_PREMISE_API_URL", "http://127.0.0.1:8001"),
            cloud_api_url=e.get("CLOUD_API_URL", "http://127.0.0.1:8002"),
            audit_retention_days=int(e.get("GATEWAY_AUDIT_RETENTION_DAYS", "730")),
            cors_origins=tuple(cors),
            rate_limit_enabled=e.get("GATEWAY_RATE_LIMIT_ENABLED", "true").lower() == "true",
            rate_limit_per_min=int(e.get("GATEWAY_RATE_LIMIT_PER_MIN", "60")),
            log_level=e.get("GATEWAY_LOG_LEVEL", "info"),
            output_dir=Path(e.get("GATEWAY_OUTPUT_DIR", "application/gateway/output")),
            audit_dir=Path(e.get("GATEWAY_AUDIT_DIR", "application/gateway/audit")),
            on_premise_audit_dir=Path(
                e.get("ON_PREMISE_AUDIT_DIR", "application/on_premise/audit")
            ),
        )
