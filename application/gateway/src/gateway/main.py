"""ASGI entrypoint for `uvicorn gateway.main:app`."""

from __future__ import annotations

from gateway.api.app import build_app

app = build_app()
