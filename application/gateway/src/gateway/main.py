"""ASGI entrypoint for `uvicorn gateway.main:app`."""

from __future__ import annotations

from gateway.bootstrap import build_production_app

app = build_production_app()
