"""Health probe — checks downstream reachability."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request

from gateway.clients.cloud import CloudClient
from gateway.clients.on_premise import OnPremiseClient

router = APIRouter(tags=["health"])


def _get_on_premise(request: Request) -> OnPremiseClient | None:
    return getattr(request.app.state, "on_premise_client", None)


def _get_cloud(request: Request) -> CloudClient | None:
    return getattr(request.app.state, "cloud_client", None)


@router.get("/v1/health")
async def health(
    on_premise: Annotated[OnPremiseClient | None, Depends(_get_on_premise)],
    cloud: Annotated[CloudClient | None, Depends(_get_cloud)],
) -> dict[str, Any]:
    on_prem_ok = await on_premise.health() if on_premise is not None else False
    cloud_ok = await cloud.health() if cloud is not None else False
    overall = "ok" if (on_prem_ok and cloud_ok) else "degraded"
    return {
        "status": overall,
        "on_premise_reachable": on_prem_ok,
        "cloud_reachable": cloud_ok,
        "version": "0.1.0",
    }
