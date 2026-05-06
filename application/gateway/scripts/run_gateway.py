"""Typer CLI: launch the gateway via uvicorn.

Reads host/port from `.env` (`GATEWAY_HOST`/`GATEWAY_PORT`) and forwards
the rest to uvicorn. Use `--reload` for dev iteration.

Examples:
    uv run python application/gateway/scripts/run_gateway.py
    uv run python application/gateway/scripts/run_gateway.py --reload
    uv run python application/gateway/scripts/run_gateway.py --port 8081
"""

from __future__ import annotations

import typer
import uvicorn
from dotenv import load_dotenv
from rich.console import Console

from gateway.config import GatewayConfig

app = typer.Typer(add_completion=False, no_args_is_help=False)
console = Console()


@app.command()
def run(
    host: str = typer.Option(None, "--host", help="Override GATEWAY_HOST."),
    port: int = typer.Option(None, "--port", help="Override GATEWAY_PORT."),
    reload: bool = typer.Option(False, "--reload", help="Enable uvicorn auto-reload."),
    log_level: str = typer.Option(None, "--log-level", help="Override GATEWAY_LOG_LEVEL."),
) -> None:
    """Launch the gateway. Health: `curl http://<host>:<port>/v1/health`."""
    load_dotenv()
    cfg = GatewayConfig.from_env()
    bind_host = host or cfg.host
    bind_port = port or cfg.port
    bind_log = (log_level or cfg.log_level).lower()

    if bind_host not in {"127.0.0.1", "localhost"}:
        console.print(
            f"[yellow]WARN[/]: binding to {bind_host} — front with TLS reverse proxy "
            "before exposing to non-loopback traffic"
        )

    console.print(
        f"[green]gateway[/] starting on http://{bind_host}:{bind_port} "
        f"(on-prem={cfg.on_premise_api_url}, cloud={cfg.cloud_api_url})"
    )
    uvicorn.run(
        "gateway.main:app",
        host=bind_host,
        port=bind_port,
        reload=reload,
        log_level=bind_log,
    )


if __name__ == "__main__":
    app()
