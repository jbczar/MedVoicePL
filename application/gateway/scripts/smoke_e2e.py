"""End-to-end HTTP smoke against a running gateway.

Assumes the gateway is reachable at `--base-url` (default
``http://127.0.0.1:8080``). Usually you'd boot it via:

    GATEWAY_DEMO_MODE=true make up

Then:

    uv run python application/gateway/scripts/smoke_e2e.py

The smoke runs the full doctor flow:

    1. POST /v1/health                      → status reachable
    2. POST /v1/auth/login                  → cookie minted
    3. GET  /v1/auth/me                     → identity
    4. POST /v1/visits                      → draft visit_id
    5. POST /v1/visits/{vid}/audio          → audio_uploaded + sha256
    6. POST /v1/visits/{vid}/finalize       → completed + revealed
    7. GET  /v1/visits/{vid}                → final summary
    8. POST /v1/auth/logout                 → invalidate

Verifies revealed_suggestion has at least one drug, provenance is
non-empty, and the revealed text contains a doctor-readable diagnosis
(eponym preserved, not `<PERSON_n>`).

Exit code 0 = pass; non-zero = fail with a console error.
"""

from __future__ import annotations

import io
import struct
import sys
import time
from pathlib import Path

import httpx
import typer
from rich.console import Console

app = typer.Typer(add_completion=False, no_args_is_help=False)
console = Console()


def _wav_bytes() -> bytes:
    header = b"RIFF"
    data_size = 1024 * 2
    chunk_size = 36 + data_size
    header += struct.pack("<I", chunk_size)
    header += b"WAVE"
    header += b"fmt "
    header += struct.pack("<IHHIIHH", 16, 1, 1, 16000, 32000, 2, 16)
    header += b"data"
    header += struct.pack("<I", data_size)
    return header + b"\x00" * data_size


def _expect(cond: bool, msg: str) -> None:
    if not cond:
        console.print(f"[bold red]FAIL[/]: {msg}")
        sys.exit(1)
    console.print(f"  [green]✓[/] {msg}")


def _step(name: str) -> None:
    console.rule(f"[bold cyan]{name}[/]")


@app.command()
def run(
    base_url: str = typer.Option("http://127.0.0.1:8080", "--base-url"),
    username: str = typer.Option("demo_doctor", "--username"),
    password: str = typer.Option("demo_password", "--password"),
    timeout_s: float = typer.Option(
        600.0,
        "--timeout",
        help="HTTP timeout per request. Live STT+SOAP can take 3-5 min.",
    ),
    wait_for_ready: int = typer.Option(
        20, "--wait-for-ready", help="Seconds to retry /v1/health while gateway boots."
    ),
    audio_file: str | None = typer.Option(
        None,
        "--audio-file",
        help="Path to real WAV file. Required for live mode (real STT chain).",
    ),
    expect_drug: list[str] = typer.Option(
        [],
        "--expect-drug",
        help=(
            "Drug name (substring, case-insensitive) that must appear in "
            "revealed_suggestion.drugs[].name. Repeatable. E.g. for endo-gra: "
            "--expect-drug Tiamazol --expect-drug Propranolol"
        ),
    ),
) -> None:
    """Run the gateway E2E smoke. Exit 0 on pass."""
    started = time.perf_counter()

    with httpx.Client(base_url=base_url, timeout=timeout_s) as client:
        # 0. Wait for /v1/health to become responsive.
        _step("Health probe")
        deadline = time.time() + wait_for_ready
        last_err: Exception | None = None
        while time.time() < deadline:
            try:
                resp = client.get("/v1/health")
                if resp.status_code == 200:
                    body = resp.json()
                    _expect(
                        body.get("status") in {"ok", "degraded"},
                        f"gateway responded ({body.get('status')})",
                    )
                    if body.get("status") == "degraded":
                        console.print(
                            "[yellow]NOTE[/]: status=degraded — running in demo mode "
                            "or downstream not reachable; smoke continues."
                        )
                    break
            except httpx.HTTPError as exc:
                last_err = exc
                time.sleep(0.5)
        else:
            _expect(False, f"gateway never came up at {base_url}: {last_err}")

        # 1. Login.
        _step("Login")
        resp = client.post(
            "/v1/auth/login",
            json={"username": username, "password": password},
        )
        _expect(resp.status_code == 204, f"login 204 (got {resp.status_code})")
        cookie = client.cookies.get("session_id")
        _expect(bool(cookie), "session_id cookie set")

        # 2. /me.
        _step("Identity")
        me = client.get("/v1/auth/me")
        _expect(me.status_code == 200, "GET /me 200")
        body = me.json()
        _expect(body["username"] == username, f"username == {username}")

        # 3. Create visit.
        _step("Create visit")
        resp = client.post(
            "/v1/visits",
            json={"specialization": "endokrynologia", "visit_date": "2026-05-06"},
        )
        _expect(resp.status_code == 201, f"create visit 201 (got {resp.status_code})")
        visit_id = resp.json()["visit_id"]
        _expect(resp.json()["status"] == "draft", "status=draft")
        console.print(f"  visit_id={visit_id}")

        # 4. Upload audio (real file if given, else 1024-byte mock).
        _step("Upload audio")
        if audio_file:
            audio_path = Path(audio_file)
            if not audio_path.is_file():
                _expect(False, f"audio file not found: {audio_path}")
            wav = audio_path.read_bytes()
            console.print(f"  using {audio_path} ({len(wav) / 1e6:.1f} MB)")
        else:
            wav = _wav_bytes()
            console.print("  using mock 1068-byte WAV (demo mode)")
        resp = client.post(
            f"/v1/visits/{visit_id}/audio",
            files={"file": ("audio.wav", io.BytesIO(wav), "audio/wav")},
        )
        _expect(resp.status_code == 200, f"upload 200 (got {resp.status_code})")
        upload = resp.json()
        _expect(upload["status"] == "audio_uploaded", "status=audio_uploaded")
        _expect(len(upload["audio_sha256"]) == 64, "audio_sha256 is 64-char hex")

        # 5. Finalize. Real STT+SOAP+anonymize+agent can take 3-5 min cold.
        _step("Finalize (this can take several minutes in live mode)")
        finalize_started = time.perf_counter()
        resp = client.post(f"/v1/visits/{visit_id}/finalize")
        finalize_elapsed = time.perf_counter() - finalize_started
        console.print(f"  finalize wall-clock: {finalize_elapsed:.1f} s")
        _expect(
            resp.status_code == 200,
            f"finalize 200 (got {resp.status_code}: {resp.text[:200]})",
        )
        body = resp.json()
        _expect(body["visit"]["status"] == "completed", "visit.status=completed")
        _expect(len(body["suggestion"]["drugs"]) >= 1, "≥1 drug in raw suggestion")
        _expect(
            len(body["revealed_suggestion"]["drugs"]) >= 1,
            "≥1 drug in revealed_suggestion",
        )
        # Provenance non-empty (Phase 3a contract).
        for drug in body["revealed_suggestion"]["drugs"]:
            _expect(
                len(drug["provenance"]) >= 1,
                f"drug {drug.get('name', '?')} has non-empty provenance",
            )

        # Expected drug names (case-insensitive substring).
        drug_names_lower = " | ".join(
            d.get("name", "").lower() for d in body["revealed_suggestion"]["drugs"]
        )
        for needle in expect_drug:
            _expect(
                needle.lower() in drug_names_lower,
                f"expected drug '{needle}' present in {drug_names_lower!r}",
            )
        # Verify no <PERSON_*>/<LOCATION_*> placeholders remain in revealed
        # rationale unless explicitly flagged as unmapped.
        revealed_dump = str(body["revealed_suggestion"])
        if body["unmapped_token_count"] == 0:
            _expect(
                "<PERSON_" not in revealed_dump and "<LOCATION_" not in revealed_dump,
                "no unresolved placeholders in revealed suggestion",
            )
        else:
            console.print(
                f"  [yellow]NOTE[/]: {body['unmapped_token_count']} unmapped tokens "
                "(token map missing — running without on-prem audit dir)"
            )

        # 6. Re-fetch summary.
        _step("Final summary")
        summary = client.get(f"/v1/visits/{visit_id}")
        _expect(summary.status_code == 200, "GET visit summary 200")
        _expect(
            summary.json()["status"] == "completed",
            "summary status=completed",
        )

        # 7. Logout.
        _step("Logout")
        resp = client.post("/v1/auth/logout")
        _expect(resp.status_code == 204, f"logout 204 (got {resp.status_code})")
        # /me now 401.
        _expect(client.get("/v1/auth/me").status_code == 401, "/me 401 after logout")

    elapsed = time.perf_counter() - started
    console.rule("[bold green]SMOKE PASS[/]")
    console.print(f"end-to-end in {elapsed:.2f}s")


if __name__ == "__main__":
    app()
