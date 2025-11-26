"""Typer CLI for OpenRSVP."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

import typer
import uvicorn

from .config import settings
from .decay import run_decay_cycle, vacuum_database
from .scheduler import start_scheduler, stop_scheduler
from .seed import seed_fake_data
from .storage import fetch_root_token, init_db, rotate_root_token

app = typer.Typer(help="OpenRSVP command-line interface")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Show help when no subcommand is provided."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


@app.command("admin-token")
def admin_token() -> None:
    """Print the current root admin token."""
    init_db()
    token = fetch_root_token()
    typer.echo(token)


@app.command("rotate-admin-token")
def rotate_admin_token() -> None:
    """Rotate the root admin token."""
    init_db()
    token = rotate_root_token()
    typer.echo(token)


@app.command("decay")
def decay(
    vacuum: bool = typer.Option(
        False,
        "--vacuum",
        help="Run SQLite VACUUM after decay completes",
    ),
) -> None:
    """Run the decay cycle manually."""
    init_db()
    stats = run_decay_cycle()
    typer.echo(f"Decay complete: {stats}")
    if vacuum:
        vacuum_database()
        typer.echo("Database vacuum complete.")


@app.command("runserver")
def runserver(
    host: str = typer.Option(settings.app_host, "--host", help="Host to bind"),
    port: int = typer.Option(settings.app_port, "--port", help="Port to bind"),
):
    """Start FastAPI with APScheduler."""
    init_db()
    start_scheduler()
    config = uvicorn.Config("openrsvp.api:app", host=host, port=port, reload=False)
    server = uvicorn.Server(config)
    try:
        typer.echo(f"Starting OpenRSVP on {host}:{port}")
        server.run()
    finally:
        stop_scheduler()


@app.command("seed-data")
def seed_data(
    channels: int = typer.Option(
        5, "--channels", min=0, help="Number of channels to create"
    ),
    max_events: int = typer.Option(
        3, "--max-events", min=1, help="Maximum events to assign to each channel"
    ),
    extra_events: int = typer.Option(
        0, "--extra-events", min=0, help="Number of events without channels"
    ),
    max_rsvps: int = typer.Option(
        5, "--max-rsvps", min=0, help="Maximum RSVPs to attach to each event"
    ),
    private_percent: int = typer.Option(
        10,
        "--private-percent",
        min=0,
        max=100,
        help="Percentage of events that should be private (0-100)",
    ),
    private_rsvp_percent: int = typer.Option(
        20,
        "--private-rsvp-percent",
        min=0,
        max=100,
        help="Percentage of RSVPs that should be private (0-100)",
    ),
):
    """Populate the database with fake channels and events for testing."""
    stats = seed_fake_data(
        channel_count=channels,
        max_events_per_channel=max_events,
        extra_events=extra_events,
        max_rsvps_per_event=max_rsvps,
        private_percentage=private_percent,
        private_rsvp_percentage=private_rsvp_percent,
    )
    typer.echo(
        f"Seed complete: {stats['channels']} channels, {stats['events']} events, "
        f"{stats['rsvps']} RSVPs created."
    )


@app.command("test")
def run_tests(pytest_args: list[str] = typer.Argument(None, help="Extra pytest args")):
    """Run the test suite with helpful defaults."""
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", ".")
    env.setdefault("UV_CACHE_DIR", ".uv-cache")
    uv_path = shutil.which("uv")
    if uv_path:
        cmd = [uv_path, "run", "pytest"]
    else:
        typer.echo("uv not found, falling back to python -m pytest")
        cmd = [sys.executable, "-m", "pytest"]
    if pytest_args:
        cmd.extend(pytest_args)
    typer.echo(f"Running tests: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env)
    raise typer.Exit(code=result.returncode)


if __name__ == "__main__":
    app()
