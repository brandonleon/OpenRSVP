"""Typer CLI for OpenRSVP."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from sqlalchemy.exc import OperationalError
import typer
import uvicorn

from .config import (
    load_settings,
    settings,
    settings_as_dict,
    update_config_file,
)
from .decay import run_decay_cycle, vacuum_database
from .scheduler import start_scheduler, stop_scheduler
from .seed import seed_fake_data
from .storage import (
    ensure_root_token,
    fetch_root_token,
    init_db,
    rotate_root_token,
    upgrade_database,
)

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
    try:
        init_db()
        token = rotate_root_token()
    except OperationalError as exc:
        message = str(getattr(exc, "orig", exc)).lower()
        if "readonly" in message or "read-only" in message:
            typer.secho(
                "Unable to rotate the root admin token because the database is read-only. "
                f"Ensure the process can write to {settings.database_path} "
                "(run with sudo or adjust file ownership/permissions).",
                err=True,
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)
        raise
    typer.echo(token)


@app.command("upgrade-db")
def upgrade_db(
    no_backup: bool = typer.Option(
        False,
        "--no-backup",
        help="Skip creating a .bak copy of the database before upgrading",
    ),
) -> None:
    """Upgrade the SQLite database schema if needed."""
    try:
        actions = upgrade_database(make_backup=not no_backup)
    except OperationalError as exc:
        message = str(getattr(exc, "orig", exc)).lower()
        if "readonly" in message or "read-only" in message:
            typer.secho(
                "Unable to upgrade because the database is read-only. "
                f"Ensure write access to {settings.database_path}.",
                err=True,
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)
        raise

    if not actions:
        typer.echo("Database already up to date.")
        return

    ensure_root_token()
    typer.echo("Database upgrade complete:")
    for action in actions:
        typer.echo(f"- {action}")


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
        settings.seed_channels, "--channels", min=0, help="Number of channels to create"
    ),
    max_events: int = typer.Option(
        settings.seed_events_per_channel,
        "--max-events",
        min=1,
        help="Maximum events to assign to each channel",
    ),
    extra_events: int = typer.Option(
        settings.seed_extra_events,
        "--extra-events",
        min=0,
        help="Number of events without channels",
    ),
    max_rsvps: int = typer.Option(
        settings.seed_rsvps_per_event,
        "--max-rsvps",
        min=0,
        help="Maximum RSVPs to attach to each event",
    ),
    private_percent: int = typer.Option(
        settings.seed_private_percent,
        "--private-percent",
        min=0,
        max=100,
        help="Percentage of events that should be private (0-100)",
    ),
    private_rsvp_percent: int = typer.Option(
        settings.seed_private_rsvp_percent,
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


@app.command("config")
def configure(
    show: bool = typer.Option(
        False, "--show", help="Show the current effective configuration"
    ),
    decay_factor: float | None = typer.Option(
        None,
        "--decay-factor",
        min=0.0,
        max=1.0,
        help="Decay factor applied per day (0-1)",
    ),
    decay_interval_hours: int | None = typer.Option(
        None, "--decay-interval-hours", min=1, help="Hours between decay runs"
    ),
    hide_threshold: float | None = typer.Option(
        None,
        "--hide-threshold",
        help="Score below which events hide from listings",
    ),
    delete_threshold: float | None = typer.Option(
        None, "--delete-threshold", help="Score at/below which events can be deleted"
    ),
    hide_after_days: int | None = typer.Option(
        None, "--hide-after-days", help="Always show events newer than this many days"
    ),
    delete_after_days: int | None = typer.Option(
        None,
        "--delete-after-days",
        help="Minimum age (days since creation) before deletion is allowed",
    ),
    delete_grace_days_after_end: int | None = typer.Option(
        None,
        "--delete-grace-days-after-end",
        help="Days after event end before deletion is allowed",
    ),
    protect_upcoming_events: bool | None = typer.Option(
        None,
        "--protect-upcoming-events/--no-protect-upcoming-events",
        help="Skip deletion for events that have not started",
    ),
    vacuum_hours: int | None = typer.Option(
        None, "--vacuum-hours", help="Hours between SQLite VACUUM runs"
    ),
    host: str | None = typer.Option(None, "--host", help="Default host for runserver"),
    port: int | None = typer.Option(None, "--port", help="Default port for runserver"),
    config_path: Path | None = typer.Option(
        None, "--config-path", help="Path to openrsvp.toml (default: ./openrsvp.toml)"
    ),
    events_per_page: int | None = typer.Option(
        None, "--events-per-page", min=1, help="Public pagination size"
    ),
    admin_events_per_page: int | None = typer.Option(
        None, "--admin-events-per-page", min=1, help="Admin pagination size"
    ),
    seed_channels: int | None = typer.Option(
        None, "--seed-channels", min=0, help="Default seed-data channels"
    ),
    seed_events_per_channel: int | None = typer.Option(
        None, "--seed-events-per-channel", min=0, help="Default seed-data events/channel"
    ),
    seed_extra_events: int | None = typer.Option(
        None, "--seed-extra-events", min=0, help="Default seed-data extra events"
    ),
    seed_rsvps_per_event: int | None = typer.Option(
        None, "--seed-rsvps-per-event", min=0, help="Default seed-data RSVPs per event"
    ),
    seed_private_percent: int | None = typer.Option(
        None,
        "--seed-private-percent",
        min=0,
        max=100,
        help="Default percent of private events for seed-data",
    ),
    seed_private_rsvp_percent: int | None = typer.Option(
        None,
        "--seed-private-rsvp-percent",
        min=0,
        max=100,
        help="Default percent of private RSVPs for seed-data",
    ),
    enable_scheduler: bool | None = typer.Option(
        None,
        "--enable-scheduler/--disable-scheduler",
        help="Toggle background scheduler (decay/vacuum)",
    ),
    initial_event_score: float | None = typer.Option(
        None,
        "--initial-event-score",
        help="Starting score for new events",
    ),
    initial_channel_score: float | None = typer.Option(
        None,
        "--initial-channel-score",
        help="Starting score for new channels",
    ),
):
    """View or update the persistent configuration file."""

    updates = {
        "decay_factor": decay_factor,
        "decay_interval_hours": decay_interval_hours,
        "hide_threshold": hide_threshold,
        "delete_threshold": delete_threshold,
        "hide_after_days": hide_after_days,
        "delete_after_days": delete_after_days,
        "delete_grace_days_after_end": delete_grace_days_after_end,
        "protect_upcoming_events": protect_upcoming_events,
        "sqlite_vacuum_hours": vacuum_hours,
        "app_host": host,
        "app_port": port,
        "events_per_page": events_per_page,
        "admin_events_per_page": admin_events_per_page,
        "seed_channels": seed_channels,
        "seed_events_per_channel": seed_events_per_channel,
        "seed_extra_events": seed_extra_events,
        "seed_rsvps_per_event": seed_rsvps_per_event,
        "seed_private_percent": seed_private_percent,
        "seed_private_rsvp_percent": seed_private_rsvp_percent,
        "enable_scheduler": enable_scheduler,
        "initial_event_score": initial_event_score,
        "initial_channel_score": initial_channel_score,
    }
    clean_updates = {k: v for k, v in updates.items() if v is not None}

    settings_ref = settings
    target_path = config_path or settings.config_path
    if clean_updates:
        settings_ref = update_config_file(clean_updates, path=target_path)
        typer.echo(f"Updated configuration in {target_path}")
    else:
        settings_ref = load_settings(target_path)
    if show or not clean_updates:
        effective = settings_as_dict(settings_ref)
        effective["config_path"] = str(target_path)
        typer.echo(json.dumps(effective, indent=2))


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
