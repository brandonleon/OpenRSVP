"""Global configuration for OpenRSVP."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

DEFAULTS: dict[str, Any] = {
    "decay_factor": 0.92,
    "decay_interval_hours": 1,
    "hide_threshold": 10.0,
    "delete_threshold": 2.0,
    "hide_after_days": 7,
    "delete_after_days": 30,
    "delete_grace_days_after_end": 30,
    "protect_upcoming_events": True,
    "sqlite_vacuum_hours": 12,
    "events_per_page": 10,
    "admin_events_per_page": 25,
    "seed_channels": 5,
    "seed_events_per_channel": 3,
    "seed_extra_events": 0,
    "seed_rsvps_per_event": 5,
    "seed_private_percent": 10,
    "seed_private_rsvp_percent": 20,
    "enable_scheduler": True,
    "initial_event_score": 100.0,
    "initial_channel_score": 100.0,
    "app_host": "0.0.0.0",
    "app_port": 8000,
}

TYPE_CASTERS: dict[str, Callable[[Any], Any]] = {
    "decay_factor": float,
    "decay_interval_hours": int,
    "hide_threshold": float,
    "delete_threshold": float,
    "hide_after_days": int,
    "delete_after_days": int,
    "delete_grace_days_after_end": int,
    "protect_upcoming_events": bool,
    "sqlite_vacuum_hours": int,
    "events_per_page": int,
    "admin_events_per_page": int,
    "seed_channels": int,
    "seed_events_per_channel": int,
    "seed_extra_events": int,
    "seed_rsvps_per_event": int,
    "seed_private_percent": int,
    "seed_private_rsvp_percent": int,
    "enable_scheduler": bool,
    "initial_event_score": float,
    "initial_channel_score": float,
    "app_host": str,
    "app_port": int,
}


@dataclass(frozen=True)
class Settings:
    base_dir: Path
    data_dir: Path
    database_path: Path
    decay_factor: float
    decay_interval_hours: int
    hide_threshold: float
    delete_threshold: float
    hide_after_days: int
    delete_after_days: int
    delete_grace_days_after_end: int
    protect_upcoming_events: bool
    sqlite_vacuum_hours: int
    events_per_page: int
    admin_events_per_page: int
    seed_channels: int
    seed_events_per_channel: int
    seed_extra_events: int
    seed_rsvps_per_event: int
    seed_private_percent: int
    seed_private_rsvp_percent: int
    enable_scheduler: bool
    initial_event_score: float
    initial_channel_score: float
    root_token_key: str
    app_host: str
    app_port: int
    config_path: Path

    @property
    def decay_interval(self) -> timedelta:
        return timedelta(hours=self.decay_interval_hours)

    @property
    def vacuum_interval(self) -> timedelta:
        return timedelta(hours=self.sqlite_vacuum_hours)


def _boolify(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"Cannot parse boolean value from {value!r}")


def _cast_value(key: str, value: Any) -> Any:
    if key not in TYPE_CASTERS:
        return value
    caster = TYPE_CASTERS[key]
    if caster is bool:
        return _boolify(value)
    return caster(value)


def _load_toml_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {}
    with config_path.open("rb") as handle:
        return tomllib.load(handle)


def _config_layered_value(
    key: str, *, toml_config: dict[str, Any], base_dir: Path
) -> Any:
    env_key = f"OPENRSVP_{key.upper()}"
    if env_key in os.environ:
        return _cast_value(key, os.environ[env_key])
    if key in toml_config:
        return _cast_value(key, toml_config[key])
    return DEFAULTS[key]


def _resolve_paths(
    *,
    base_dir: Path,
    data_dir: str | Path | None,
    database_path: str | Path | None,
):
    resolved_base = Path(base_dir)
    resolved_data = Path(data_dir) if data_dir else resolved_base / "data"
    if not resolved_data.is_absolute():
        resolved_data = resolved_base / resolved_data
    resolved_db = Path(database_path) if database_path else resolved_data / "openrsvp.db"
    if not resolved_db.is_absolute():
        resolved_db = resolved_base / resolved_db
    return resolved_base, resolved_data, resolved_db


def load_settings(config_override: Path | None = None) -> Settings:
    base_dir = Path(os.getenv("OPENRSVP_BASE_DIR", Path.cwd()))
    env_config = os.getenv("OPENRSVP_CONFIG")
    config_path = Path(config_override or env_config or base_dir / "openrsvp.toml")
    toml_config = _load_toml_config(config_path)

    base_dir_value, data_dir_value, database_path_value = _resolve_paths(
        base_dir=base_dir,
        data_dir=os.getenv("OPENRSVP_DATA_DIR", toml_config.get("data_dir")),
        database_path=os.getenv("OPENRSVP_DB", toml_config.get("database_path")),
    )

    settings = Settings(
        base_dir=base_dir_value,
        data_dir=data_dir_value,
        database_path=database_path_value,
        decay_factor=_config_layered_value(
            "decay_factor", toml_config=toml_config, base_dir=base_dir_value
        ),
        decay_interval_hours=_config_layered_value(
            "decay_interval_hours", toml_config=toml_config, base_dir=base_dir_value
        ),
        hide_threshold=_config_layered_value(
            "hide_threshold", toml_config=toml_config, base_dir=base_dir_value
        ),
        delete_threshold=_config_layered_value(
            "delete_threshold", toml_config=toml_config, base_dir=base_dir_value
        ),
        hide_after_days=_config_layered_value(
            "hide_after_days", toml_config=toml_config, base_dir=base_dir_value
        ),
        delete_after_days=_config_layered_value(
            "delete_after_days", toml_config=toml_config, base_dir=base_dir_value
        ),
        delete_grace_days_after_end=_config_layered_value(
            "delete_grace_days_after_end",
            toml_config=toml_config,
            base_dir=base_dir_value,
        ),
        protect_upcoming_events=_config_layered_value(
            "protect_upcoming_events",
            toml_config=toml_config,
            base_dir=base_dir_value,
        ),
        sqlite_vacuum_hours=_config_layered_value(
            "sqlite_vacuum_hours", toml_config=toml_config, base_dir=base_dir_value
        ),
        events_per_page=_config_layered_value(
            "events_per_page", toml_config=toml_config, base_dir=base_dir_value
        ),
        admin_events_per_page=_config_layered_value(
            "admin_events_per_page", toml_config=toml_config, base_dir=base_dir_value
        ),
        seed_channels=_config_layered_value(
            "seed_channels", toml_config=toml_config, base_dir=base_dir_value
        ),
        seed_events_per_channel=_config_layered_value(
            "seed_events_per_channel", toml_config=toml_config, base_dir=base_dir_value
        ),
        seed_extra_events=_config_layered_value(
            "seed_extra_events", toml_config=toml_config, base_dir=base_dir_value
        ),
        seed_rsvps_per_event=_config_layered_value(
            "seed_rsvps_per_event", toml_config=toml_config, base_dir=base_dir_value
        ),
        seed_private_percent=_config_layered_value(
            "seed_private_percent", toml_config=toml_config, base_dir=base_dir_value
        ),
        seed_private_rsvp_percent=_config_layered_value(
            "seed_private_rsvp_percent",
            toml_config=toml_config,
            base_dir=base_dir_value,
        ),
        enable_scheduler=_config_layered_value(
            "enable_scheduler", toml_config=toml_config, base_dir=base_dir_value
        ),
        initial_event_score=_config_layered_value(
            "initial_event_score", toml_config=toml_config, base_dir=base_dir_value
        ),
        initial_channel_score=_config_layered_value(
            "initial_channel_score", toml_config=toml_config, base_dir=base_dir_value
        ),
        root_token_key="root_admin_token",
        app_host=_config_layered_value(
            "app_host", toml_config=toml_config, base_dir=base_dir_value
        ),
        app_port=_config_layered_value(
            "app_port", toml_config=toml_config, base_dir=base_dir_value
        ),
        config_path=config_path,
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings


def settings_as_dict(settings: Settings) -> dict[str, Any]:
    return {
        "base_dir": str(settings.base_dir),
        "data_dir": str(settings.data_dir),
        "database_path": str(settings.database_path),
        "decay_factor": settings.decay_factor,
        "decay_interval_hours": settings.decay_interval_hours,
        "hide_threshold": settings.hide_threshold,
        "delete_threshold": settings.delete_threshold,
        "hide_after_days": settings.hide_after_days,
        "delete_after_days": settings.delete_after_days,
        "delete_grace_days_after_end": settings.delete_grace_days_after_end,
        "protect_upcoming_events": settings.protect_upcoming_events,
        "sqlite_vacuum_hours": settings.sqlite_vacuum_hours,
        "events_per_page": settings.events_per_page,
        "admin_events_per_page": settings.admin_events_per_page,
        "seed_channels": settings.seed_channels,
        "seed_events_per_channel": settings.seed_events_per_channel,
        "seed_extra_events": settings.seed_extra_events,
        "seed_rsvps_per_event": settings.seed_rsvps_per_event,
        "seed_private_percent": settings.seed_private_percent,
        "seed_private_rsvp_percent": settings.seed_private_rsvp_percent,
        "enable_scheduler": settings.enable_scheduler,
        "initial_event_score": settings.initial_event_score,
        "initial_channel_score": settings.initial_channel_score,
        "app_host": settings.app_host,
        "app_port": settings.app_port,
    }


def _toml_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def write_config_file(config: dict[str, Any], *, path: Path) -> None:
    lines = ["# OpenRSVP configuration\n"]
    for key in sorted(config.keys()):
        lines.append(f"{key} = {_toml_literal(config[key])}\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")


def update_config_file(updates: dict[str, Any], *, path: Path | None = None) -> Settings:
    current_settings = settings if "settings" in globals() else load_settings()
    target_path = path or current_settings.config_path
    existing = _load_toml_config(target_path)
    merged = {**existing}
    for key, value in updates.items():
        if key not in DEFAULTS:
            continue
        merged[key] = _cast_value(key, value)
    write_config_file(merged, path=target_path)
    new_settings = load_settings(target_path)
    globals()["settings"] = new_settings
    return new_settings


settings = load_settings()
