"""Global configuration for OpenRSVP."""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    base_dir: Path = Path(os.getenv("OPENRSVP_BASE_DIR", Path.cwd()))
    data_dir: Path = Path(os.getenv("OPENRSVP_DATA_DIR", base_dir / "data"))
    database_path: Path = Path(os.getenv("OPENRSVP_DB", data_dir / "openrsvp.db"))
    decay_factor: float = float(os.getenv("OPENRSVP_DECAY_FACTOR", 0.92))
    decay_interval_hours: int = int(os.getenv("OPENRSVP_DECAY_INTERVAL_HOURS", 1))
    hide_threshold: float = float(os.getenv("OPENRSVP_HIDE_THRESHOLD", 10))
    delete_threshold: float = float(os.getenv("OPENRSVP_DELETE_THRESHOLD", 2))
    hide_after_days: int = int(os.getenv("OPENRSVP_HIDE_AFTER_DAYS", 7))
    delete_after_days: int = int(os.getenv("OPENRSVP_DELETE_AFTER_DAYS", 30))
    sqlite_vacuum_hours: int = int(os.getenv("OPENRSVP_VACUUM_HOURS", 12))
    root_token_key: str = "root_admin_token"
    app_host: str = os.getenv("OPENRSVP_HOST", "0.0.0.0")
    app_port: int = int(os.getenv("OPENRSVP_PORT", 8000))

    @property
    def decay_interval(self) -> timedelta:
        return timedelta(hours=self.decay_interval_hours)

    @property
    def vacuum_interval(self) -> timedelta:
        return timedelta(hours=self.sqlite_vacuum_hours)


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
