"""APScheduler integration."""

from __future__ import annotations

from apscheduler.schedulers.background import BackgroundScheduler

from .config import settings
from .decay import run_decay_cycle, vacuum_database

_scheduler: BackgroundScheduler | None = None


def start_scheduler() -> BackgroundScheduler:
    global _scheduler
    if _scheduler and _scheduler.running:
        return _scheduler
    scheduler = BackgroundScheduler(timezone="UTC")
    scheduler.add_job(
        run_decay_cycle,
        "interval",
        hours=settings.decay_interval_hours,
        id="decay-cycle",
        max_instances=1,
        replace_existing=True,
    )
    scheduler.add_job(
        vacuum_database,
        "interval",
        hours=settings.sqlite_vacuum_hours,
        id="vacuum",
        max_instances=1,
        replace_existing=True,
    )
    scheduler.start()
    _scheduler = scheduler
    return scheduler


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        _scheduler = None
