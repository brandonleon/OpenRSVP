"""Add repeating event series support."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "0003_repeating_events"
down_revision = "0002_rsvp_closures"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "event_series",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("admin_token", sa.String(128), nullable=False, unique=True),
        sa.Column("recurrence_rule", sa.String(32), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    # Add column without FK constraint — SQLite doesn't enforce foreign keys
    # and batch mode requires named constraints which conflicts with SQLite.
    op.add_column("events", sa.Column("series_id", sa.String(36), nullable=True))


def downgrade() -> None:
    op.drop_column("events", "series_id")
    op.drop_table("event_series")
