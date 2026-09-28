"""Add per-event guest allowance (max_guests)."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "0004_event_max_guests"
down_revision = "0003_repeating_events"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("DROP TABLE IF EXISTS _alembic_tmp_events")
    # server_default backfills existing events with the previous hardcoded limit (5),
    # so current behavior and stored RSVPs are unchanged.
    with op.batch_alter_table("events") as batch_op:
        batch_op.add_column(
            sa.Column(
                "max_guests",
                sa.Integer(),
                nullable=False,
                server_default=sa.text("5"),
            )
        )


def downgrade() -> None:
    with op.batch_alter_table("events") as batch_op:
        batch_op.drop_column("max_guests")
