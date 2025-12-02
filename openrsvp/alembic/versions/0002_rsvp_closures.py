"""Add RSVP close scheduling controls."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "0002_rsvp_closures"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("DROP TABLE IF EXISTS _alembic_tmp_events")
    with op.batch_alter_table("events") as batch_op:
        batch_op.add_column(
            sa.Column(
                "rsvps_closed",
                sa.Boolean(),
                nullable=False,
                server_default=sa.text("0"),
            )
        )
        batch_op.add_column(
            sa.Column(
                "rsvp_close_at",
                sa.DateTime(),
                nullable=True,
            )
        )


def downgrade() -> None:
    with op.batch_alter_table("events") as batch_op:
        batch_op.drop_column("rsvp_close_at")
        batch_op.drop_column("rsvps_closed")
