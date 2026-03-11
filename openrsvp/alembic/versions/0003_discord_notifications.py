"""Add Discord notification support."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "0003_discord_notifications"
down_revision = "0002_rsvp_closures"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("events") as batch_op:
        batch_op.add_column(
            sa.Column(
                "discord_webhook_url",
                sa.String(512),
                nullable=True,
            )
        )

    op.create_table(
        "event_notifications",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "event_id",
            sa.String(36),
            sa.ForeignKey("events.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("notification_key", sa.String(64), nullable=False),
        sa.Column("sent_at", sa.DateTime(), nullable=False),
    )
    op.create_index(
        "ix_event_notifications_event_key",
        "event_notifications",
        ["event_id", "notification_key"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("ix_event_notifications_event_key", table_name="event_notifications")
    op.drop_table("event_notifications")
    with op.batch_alter_table("events") as batch_op:
        batch_op.drop_column("discord_webhook_url")
