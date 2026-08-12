"""initial schema

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-08-12 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "0001_initial_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. users
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("handle", sa.String(length=64), nullable=False),
        sa.Column("tz", sa.String(length=64), server_default="Asia/Kolkata", nullable=False),
        sa.Column("currency", sa.String(length=10), server_default="INR", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_users")),
    )
    op.create_index(op.f("ix_users_handle"), "users", ["handle"], unique=True)

    # 2. transactions
    op.create_table(
        "transactions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("amount_minor", sa.Integer(), nullable=False),
        sa.Column("direction", sa.String(length=20), server_default="debit", nullable=False),
        sa.Column("category", sa.String(length=50), server_default="OTHER", nullable=False),
        sa.Column("merchant", sa.String(length=255), nullable=True),
        sa.Column("occurred_on", sa.Date(), nullable=False),
        sa.Column("payment_method", sa.String(length=64), nullable=True),
        sa.Column("source", sa.String(length=20), server_default="chat", nullable=False),
        sa.Column("raw_text", sa.Text(), nullable=True),
        sa.Column("extraction_confidence", sa.Float(), nullable=True),
        sa.Column("confirmed", sa.Boolean(), server_default="1", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name=op.f("fk_transactions_user_id_users"), ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_transactions")),
    )
    op.create_index(
        "ix_transactions_user_occurred",
        "transactions",
        ["user_id", "occurred_on"],
        unique=False,
    )

    # 3. budgets
    op.create_table(
        "budgets",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("category", sa.String(length=50), nullable=False),
        sa.Column("period", sa.String(length=20), server_default="monthly", nullable=False),
        sa.Column("limit_minor", sa.Integer(), nullable=False),
        sa.Column("active_from", sa.Date(), nullable=False),
        sa.Column("active_to", sa.Date(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name=op.f("fk_budgets_user_id_users"), ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_budgets")),
    )

    # 4. mood_entries
    op.create_table(
        "mood_entries",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("self_report", sa.Integer(), nullable=False),
        sa.Column("sleep_hours", sa.Float(), nullable=True),
        sa.Column("energy", sa.Integer(), nullable=True),
        sa.Column("social_contact", sa.Boolean(), nullable=True),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("recorded_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.CheckConstraint("self_report >= 1 AND self_report <= 5", name=op.f("ck_mood_entries_chk_self_report_range")),
        sa.CheckConstraint("energy IS NULL OR (energy >= 1 AND energy <= 5)", name=op.f("ck_mood_entries_chk_energy_range")),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name=op.f("fk_mood_entries_user_id_users"), ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_mood_entries")),
    )
    op.create_index(
        "ix_mood_entries_user_recorded",
        "mood_entries",
        ["user_id", "recorded_at"],
        unique=False,
    )

    # 5. journal_entries
    op.create_table(
        "journal_entries",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name=op.f("fk_journal_entries_user_id_users"), ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_journal_entries")),
    )


def downgrade() -> None:
    op.drop_table("journal_entries")
    op.drop_index("ix_mood_entries_user_recorded", table_name="mood_entries")
    op.drop_table("mood_entries")
    op.drop_table("budgets")
    op.drop_index("ix_transactions_user_occurred", table_name="transactions")
    op.drop_table("transactions")
    op.drop_index(op.f("ix_users_handle"), table_name="users")
    op.drop_table("users")
