"""add journal entry mood fields

Revision ID: 0003_add_journal_mood_fields
Revises: 0002_add_mood_model_fields
Create Date: 2026-08-14 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '0003_add_journal_mood_fields'
down_revision = '0002_add_mood_model_fields'
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("journal_entries") as batch_op:
        batch_op.add_column(sa.Column('valence', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('arousal', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('emotion_tags', sa.Text(), nullable=True))
        batch_op.create_check_constraint(
            "chk_journal_valence_range",
            "valence IS NULL OR (valence >= -1.0 AND valence <= 1.0)"
        )
        batch_op.create_check_constraint(
            "chk_journal_arousal_range",
            "arousal IS NULL OR (arousal >= -1.0 AND arousal <= 1.0)"
        )
        batch_op.create_index(
            "ix_journal_entries_user_created", ["user_id", "created_at"]
        )


def downgrade() -> None:
    with op.batch_alter_table("journal_entries") as batch_op:
        batch_op.drop_index("ix_journal_entries_user_created")
        batch_op.drop_constraint("chk_journal_arousal_range", type_="check")
        batch_op.drop_constraint("chk_journal_valence_range", type_="check")
        batch_op.drop_column('emotion_tags')
        batch_op.drop_column('arousal')
        batch_op.drop_column('valence')
