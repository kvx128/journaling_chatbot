"""add mood model fields

Revision ID: 0002_add_mood_model_fields
Revises: 0001_initial_schema
Create Date: 2026-08-13 02:51:54.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '0002_add_mood_model_fields'
down_revision = '0001_initial_schema'
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("mood_entries") as batch_op:
        batch_op.add_column(sa.Column('valence', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('arousal', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('emotion_tags', sa.Text(), nullable=True))
        batch_op.create_check_constraint(
            "chk_valence_range",
            "valence IS NULL OR (valence >= -1.0 AND valence <= 1.0)"
        )
        batch_op.create_check_constraint(
            "chk_arousal_range",
            "arousal IS NULL OR (arousal >= -1.0 AND arousal <= 1.0)"
        )


def downgrade() -> None:
    with op.batch_alter_table("mood_entries") as batch_op:
        batch_op.drop_constraint("chk_arousal_range", type_="check")
        batch_op.drop_constraint("chk_valence_range", type_="check")
        batch_op.drop_column('emotion_tags')
        batch_op.drop_column('arousal')
        batch_op.drop_column('valence')
