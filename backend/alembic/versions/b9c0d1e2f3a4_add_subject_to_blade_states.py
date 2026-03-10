"""add subject column to blade_states and drop unique frame_id constraint

Revision ID: b9c0d1e2f3a4
Revises: a8b9c0d1e2f3
Create Date: 2026-03-10 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'b9c0d1e2f3a4'
down_revision: Union[str, None] = 'a8b9c0d1e2f3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add subject column (default "fencer" for existing rows)
    op.add_column('blade_states', sa.Column('subject', sa.String(20), server_default='fencer', nullable=False))

    # Drop unique constraint on frame_id if it exists (now one row per subject per frame)
    # The constraint name varies — try the most common patterns
    try:
        op.drop_constraint('blade_states_frame_id_key', 'blade_states', type_='unique')
    except Exception:
        pass  # constraint may not exist or have a different name


def downgrade() -> None:
    # Remove opponent rows before re-adding unique constraint
    op.execute("DELETE FROM blade_states WHERE subject != 'fencer'")
    op.create_unique_constraint('blade_states_frame_id_key', 'blade_states', ['frame_id'])
    op.drop_column('blade_states', 'subject')
