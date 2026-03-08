"""add subject column to actions

Revision ID: f7a8b9c0d1e2
Revises: e6f7a8b9c0d1
Create Date: 2026-03-06 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'f7a8b9c0d1e2'
down_revision: Union[str, None] = 'e6f7a8b9c0d1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('actions', sa.Column('subject', sa.String(20), nullable=True))
    op.execute("UPDATE actions SET subject = 'fencer' WHERE subject IS NULL")
    op.alter_column('actions', 'subject', nullable=False, server_default='fencer')


def downgrade() -> None:
    op.drop_column('actions', 'subject')
