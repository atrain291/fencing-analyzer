"""add blade speed columns to actions

Revision ID: d5e6f7a8b9c0
Revises: c4d5e6f7a8b9
Create Date: 2026-03-05 14:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'd5e6f7a8b9c0'
down_revision: Union[str, None] = 'c4d5e6f7a8b9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('actions', sa.Column('blade_speed_avg', sa.Float(), nullable=True))
    op.add_column('actions', sa.Column('blade_speed_peak', sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column('actions', 'blade_speed_peak')
    op.drop_column('actions', 'blade_speed_avg')
