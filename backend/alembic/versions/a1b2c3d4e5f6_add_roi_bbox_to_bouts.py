"""add roi bbox to bouts

Revision ID: a1b2c3d4e5f6
Revises: e2cf84d3406e
Create Date: 2026-03-02 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = 'e2cf84d3406e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('bouts', sa.Column('fencer_bbox', postgresql.JSON(), nullable=True))
    op.add_column('bouts', sa.Column('opponent_bbox', postgresql.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column('bouts', 'opponent_bbox')
    op.drop_column('bouts', 'fencer_bbox')
