"""add unique constraint to fencer name

Revision ID: e6f7a8b9c0d1
Revises: d5e6f7a8b9c0
Create Date: 2026-03-05 16:00:00.000000

"""
from typing import Sequence, Union

from alembic import op


revision: str = 'e6f7a8b9c0d1'
down_revision: Union[str, None] = 'd5e6f7a8b9c0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_unique_constraint('uq_fencers_name', 'fencers', ['name'])


def downgrade() -> None:
    op.drop_constraint('uq_fencers_name', 'fencers', type_='unique')
