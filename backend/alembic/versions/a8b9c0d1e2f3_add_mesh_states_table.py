"""add mesh_states table for WHAM 3D reconstruction

Revision ID: a8b9c0d1e2f3
Revises: f7a8b9c0d1e2
Create Date: 2026-03-09 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'a8b9c0d1e2f3'
down_revision: Union[str, None] = 'f7a8b9c0d1e2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'mesh_states',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('frame_id', sa.Integer(), sa.ForeignKey('frames.id'), nullable=False),
        sa.Column('subject', sa.String(20), nullable=False),
        sa.Column('body_pose', sa.JSON(), nullable=False),
        sa.Column('global_orient', sa.JSON(), nullable=False),
        sa.Column('betas', sa.JSON(), nullable=False),
        sa.Column('joints_3d', sa.JSON(), nullable=False),
        sa.Column('global_translation', sa.JSON(), nullable=True),
        sa.Column('foot_contact', sa.JSON(), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
    )
    op.create_index('ix_mesh_states_frame_subject', 'mesh_states', ['frame_id', 'subject'])


def downgrade() -> None:
    op.drop_index('ix_mesh_states_frame_subject', 'mesh_states')
    op.drop_table('mesh_states')
