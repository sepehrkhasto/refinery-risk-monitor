"""add prediction_logs table

Revision ID: 0002
Revises: 0001
Create Date: 2026-05-19 21:00:00
"""

from alembic import op
import sqlalchemy as sa


revision = '0002'
down_revision = '0001'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'prediction_logs',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('unit_id', sa.String(36), sa.ForeignKey('units.id'), nullable=False, index=True),
        sa.Column('timestamp', sa.DateTime, server_default=sa.func.now(), index=True),
        sa.Column('predicted_p10', sa.Float),
        sa.Column('predicted_p50', sa.Float),
        sa.Column('predicted_p90', sa.Float),
        sa.Column('actual_risk', sa.Float, nullable=True),
        sa.Column('error_mae', sa.Float, nullable=True),
        sa.Column('error_rmse', sa.Float, nullable=True),
        sa.Column('status', sa.String(20), server_default='pending'),
    )


def downgrade():
    op.drop_table('prediction_logs')