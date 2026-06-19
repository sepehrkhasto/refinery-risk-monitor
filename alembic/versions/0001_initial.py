"""Create initial database tables

Revision ID: 0001
Revises: None
Create Date: 2025-06-15 10:00:00
"""

from alembic import op
import sqlalchemy as sa

revision = '0001'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    op.create_table(
        'units',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('name', sa.String(100), unique=True, nullable=False, index=True),
        sa.Column('capacity', sa.Float),
        sa.Column('max_temperature', sa.Float),
        sa.Column('max_pressure', sa.Float),
        sa.Column('location', sa.String(200)),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
    )

    op.create_table(
        'sensor_readings',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('unit_id', sa.String(36), sa.ForeignKey('units.id'), nullable=False, index=True),
        sa.Column('timestamp', sa.DateTime, server_default=sa.func.now(), index=True),
        sa.Column('temperature_in', sa.Float),
        sa.Column('temperature_out', sa.Float),
        sa.Column('pressure_in', sa.Float),
        sa.Column('pressure_out', sa.Float),
        sa.Column('flow_rate', sa.Float),
        sa.Column('level', sa.Float),
        sa.Column('vibration', sa.Float),
        sa.Column('bearing_temp', sa.Float),
        sa.Column('motor_rpm', sa.Float),
        sa.Column('motor_current', sa.Float),
        sa.Column('torque', sa.Float),
        sa.Column('seal_pressure', sa.Float),
        sa.Column('valve_position_1', sa.Float),
        sa.Column('valve_position_2', sa.Float),
        sa.Column('gas_concentration', sa.Float),
        sa.Column('conductivity', sa.Float),
        sa.Column('ph', sa.Float),
        sa.Column('temp_ambient_coil', sa.Float),
        sa.Column('pressure_steam', sa.Float),
        sa.Column('oil_level', sa.Float),
        sa.Column('cooling_water_temp', sa.Float),
    )
    op.create_index('ix_sensor_unit_time', 'sensor_readings', ['unit_id', 'timestamp'])

    op.create_table(
        'environment_data',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('sensor_reading_id', sa.String(36), sa.ForeignKey('sensor_readings.id'), unique=True, nullable=False),
        sa.Column('ambient_temp', sa.Float),
        sa.Column('humidity', sa.Float),
        sa.Column('wind_speed', sa.Float),
        sa.Column('atmospheric_pressure', sa.Float),
        sa.Column('rainfall', sa.Float),
        sa.Column('solar_radiation', sa.Float),
    )

    op.create_table(
        'maintenance_logs',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('unit_id', sa.String(36), sa.ForeignKey('units.id'), nullable=False, index=True),
        sa.Column('timestamp', sa.DateTime, server_default=sa.func.now()),
        sa.Column('last_service_days', sa.Integer),
        sa.Column('component_age_days', sa.Integer),
        sa.Column('maintenance_type', sa.String(100)),
        sa.Column('replacement_count', sa.Integer),
        sa.Column('mtbf', sa.Float),
        sa.Column('mttr', sa.Float),
        sa.Column('last_repair_cost', sa.Float),
    )

    op.create_table(
        'hse_reports',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('unit_id', sa.String(36), sa.ForeignKey('units.id'), nullable=False, index=True),
        sa.Column('timestamp', sa.DateTime, server_default=sa.func.now()),
        sa.Column('last_incident_days', sa.Integer),
        sa.Column('incident_type', sa.String(100)),
        sa.Column('safety_audit_score', sa.Float),
        sa.Column('risk_assessment_level', sa.String(50)),
        sa.Column('operator_training_level', sa.String(50)),
    )

    op.create_table(
        'operator_notes',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('sensor_reading_id', sa.String(36), sa.ForeignKey('sensor_readings.id'), unique=True, nullable=False),
        sa.Column('operator_notes', sa.Text),
        sa.Column('shift_handover_issues', sa.Text),
        sa.Column('manual_alert', sa.String(100)),
        sa.Column('observation', sa.Text),
    )

    op.create_table(
        'sensor_health',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('sensor_reading_id', sa.String(36), sa.ForeignKey('sensor_readings.id'), unique=True, nullable=False),
        sa.Column('sensor_health_status', sa.String(50)),
        sa.Column('calibration_days_left', sa.Integer),
        sa.Column('drift_indicator', sa.Float),
    )

    op.create_table(
        'risk_assessments',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('sensor_reading_id', sa.String(36), sa.ForeignKey('sensor_readings.id'), unique=True, nullable=False),
        sa.Column('unit_id', sa.String(36), sa.ForeignKey('units.id'), nullable=False, index=True),
        sa.Column('timestamp', sa.DateTime, server_default=sa.func.now(), index=True),
        sa.Column('risk_score', sa.Float, nullable=False),
        sa.Column('status', sa.String(50), nullable=False),
        sa.Column('root_cause', sa.String(200)),
        sa.Column('prediction_confidence', sa.Float),
    )
    op.create_index('ix_risk_unit_time', 'risk_assessments', ['unit_id', 'timestamp'])

    op.create_table(
        'users',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('username', sa.String(50), unique=True, nullable=False, index=True),
        sa.Column('hashed_password', sa.String(200), nullable=False),
        sa.Column('full_name', sa.String(100)),
        sa.Column('role', sa.String(20), nullable=False, server_default='operator'),
        sa.Column('is_active', sa.Boolean, server_default=sa.text('1')),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
    )

    op.create_table(
        'operator_reports',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('user_id', sa.String(36), sa.ForeignKey('users.id'), nullable=False, index=True),
        sa.Column('unit_id', sa.String(36), sa.ForeignKey('units.id'), nullable=False),
        sa.Column('timestamp', sa.DateTime, server_default=sa.func.now(), index=True),
        sa.Column('description', sa.Text, nullable=False),
        sa.Column('sensor_name', sa.String(100)),
        sa.Column('reported_issue', sa.String(200)),
        sa.Column('risk_impact', sa.String(20)),
        sa.Column('status', sa.String(20), server_default='open'),
    )

def downgrade():
    op.drop_table('operator_reports')
    op.drop_table('users')
    op.drop_index('ix_risk_unit_time', table_name='risk_assessments')
    op.drop_table('risk_assessments')
    op.drop_table('sensor_health')
    op.drop_table('operator_notes')
    op.drop_table('hse_reports')
    op.drop_table('maintenance_logs')
    op.drop_table('environment_data')
    op.drop_index('ix_sensor_unit_time', table_name='sensor_readings')
    op.drop_table('sensor_readings')
    op.drop_table('units')