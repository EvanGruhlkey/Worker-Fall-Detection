"""Initial tables: job, fallevent, sitelocation

Revision ID: 0001
Revises:
Create Date: 2026-10-30
"""

import sqlalchemy as sa
import sqlmodel
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def _choice(name, *values):
    return sa.Enum(*values, name=name, native_enum=False, length=32)


def upgrade():
    op.create_table(
        "job",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("kind", _choice("jobkind", "video", "camera"), nullable=False),
        sa.Column("name", sqlmodel.AutoString(), nullable=False),
        sa.Column("source", sqlmodel.AutoString(), nullable=False),
        sa.Column("label", sqlmodel.AutoString(), nullable=True),
        sa.Column("location", sqlmodel.AutoString(), nullable=True),
        sa.Column("coords", sqlmodel.AutoString(), nullable=True),
        sa.Column(
            "status",
            _choice("jobstatus", "queued", "running", "completed", "stopped", "failed"),
            nullable=False,
        ),
        sa.Column("frames_processed", sa.Integer(), nullable=False),
        sa.Column("total_frames", sa.Integer(), nullable=True),
        sa.Column("fps", sa.Float(), nullable=True),
        sa.Column("fall_count", sa.Integer(), nullable=False),
        sa.Column("error", sqlmodel.AutoString(), nullable=True),
        sa.Column("output_video", sqlmodel.AutoString(), nullable=True),
        sa.Column("output_json", sqlmodel.AutoString(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_frame_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "fallevent",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("job_id", sa.Integer(), nullable=False),
        sa.Column("event_type", _choice("eventtype", "fall"), nullable=False),
        sa.Column("camera_name", sqlmodel.AutoString(), nullable=True),
        sa.Column("location", sqlmodel.AutoString(), nullable=True),
        sa.Column("coords", sqlmodel.AutoString(), nullable=True),
        sa.Column("track_id", sa.Integer(), nullable=True),
        sa.Column("frame_index", sa.Integer(), nullable=False),
        sa.Column("video_time", sa.Float(), nullable=False),
        sa.Column("detected_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("reason", sqlmodel.AutoString(), nullable=False),
        sa.Column(
            "status",
            _choice("reviewstatus", "unacknowledged", "needs_review", "resolved", "false_alarm"),
            nullable=False,
        ),
        sa.Column("reviewer", sqlmodel.AutoString(), nullable=True),
        sa.Column("notes", sqlmodel.AutoString(), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("snapshot_path", sqlmodel.AutoString(), nullable=True),
        sa.Column("clip_path", sqlmodel.AutoString(), nullable=True),
        sa.ForeignKeyConstraint(["job_id"], ["job.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_fallevent_job_id", "fallevent", ["job_id"])
    op.create_index("ix_fallevent_detected_at", "fallevent", ["detected_at"])
    op.create_index("ix_fallevent_status", "fallevent", ["status"])
    op.create_table(
        "sitelocation",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sqlmodel.AutoString(), nullable=True),
        sa.Column("latitude", sa.Float(), nullable=True),
        sa.Column("longitude", sa.Float(), nullable=True),
        sa.Column("accuracy", sa.Float(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade():
    op.drop_table("sitelocation")
    op.drop_index("ix_fallevent_status", table_name="fallevent")
    op.drop_index("ix_fallevent_detected_at", table_name="fallevent")
    op.drop_index("ix_fallevent_job_id", table_name="fallevent")
    op.drop_table("fallevent")
    op.drop_table("job")
