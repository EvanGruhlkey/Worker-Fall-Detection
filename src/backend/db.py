"""Database tables and session helpers (PostgreSQL via SQLModel).

The schema is created and upgraded by the Alembic migrations in
``backend/migrations``, which run on startup. After changing a table here,
generate a migration (see docs/database.md).
"""

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from sqlalchemy.engine import make_url
from sqlalchemy.exc import OperationalError
from sqlmodel import Field, Session, SQLModel, create_engine

MIGRATIONS_DIR = Path(__file__).resolve().with_name("migrations")


def utc_now():
    return datetime.now(timezone.utc)


def _timestamp(**kwargs):
    """A timezone-aware column (timestamptz); values come back in UTC."""
    return Field(sa_type=sa.DateTime(timezone=True), **kwargs)


def _choice(enum_class, **kwargs):
    """Store an enum as plain text, so adding a value needs no type change."""
    return Field(sa_type=sa.Enum(enum_class, native_enum=False, length=32), **kwargs)


class JobKind(str, Enum):
    video = "video"
    camera = "camera"


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    stopped = "stopped"
    failed = "failed"


class ReviewStatus(str, Enum):
    """Matches the dashboard's event statuses.

    New detections start as ``unacknowledged``. Someone acknowledging the alert
    moves it to ``needs_review``; after checking the footage they close it as
    ``resolved`` (a real incident, handled) or ``false_alarm``.
    """

    unacknowledged = "unacknowledged"
    needs_review = "needs_review"
    resolved = "resolved"
    false_alarm = "false_alarm"


class EventType(str, Enum):
    fall = "fall"


class Job(SQLModel, table=True):
    """One processing run: an uploaded video or a live camera session."""

    id: int | None = Field(default=None, primary_key=True)
    kind: JobKind = _choice(JobKind)
    name: str
    source: str
    # Shown on the dashboard under the camera name, e.g. "North entrance".
    label: str | None = None
    # Overrides the site location for this camera; copied onto its events.
    location: str | None = None
    coords: str | None = None
    status: JobStatus = _choice(JobStatus, default=JobStatus.queued)
    frames_processed: int = 0
    total_frames: int | None = None
    fps: float | None = None
    fall_count: int = 0
    error: str | None = None
    # Paths relative to the data directory.
    output_video: str | None = None
    output_json: str | None = None
    created_at: datetime = _timestamp(default_factory=utc_now)
    started_at: datetime | None = _timestamp(default=None)
    # Updated while frames are flowing, so uptime is right after a crash.
    last_frame_at: datetime | None = _timestamp(default=None)
    finished_at: datetime | None = _timestamp(default=None)


class FallEvent(SQLModel, table=True):
    """A detected fall, waiting for (or after) human review.

    Camera and location are copied from the job when the event is recorded, so
    the log stays accurate if a camera is later renamed or moved.
    """

    id: int | None = Field(default=None, primary_key=True)
    job_id: int = Field(foreign_key="job.id", index=True)
    event_type: EventType = _choice(EventType, default=EventType.fall)
    camera_name: str | None = None
    location: str | None = None
    coords: str | None = None
    track_id: int | None = None
    frame_index: int
    # Position in the source video, in seconds (camera: since the session began).
    video_time: float
    detected_at: datetime = _timestamp(default_factory=utc_now, index=True)
    confidence: float
    reason: str
    status: ReviewStatus = _choice(ReviewStatus, default=ReviewStatus.unacknowledged, index=True)
    reviewer: str | None = None
    notes: str | None = None
    reviewed_at: datetime | None = _timestamp(default=None)
    snapshot_path: str | None = None
    clip_path: str | None = None


class SiteLocation(SQLModel, table=True):
    """Where the system is now, as last reported by the dashboard's browser.

    A single row (id 1). It's kept across restarts, so events recorded before
    the dashboard is opened again still get the last known location.
    """

    id: int | None = Field(default=1, primary_key=True)
    name: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    # Browser's estimate of the error radius, in meters.
    accuracy: float | None = None
    updated_at: datetime | None = _timestamp(default=None)


_engine = None


def alembic_config(database_url):
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS_DIR))
    # ConfigParser treats % as interpolation; passwords may contain it.
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    return config


def create_db_engine(database_url):
    return create_engine(
        database_url,
        # Reconnect quietly if the database restarted while we were idle.
        pool_pre_ping=True,
        # Timestamps are stored and read back in UTC whatever the server's zone.
        connect_args={"options": "-c timezone=UTC"},
    )


def init_db(database_url):
    """Connect, bring the schema up to date, and make the engine current."""
    global _engine
    close_db()
    engine = create_db_engine(database_url)
    try:
        with engine.begin() as connection:
            config = alembic_config(database_url)
            config.attributes["connection"] = connection
            command.upgrade(config, "head")
    except OperationalError as error:
        engine.dispose()
        where = make_url(database_url).render_as_string(hide_password=True)
        raise RuntimeError(
            f"Could not connect to the database at {where}. Check that PostgreSQL "
            "is running and the FALL_DB_* settings in .env are right "
            "(see docs/database.md)."
        ) from error
    _engine = engine
    return engine


def close_db():
    global _engine
    if _engine is not None:
        _engine.dispose()
        _engine = None


def get_engine():
    if _engine is None:
        raise RuntimeError("Database not initialised; call init_db() first")
    return _engine


def get_session():
    """FastAPI dependency that yields a database session."""
    with Session(get_engine()) as session:
        yield session


def update_row(model, row_id, **changes):
    """Apply changes to one row from a worker thread; returns the fresh row."""
    with Session(get_engine()) as session:
        row = session.get(model, row_id)
        if row is None:
            return None
        for key, value in changes.items():
            setattr(row, key, value)
        session.add(row)
        session.commit()
        session.refresh(row)
        return row
