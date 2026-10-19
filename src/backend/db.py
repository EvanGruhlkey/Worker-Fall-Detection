"""Database tables and session helpers (SQLite by default via SQLModel)."""

from datetime import datetime, timezone
from enum import Enum

from sqlalchemy import event
from sqlmodel import Field, Session, SQLModel, create_engine


def utc_now():
    return datetime.now(timezone.utc)


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
    pending = "pending"
    confirmed = "confirmed"
    false_alarm = "false_alarm"


class Job(SQLModel, table=True):
    """One processing run: an uploaded video or a live camera session."""

    id: int | None = Field(default=None, primary_key=True)
    kind: JobKind
    name: str
    source: str
    status: JobStatus = JobStatus.queued
    frames_processed: int = 0
    total_frames: int | None = None
    fps: float | None = None
    fall_count: int = 0
    error: str | None = None
    # Paths relative to the data directory.
    output_video: str | None = None
    output_json: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None


class FallEvent(SQLModel, table=True):
    """A detected fall, waiting for (or after) human review."""

    id: int | None = Field(default=None, primary_key=True)
    job_id: int = Field(foreign_key="job.id", index=True)
    track_id: int | None = None
    frame_index: int
    # Position in the source video, in seconds (camera: since the session began).
    video_time: float
    detected_at: datetime = Field(default_factory=utc_now, index=True)
    confidence: float
    reason: str
    status: ReviewStatus = Field(default=ReviewStatus.pending, index=True)
    reviewer: str | None = None
    notes: str | None = None
    reviewed_at: datetime | None = None
    snapshot_path: str | None = None
    clip_path: str | None = None


_engine = None


def init_db(database_url):
    global _engine
    connect_args = {}
    if database_url.startswith("sqlite"):
        # Worker threads write to the database alongside request handlers.
        connect_args = {"check_same_thread": False, "timeout": 30}
    _engine = create_engine(database_url, connect_args=connect_args)

    if database_url.startswith("sqlite"):

        @event.listens_for(_engine, "connect")
        def _sqlite_pragmas(dbapi_connection, _record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    SQLModel.metadata.create_all(_engine)
    return _engine


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
