"""Database tables and session helpers (SQLite by default via SQLModel)."""

from datetime import datetime, timezone
from enum import Enum

from sqlalchemy import event, inspect
from sqlmodel import Field, Session, SQLModel, create_engine


def utc_now():
    return datetime.now(timezone.utc)


def as_utc(value):
    """SQLite drops the timezone; stored datetimes are always UTC."""
    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=timezone.utc)


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
    kind: JobKind
    name: str
    source: str
    # Shown on the dashboard under the camera name, e.g. "North entrance".
    label: str | None = None
    # Overrides the site location for this camera; copied onto its events.
    location: str | None = None
    coords: str | None = None
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
    # Updated while frames are flowing, so uptime is right after a crash.
    last_frame_at: datetime | None = None
    finished_at: datetime | None = None


class FallEvent(SQLModel, table=True):
    """A detected fall, waiting for (or after) human review.

    Camera and location are copied from the job when the event is recorded, so
    the log stays accurate if a camera is later renamed or moved.
    """

    id: int | None = Field(default=None, primary_key=True)
    job_id: int = Field(foreign_key="job.id", index=True)
    event_type: EventType = EventType.fall
    camera_name: str | None = None
    location: str | None = None
    coords: str | None = None
    track_id: int | None = None
    frame_index: int
    # Position in the source video, in seconds (camera: since the session began).
    video_time: float
    detected_at: datetime = Field(default_factory=utc_now, index=True)
    confidence: float
    reason: str
    status: ReviewStatus = Field(default=ReviewStatus.unacknowledged, index=True)
    reviewer: str | None = None
    notes: str | None = None
    reviewed_at: datetime | None = None
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
    updated_at: datetime | None = None


# Review statuses used before the dashboard integration.
_STATUS_RENAMES = {"pending": "unacknowledged", "confirmed": "resolved"}

_engine = None


def _migrate_sqlite(engine):
    """Bring an older SQLite database up to the current tables.

    create_all() only creates missing tables, so new columns are added here
    and old review statuses are renamed. Existing rows are kept.
    """
    inspector = inspect(engine)
    with engine.begin() as connection:
        for table in SQLModel.metadata.sorted_tables:
            if not inspector.has_table(table.name):
                continue
            existing = {column["name"] for column in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name not in existing:
                    column_type = column.type.compile(engine.dialect)
                    connection.exec_driver_sql(
                        f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {column_type}'
                    )
        connection.exec_driver_sql(
            "UPDATE fallevent SET event_type = 'fall' WHERE event_type IS NULL"
        )
        for old, new in _STATUS_RENAMES.items():
            connection.exec_driver_sql(
                "UPDATE fallevent SET status = ? WHERE status = ?", (new, old)
            )


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
    if database_url.startswith("sqlite"):
        _migrate_sqlite(_engine)
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
