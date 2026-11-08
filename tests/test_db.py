from datetime import datetime, timedelta, timezone

from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlmodel import Session, SQLModel

from backend.db import FallEvent, Job, JobKind, ReviewStatus, close_db, create_db_engine, init_db


def test_migrations_match_models(database_url):
    """Fails when a table in db.py changed without a migration (docs/database.md)."""
    engine = create_db_engine(database_url)
    try:
        with engine.connect() as connection:
            context = MigrationContext.configure(connection, opts={"compare_type": True})
            assert compare_metadata(context, SQLModel.metadata) == []
    finally:
        engine.dispose()


def test_rows_round_trip_with_utc_times(database_url):
    engine = init_db(database_url)
    try:
        # A non-UTC time must come back as the same moment, in UTC.
        started = datetime(2026, 10, 30, 9, 0, tzinfo=timezone(timedelta(hours=-5)))
        with Session(engine) as session:
            job = Job(kind=JobKind.camera, name="Camera 0", source="0", started_at=started)
            session.add(job)
            session.commit()
            event = FallEvent(
                job_id=job.id, frame_index=1, video_time=0.1, confidence=0.9, reason="test"
            )
            session.add(event)
            session.commit()
            job_id, event_id = job.id, event.id

        with Session(engine) as session:
            job = session.get(Job, job_id)
            event = session.get(FallEvent, event_id)
            assert job.started_at == started
            assert job.started_at.utcoffset() == timedelta(0)
            assert event.detected_at.tzinfo is not None
            assert event.status == ReviewStatus.unacknowledged
            assert job.kind == JobKind.camera
    finally:
        close_db()
