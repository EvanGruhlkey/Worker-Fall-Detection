"""Fall event log: list, inspect, review, and fetch evidence."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session, func, select

from ..db import FallEvent, Job, JobStatus, ReviewStatus, get_session, utc_now
from ..deps import data_file_response, get_manager, get_settings
from ..schemas import EventPage, EventRead, EventReview, Stats

router = APIRouter(prefix="/api", tags=["events"])


def to_event_read(event, job_name=None):
    return EventRead(
        **event.model_dump(exclude={"snapshot_path", "clip_path"}),
        job_name=job_name,
        snapshot_url=f"/api/events/{event.id}/snapshot" if event.snapshot_path else None,
        clip_url=f"/api/events/{event.id}/clip" if event.clip_path else None,
    )


def _get_event(session, event_id):
    event = session.get(FallEvent, event_id)
    if event is None:
        raise HTTPException(404, "Event not found")
    return event


@router.get("/events", response_model=EventPage)
def list_events(
    status: ReviewStatus | None = None,
    job_id: int | None = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    session: Session = Depends(get_session),
):
    """Fall log, newest first. Filter by review status or job."""
    filters = []
    if status:
        filters.append(FallEvent.status == status)
    if job_id is not None:
        filters.append(FallEvent.job_id == job_id)

    total = session.exec(select(func.count()).select_from(FallEvent).where(*filters)).one()
    rows = session.exec(
        select(FallEvent, Job.name)
        .join(Job, Job.id == FallEvent.job_id)
        .where(*filters)
        .order_by(FallEvent.detected_at.desc(), FallEvent.id.desc())
        .offset(offset)
        .limit(limit)
    ).all()
    return EventPage(items=[to_event_read(e, name) for e, name in rows], total=total)


@router.get("/events/{event_id}", response_model=EventRead)
def get_event(event_id: int, session: Session = Depends(get_session)):
    event = _get_event(session, event_id)
    job = session.get(Job, event.job_id)
    return to_event_read(event, job.name if job else None)


@router.patch("/events/{event_id}", response_model=EventRead)
def review_event(event_id: int, review: EventReview, session: Session = Depends(get_session)):
    """Mark a fall as confirmed or a false alarm (or reset it to pending)."""
    event = _get_event(session, event_id)
    event.status = review.status
    event.reviewer = review.reviewer
    event.notes = review.notes
    event.reviewed_at = None if review.status == ReviewStatus.pending else utc_now()
    session.add(event)
    session.commit()
    session.refresh(event)
    job = session.get(Job, event.job_id)
    return to_event_read(event, job.name if job else None)


@router.get("/events/{event_id}/snapshot")
def event_snapshot(event_id: int, session: Session = Depends(get_session), settings=Depends(get_settings)):
    """Annotated frame at the moment the fall was detected."""
    return data_file_response(settings, _get_event(session, event_id).snapshot_path, "image/jpeg")


@router.get("/events/{event_id}/clip")
def event_clip(event_id: int, session: Session = Depends(get_session), settings=Depends(get_settings)):
    """Short H.264 clip from a few seconds before to a few seconds after the fall."""
    return data_file_response(settings, _get_event(session, event_id).clip_path, "video/mp4")


@router.get("/stats", response_model=Stats)
def stats(session: Session = Depends(get_session), manager=Depends(get_manager)):
    counts = dict(
        session.exec(select(FallEvent.status, func.count()).group_by(FallEvent.status)).all()
    )
    running = session.exec(
        select(func.count()).select_from(Job).where(Job.status == JobStatus.running)
    ).one()
    return Stats(
        total_events=sum(counts.values()),
        pending=counts.get(ReviewStatus.pending, 0),
        confirmed=counts.get(ReviewStatus.confirmed, 0),
        false_alarm=counts.get(ReviewStatus.false_alarm, 0),
        active_cameras=len(manager.active_cameras()),
        running_jobs=running,
    )
