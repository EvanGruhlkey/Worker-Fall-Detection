"""Dashboard endpoints, shaped like the UI's types (ui/src/types.ts).

The rest of the API returns raw records; these endpoints do the presentation
work the dashboard needs (local times, percentages, status labels, stat cards)
so the frontend can render the responses as-is.
"""

import csv
import io
from datetime import datetime, timedelta, timezone
from enum import Enum

from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response
from sqlmodel import Session, func, select
from sqlalchemy import or_

from ..db import (
    EventType, FallEvent, Job, JobKind, JobStatus, ReviewStatus, get_session,
)
from ..deps import get_manager, get_settings
from ..location import current_site_location
from ..schemas import (
    STATUS_LABELS, Dashboard, DashboardCamera, DashboardEvent, DashboardLocation,
    DashboardStat, DashboardUpload,
)

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])

# Event type -> (name, icon, color) as the event log shows it.
EVENT_DISPLAY = {
    EventType.fall: ("Fall detected", "fall", "var(--red)"),
}
FALSE_ALARM_ICON, FALSE_ALARM_COLOR = "warn", "var(--amber)"

HEALTH_LABELS = {
    "good": "Good",
    "blurry": "Blurry",
    "obstructed": "Obstructed",
    "starting": "Starting",
    "offline": "Offline",
}
# Worst first, so several cameras report the most urgent problem.
HEALTH_SEVERITY = ["obstructed", "blurry", "starting", "good"]

UPTIME_WINDOW = timedelta(hours=24)
RECENT_UPLOADS = 5


class Scope(str, Enum):
    today = "today"
    all = "all"


def local_time_label(moment):
    """'10:42:11 AM' today, 'Oct 19, 10:42:11 AM' before, in the server's local time zone."""
    local = moment.astimezone()
    label = local.strftime("%I:%M:%S %p").lstrip("0")
    today = datetime.now().astimezone().date()
    if local.date() == today:
        return label
    year = f" {local.year}" if local.year != today.year else ""
    return f"{local:%b} {local.day}{year}, {label}"


def start_of_today(now):
    """Local midnight, as a UTC datetime."""
    local = now.astimezone()
    return local.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)


def to_dashboard_event(event, source):
    fallback = (EventType(event.event_type).value.replace("_", " ").capitalize(), "alert", "var(--muted)")
    name, icon, color = EVENT_DISPLAY.get(event.event_type, fallback)
    if event.status == ReviewStatus.false_alarm:
        icon, color = FALSE_ALARM_ICON, FALSE_ALARM_COLOR
    return DashboardEvent(
        id=event.id,
        time=local_time_label(event.detected_at),
        name=name,
        icon=icon,
        color=color,
        confidence=None if event.confidence is None else round(event.confidence * 100),
        status=STATUS_LABELS[event.status],
        detected_at=event.detected_at,
        source=source,
        job_id=event.job_id,
        camera=event.camera_name,
        location=event.location,
        coords=event.coords,
        video_time=event.video_time,
        track_id=event.track_id,
        reason=event.reason,
        reviewer=event.reviewer,
        notes=event.notes,
        reviewed_at=event.reviewed_at,
        snapshot_url=f"/api/events/{event.id}/snapshot" if event.snapshot_path else None,
        clip_url=f"/api/events/{event.id}/clip" if event.clip_path else None,
    )


def query_events(session, scope, status, limit, now):
    """(event, source kind) pairs, newest first."""
    query = (
        select(FallEvent, Job.kind)
        .join(Job, Job.id == FallEvent.job_id)
        .order_by(FallEvent.detected_at.desc(), FallEvent.id.desc())
    )
    if scope == Scope.today:
        query = query.where(FallEvent.detected_at >= start_of_today(now))
    if status:
        query = query.where(FallEvent.status == status)
    if limit:
        query = query.limit(limit)
    return session.exec(query).all()


def uptime_ratio(intervals, start, end):
    """Share of [start, end] covered by at least one (begin, finish) interval."""
    if end <= start:
        return None
    covered = timedelta()
    cursor = start
    complete_intervals = [
        (begin, finish)
        for begin, finish in intervals
        if begin is not None and finish is not None
    ]
    for begin, finish in sorted(complete_intervals, key=lambda interval: interval[0]):
        begin, finish = max(begin, cursor), min(finish, end)
        if finish > begin:
            covered += finish - begin
            cursor = finish
    return covered / (end - start)


def _active_camera_jobs(session, manager):
    """Running camera jobs, newest first, paired with their worker thread."""
    workers = manager.active_cameras()
    if not workers:
        return []
    jobs = session.exec(
        select(Job).where(Job.id.in_(list(workers))).order_by(Job.id.desc())
    ).all()
    return [(job, workers[job.id]) for job in jobs]


def _latest_camera_job(session):
    return session.exec(
        select(Job).where(Job.kind == JobKind.camera).order_by(Job.id.desc())
    ).first()


def incidents_stat(session):
    counts = dict(
        session.exec(
            select(FallEvent.status, func.count())
            .where(FallEvent.status.in_([ReviewStatus.unacknowledged, ReviewStatus.needs_review]))
            .group_by(FallEvent.status)
        ).all()
    )
    unacknowledged = counts.get(ReviewStatus.unacknowledged, 0)
    needs_review = counts.get(ReviewStatus.needs_review, 0)
    total = unacknowledged + needs_review
    if total:
        note = f"{unacknowledged} unacknowledged · {needs_review} need review"
    else:
        note = "No open incidents"
    return DashboardStat(
        label="Active incidents", value=str(total), note=note,
        tone="red" if total else "green", icon="alert",
    )


def cameras_stat(active, any_camera_ever):
    if active:
        names = ", ".join(job.name for job, _ in active)
        return DashboardStat(
            label="Cameras online", value=str(len(active)), note=names, tone="green", icon="cam",
        )
    return DashboardStat(
        label="Cameras online", value="0", note="No camera running",
        tone="red" if any_camera_ever else "default", icon="cam",
    )


def uptime_stat(session, active_ids, now):
    window_start = now - UPTIME_WINDOW
    jobs = session.exec(
        select(Job).where(
            Job.kind == JobKind.camera,
            Job.started_at.is_not(None),
            (Job.finished_at.is_(None)) | (Job.finished_at >= window_start),
        )
    ).all()
    first_start = session.exec(
        select(func.min(Job.started_at)).where(Job.kind == JobKind.camera)
    ).one()

    intervals = []
    for job in jobs:
        if job.id in active_ids:
            finish = now
        else:
            finish = job.finished_at or job.last_frame_at or job.started_at
        intervals.append((job.started_at, finish))

    # Measure from the first camera session if that is less than 24 h ago,
    # so a system installed an hour ago isn't reported as 4% up.
    start = max(window_start, first_start) if first_start else None
    ratio = uptime_ratio(intervals, start, now) if start else None
    if ratio is None:
        return DashboardStat(
            label="Detection uptime", value="-", note="No camera sessions yet",
            tone="default", icon="pulse",
        )
    percent = ratio * 100
    tone = "green" if percent >= 95 else "red" if percent < 80 else "default"
    hours = (now - start).total_seconds() / 3600
    if hours >= 23.9:
        note = "Last 24 hours"
    elif hours >= 1:
        note = f"Last {hours:.1f} hours"
    elif hours * 60 >= 1:
        note = f"Last {hours * 60:.0f} min"
    else:
        note = "Just started"
    return DashboardStat(
        label="Detection uptime", value=f"{percent:.1f}%", note=note, tone=tone, icon="pulse",
    )


def health_stat(active):
    if not active:
        return DashboardStat(
            label="Camera health", value="Offline", note="No camera running",
            tone="default", icon="shield",
        )
    job, worker = min(active, key=lambda pair: HEALTH_SEVERITY.index(pair[1].health))
    health = worker.health
    notes = {
        "good": "Feed clear" if len(active) == 1 else "All feeds clear",
        "blurry": f"{job.name}: image out of focus or dirty lens",
        "obstructed": f"{job.name}: view blocked or too dark",
        "starting": f"{job.name}: waiting for first frames",
    }
    tone = "green" if health == "good" else "red" if health == "obstructed" else "default"
    return DashboardStat(
        label="Camera health", value=HEALTH_LABELS[health], note=notes[health], tone=tone,
        icon="shield",
    )


def build_camera(active, latest):
    if active:
        job, worker = active[0]
        return DashboardCamera(
            id=_camera_id(job), name=job.name, label=job.label or "Live camera",
            fps=round(worker.fps), online=True, job_id=job.id, health=worker.health,
            source=job.source if job.source.isdigit() else None,
            stream_url=f"/api/cameras/{job.id}/stream",
        )
    if latest:
        return DashboardCamera(
            id=_camera_id(latest), name=latest.name, label=latest.label or "Offline",
            fps=0, online=False, job_id=latest.id,
            source=latest.source if latest.source.isdigit() else None,
        )
    return DashboardCamera(id="—", name="No camera", label="Not connected", fps=0, online=False)


def _camera_id(job):
    return f"CAM-{int(job.source):02d}" if job.source.isdigit() else f"CAM-{job.id:02d}"


def build_uploads(session):
    jobs = session.exec(
        select(Job).where(or_(Job.kind == JobKind.video, Job.label == 'Video simulation'))
        .order_by(Job.id.desc()).limit(RECENT_UPLOADS)
    ).all()
    uploads = []
    for job in jobs:
        progress = None
        if job.status == JobStatus.completed:
            progress = 1.0
        elif job.total_frames:
            progress = min(1.0, job.frames_processed / job.total_frames)
        uploads.append(DashboardUpload(
            id=job.id, name=job.name, status=job.status, progress=progress,
            fall_count=job.fall_count, error=job.error, created_at=job.created_at,
            output_video_url=f"/api/jobs/{job.id}/video" if job.output_video else None,
        ))
    return uploads


def build_location(settings, session, camera_job):
    site_name, site_coords = current_site_location(settings, session)
    return DashboardLocation(
        name=(camera_job and camera_job.location) or site_name,
        coords=(camera_job and camera_job.coords) or site_coords or "",
    )


@router.get("", response_model=Dashboard)
def dashboard(
    scope: Scope = Scope.today,
    status: ReviewStatus | None = None,
    limit: int = Query(500, ge=1, le=1000),
    session: Session = Depends(get_session),
    settings=Depends(get_settings),
    manager=Depends(get_manager),
):
    """Everything the dashboard page shows, in one response. Poll every few seconds."""
    now = datetime.now(timezone.utc)
    active = _active_camera_jobs(session, manager)
    latest = _latest_camera_job(session)
    return Dashboard(
        location=build_location(settings, session, active[0][0] if active else latest),
        stats=[
            incidents_stat(session),
            cameras_stat(active, latest is not None),
            uptime_stat(session, {job.id for job, _ in active}, now),
            health_stat(active),
        ],
        camera=build_camera(active, latest),
        events=[to_dashboard_event(e, kind) for e, kind in query_events(session, scope, status, limit, now)],
        uploads=build_uploads(session),
    )


CSV_COLUMNS = [
    "id", "detected_at_utc", "local_time", "event", "camera", "location", "coords",
    "confidence_percent", "status", "track_id", "reason", "reviewer", "notes", "reviewed_at_utc",
    "source", "video_time_seconds",
]


@router.get("/events.csv", response_class=Response)
def export_events(
    scope: Scope = Scope.all,
    status: ReviewStatus | None = None,
    session: Session = Depends(get_session),
):
    """The event log as a CSV download (the dashboard's "Export log" button)."""
    now = datetime.now(timezone.utc)
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(CSV_COLUMNS)
    for event, kind in query_events(session, scope, status, None, now):
        row = to_dashboard_event(event, kind)
        reviewed = event.reviewed_at
        writer.writerow([
            event.id, row.detected_at.isoformat(), row.time, row.name, row.camera,
            row.location, row.coords, row.confidence, row.status, row.track_id,
            row.reason, event.reviewer, event.notes, reviewed.isoformat() if reviewed else None,
            kind.value, event.video_time,
        ])
    filename = f"fall-events-{now.astimezone():%Y-%m-%d}.csv"
    return Response(
        buffer.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
