"""Request and response models for the API."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .db import EventType, JobKind, JobStatus, ReviewStatus

# Status names as the dashboard shows them (ui/src/types.ts EventStatus).
STATUS_LABELS = {
    ReviewStatus.unacknowledged: "Unacknowledged",
    ReviewStatus.needs_review: "Needs review",
    ReviewStatus.resolved: "Resolved",
    ReviewStatus.false_alarm: "False alarm",
}


class JobRead(BaseModel):
    id: int
    kind: JobKind
    name: str
    label: str | None
    location: str | None
    coords: str | None
    status: JobStatus
    frames_processed: int
    total_frames: int | None
    fps: float | None
    fall_count: int
    error: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    progress: float | None = None
    output_video_url: str | None = None
    output_json_url: str | None = None
    stream_url: str | None = None


class CameraStart(BaseModel):
    source: str = Field(
        "0",
        description="Camera index (e.g. '0') or a stream URL such as rtsp://...",
    )
    name: str | None = None
    label: str | None = Field(None, description="Short description, e.g. 'Loading dock'")
    location: str | None = Field(None, description="Overrides FALL_SITE_LOCATION")
    coords: str | None = Field(None, description="Overrides FALL_SITE_COORDS")
    width: int | None = Field(None, gt=0)
    height: int | None = Field(None, gt=0)


class EventRead(BaseModel):
    id: int
    job_id: int
    job_name: str | None = None
    event_type: EventType
    camera_name: str | None
    location: str | None
    coords: str | None
    track_id: int | None
    frame_index: int
    video_time: float
    detected_at: datetime
    confidence: float
    reason: str
    status: ReviewStatus
    reviewer: str | None
    notes: str | None
    reviewed_at: datetime | None
    snapshot_url: str | None = None
    clip_url: str | None = None


class EventReview(BaseModel):
    status: ReviewStatus
    reviewer: str | None = Field(None, max_length=100)
    notes: str | None = Field(None, max_length=2000)

    @field_validator("status", mode="before")
    @classmethod
    def _accept_dashboard_labels(cls, value):
        """Accept "Needs review" as well as "needs_review"."""
        for status, label in STATUS_LABELS.items():
            if value == label:
                return status
        return value


class EventPage(BaseModel):
    items: list[EventRead]
    total: int


class Stats(BaseModel):
    total_events: int
    unacknowledged: int
    needs_review: int
    resolved: int
    false_alarm: int
    active_cameras: int
    running_jobs: int


# Dashboard models: the same shapes as ui/src/types.ts, so the frontend can use
# them without converting. Extra fields are additions the UI can ignore.

Tone = Literal["red", "green", "default"]


class DashboardLocation(BaseModel):
    name: str
    coords: str


class DashboardStat(BaseModel):
    label: str
    value: str
    note: str
    tone: Tone
    icon: str


class DashboardCamera(BaseModel):
    id: str
    name: str
    label: str
    fps: float
    online: bool
    job_id: int | None = None
    health: str = "offline"
    stream_url: str | None = None


class DashboardEvent(BaseModel):
    id: int
    time: str = Field(description="Local time of detection, e.g. '10:42:11 AM'")
    name: str
    icon: str
    color: str
    confidence: int | None = Field(description="Percent")
    status: Literal["Unacknowledged", "Needs review", "Resolved", "False alarm"]
    # Additions beyond ui/src/types.ts FallEvent:
    detected_at: datetime
    source: JobKind = Field(description="'camera' (live) or 'video' (uploaded)")
    job_id: int
    camera: str | None = Field(description="Camera name, or the file name for uploads")
    location: str | None
    coords: str | None
    video_time: float = Field(description="Seconds into the video / camera session")
    track_id: int | None
    reason: str
    reviewer: str | None
    notes: str | None
    reviewed_at: datetime | None
    snapshot_url: str | None
    clip_url: str | None


class DashboardUpload(BaseModel):
    """An uploaded video and its processing progress."""

    id: int
    name: str
    status: JobStatus
    progress: float | None = Field(description="0 to 1, while running")
    fall_count: int
    error: str | None
    created_at: datetime
    output_video_url: str | None


class Dashboard(BaseModel):
    location: DashboardLocation
    stats: list[DashboardStat]
    camera: DashboardCamera
    events: list[DashboardEvent]
    uploads: list[DashboardUpload] = Field(description="Most recent uploads, newest first")
