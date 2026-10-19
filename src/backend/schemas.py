"""Request and response models for the API."""

from datetime import datetime

from pydantic import BaseModel, Field

from .db import JobKind, JobStatus, ReviewStatus


class JobRead(BaseModel):
    id: int
    kind: JobKind
    name: str
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
    width: int | None = Field(None, gt=0)
    height: int | None = Field(None, gt=0)


class EventRead(BaseModel):
    id: int
    job_id: int
    job_name: str | None = None
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


class EventPage(BaseModel):
    items: list[EventRead]
    total: int


class Stats(BaseModel):
    total_events: int
    pending: int
    confirmed: int
    false_alarm: int
    active_cameras: int
    running_jobs: int
