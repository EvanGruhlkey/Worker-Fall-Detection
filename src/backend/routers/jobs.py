"""Endpoints for uploaded videos, live cameras, and their processing jobs."""

import asyncio
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse
from sqlmodel import Session, select

from ..db import Job, JobKind, JobStatus, get_session
from ..deps import data_file_response, get_manager, get_settings
from ..schemas import CameraOption, CameraStart, JobRead

router = APIRouter(prefix="/api", tags=["jobs"])

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}
UPLOAD_CHUNK = 1024 * 1024


def to_job_read(job, manager=None):
    progress = None
    if job.status == JobStatus.completed:
        progress = 1.0
    elif job.total_frames:
        progress = min(1.0, job.frames_processed / job.total_frames)

    stream_url = None
    if manager and job.kind == JobKind.camera and job.id in manager.active_cameras():
        stream_url = f"/api/cameras/{job.id}/stream"

    return JobRead(
        **job.model_dump(exclude={"source", "output_video", "output_json", "last_frame_at"}),
        progress=progress,
        output_video_url=f"/api/jobs/{job.id}/video" if job.output_video else None,
        output_json_url=f"/api/jobs/{job.id}/json" if job.output_json else None,
        stream_url=stream_url,
    )


def _get_job(session, job_id):
    job = session.get(Job, job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job


@router.post("/videos", response_model=JobRead, status_code=201)
async def upload_video(
    file: UploadFile = File(...),
    live: bool = Form(False),
    session: Session = Depends(get_session),
    settings=Depends(get_settings),
    manager=Depends(get_manager),
):
    """Upload a prerecorded video and queue it for fall detection."""
    original_name = Path(file.filename or "video.mp4").name
    if Path(original_name).suffix.lower() not in VIDEO_EXTENSIONS:
        raise HTTPException(
            400, f"Unsupported file type; use one of {sorted(VIDEO_EXTENSIONS)}"
        )

    destination = settings.input_dir / f"{uuid.uuid4().hex[:8]}_{original_name}"
    with destination.open("wb") as out:
        while chunk := await file.read(UPLOAD_CHUNK):
            out.write(chunk)

    if live:
        return to_job_read(_start_simulation(manager, destination, original_name), manager)
    job = Job(
        kind=JobKind.video, station=settings.station_name, name=original_name,
        source=str(destination),
    )
    session.add(job)
    session.commit()
    session.refresh(job)
    manager.submit_video(job.id)
    return to_job_read(job, manager)


def _start_simulation(manager, source, name):
    for job_id, worker in list(manager.active_cameras().items()):
        if '://' not in worker.source and Path(worker.source).is_file():
            manager.stop_camera(job_id)
    return manager.create_camera(str(source), name=name, label='Video simulation')


@router.post('/jobs/{job_id}/simulate', response_model=JobRead, status_code=201)
def simulate_video(job_id: int, session: Session = Depends(get_session), manager=Depends(get_manager)):
    job = _get_job(session, job_id)
    source = Path(job.source)
    if not source.is_file():
        raise HTTPException(404, 'The original video is not available on this computer')
    return to_job_read(_start_simulation(manager, source, job.name), manager)


@router.get("/jobs", response_model=list[JobRead])
def list_jobs(
    kind: JobKind | None = None,
    status: JobStatus | None = None,
    limit: int = Query(50, ge=1, le=500),
    session: Session = Depends(get_session),
    manager=Depends(get_manager),
):
    query = select(Job).order_by(Job.created_at.desc()).limit(limit)
    if kind:
        query = query.where(Job.kind == kind)
    if status:
        query = query.where(Job.status == status)
    return [to_job_read(job, manager) for job in session.exec(query).all()]


@router.get("/jobs/{job_id}", response_model=JobRead)
def get_job(job_id: int, session: Session = Depends(get_session), manager=Depends(get_manager)):
    return to_job_read(_get_job(session, job_id), manager)


@router.get("/jobs/{job_id}/video")
def job_video(job_id: int, session: Session = Depends(get_session), settings=Depends(get_settings)):
    """The full annotated output video of a processed upload."""
    job = _get_job(session, job_id)
    return data_file_response(settings, job.output_video, "video/mp4")


@router.get("/jobs/{job_id}/json")
def job_json(job_id: int, session: Session = Depends(get_session), settings=Depends(get_settings)):
    """Per-frame pose and tracking data (same format as src/detectors/yolo_pose.py)."""
    job = _get_job(session, job_id)
    return data_file_response(
        settings, job.output_json, "application/json", filename=f"job_{job_id}.json"
    )


@router.post("/cameras", response_model=JobRead, status_code=201)
def start_camera(body: CameraStart, manager=Depends(get_manager)):
    """Start live fall detection on a camera index or stream URL."""
    try:
        job = manager.create_camera(
            body.source, name=body.name, label=body.label, location=body.location,
            coords=body.coords, width=body.width, height=body.height,
        )
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    return to_job_read(job, manager)


@router.get("/cameras/options", response_model=list[CameraOption])
def camera_options(manager=Depends(get_manager)):
    """Discover local cameras that can be selected in the dashboard."""
    return [
        CameraOption(source=source, label=f"Camera {source}")
        for source in manager.camera_options()
    ]


@router.post("/cameras/{job_id}/stop", response_model=JobRead)
async def stop_camera(job_id: int, session: Session = Depends(get_session), manager=Depends(get_manager)):
    if not manager.get_camera(job_id):
        raise HTTPException(404, "Camera session not found")
    await asyncio.to_thread(manager.stop_camera, job_id)
    job = _get_job(session, job_id)
    session.refresh(job)
    return to_job_read(job, manager)


@router.get("/cameras/{job_id}/stream")
async def camera_stream(job_id: int, request: Request, manager=Depends(get_manager)):
    """Live annotated feed as MJPEG; use it directly as an <img> src."""
    worker = manager.get_camera(job_id)
    if worker is None or not worker.is_alive():
        raise HTTPException(404, "Camera is not running")

    async def frames():
        last = 0
        while not await request.is_disconnected():
            number, jpeg = await asyncio.to_thread(worker.wait_for_frame, last, 1.0)
            if number == last or jpeg is None:
                if not worker.is_alive():
                    break
                continue
            last = number
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"

    return StreamingResponse(
        frames(), media_type="multipart/x-mixed-replace; boundary=frame"
    )
