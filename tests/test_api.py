import time
from pathlib import Path

import cv2
import pytest
import ultralytics
from sqlmodel import Session

from backend.db import FallEvent, Job, JobKind, JobStatus, get_engine
from backend.fall_detector import FallDetection
from backend.pipeline import FrameProcessor


def add_event(job_kind=JobKind.video, **event_fields):
    with Session(get_engine()) as session:
        job = Job(kind=job_kind, name="site.mp4", source="site.mp4", status=JobStatus.completed)
        session.add(job)
        session.commit()
        session.refresh(job)
        event = FallEvent(
            job_id=job.id, track_id=3, frame_index=90, video_time=3.0,
            confidence=0.8, reason="test", **event_fields,
        )
        session.add(event)
        session.commit()
        session.refresh(event)
        return job.id, event.id


def test_health(client):
    assert client.get("/api/health").json()["status"] == "ok"


def test_cors_allows_separate_frontend(client):
    response = client.get("/api/health", headers={"Origin": "http://localhost:5173"})
    assert response.headers["access-control-allow-origin"] == "*"


def test_event_review_flow(client):
    job_id, event_id = add_event()

    page = client.get("/api/events", params={"status": "pending"}).json()
    assert page["total"] == 1
    assert page["items"][0]["job_name"] == "site.mp4"

    response = client.patch(
        f"/api/events/{event_id}",
        json={"status": "false_alarm", "reviewer": "Sam", "notes": "Kneeling to tie boot"},
    )
    assert response.status_code == 200
    event = response.json()
    assert event["status"] == "false_alarm"
    assert event["reviewer"] == "Sam"
    assert event["reviewed_at"] is not None

    assert client.get("/api/events", params={"status": "pending"}).json()["total"] == 0
    stats = client.get("/api/stats").json()
    assert stats["false_alarm"] == 1 and stats["pending"] == 0

    reset = client.patch(f"/api/events/{event_id}", json={"status": "pending"}).json()
    assert reset["reviewed_at"] is None


def test_missing_resources_return_404(client):
    assert client.get("/api/events/999").status_code == 404
    assert client.patch("/api/events/999", json={"status": "confirmed"}).status_code == 404
    assert client.get("/api/jobs/999").status_code == 404
    _, event_id = add_event()
    # Event exists but its clip hasn't been written.
    assert client.get(f"/api/events/{event_id}/clip").status_code == 404


def test_files_outside_data_dir_are_refused(client):
    _, event_id = add_event(snapshot_path="../../outside.jpg")
    assert client.get(f"/api/events/{event_id}/snapshot").status_code == 404


def test_upload_rejects_non_video(client):
    response = client.post("/api/videos", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 400


def test_invalid_review_status_is_rejected(client):
    _, event_id = add_event()
    assert client.patch(f"/api/events/{event_id}", json={"status": "maybe"}).status_code == 422


@pytest.fixture
def sample_video(tmp_path):
    """Two seconds of a real photo with people in it, so the model detects someone."""
    image = cv2.imread(str(Path(ultralytics.__file__).parent / "assets" / "bus.jpg"))
    image = cv2.resize(image, (640, 864))
    path = tmp_path / "sample.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15, (640, 864))
    for _ in range(30):
        writer.write(image)
    writer.release()
    return path


def test_video_upload_end_to_end(client, sample_video):
    with sample_video.open("rb") as video:
        response = client.post("/api/videos", files={"file": ("sample.mp4", video, "video/mp4")})
    assert response.status_code == 201
    job_id = response.json()["id"]

    deadline = time.time() + 120
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("completed", "failed"):
            break
        time.sleep(0.5)

    assert job["status"] == "completed", job["error"]
    assert job["frames_processed"] == 30
    assert job["progress"] == 1.0

    video = client.get(job["output_video_url"])
    assert video.status_code == 200 and video.headers["content-type"] == "video/mp4"
    data = client.get(job["output_json_url"]).json()
    assert len(data["frames"]) == 30
    assert data["frames"][0]["people"], "expected the model to detect people in bus.jpg"


class FireOnceDetector:
    """Stub that reports a fall on a chosen frame, to exercise event recording."""

    def __init__(self, at_time):
        self.at_time = at_time
        self.fired = False

    def update(self, people, timestamp):
        if self.fired or timestamp < self.at_time or not people:
            return []
        self.fired = True
        return [FallDetection(track_id=people[0]["track_id"], confidence=0.9, reason="stub")]


def test_detected_fall_creates_event_with_snapshot_and_clip(client, settings):
    with Session(get_engine()) as session:
        job = Job(kind=JobKind.video, name="stub", source="stub")
        session.add(job)
        session.commit()
        session.refresh(job)
        job_id = job.id

    frame = cv2.imread(str(Path(ultralytics.__file__).parent / "assets" / "bus.jpg"))
    processor = FrameProcessor(job_id, settings, detector=FireOnceDetector(at_time=1.0))
    for index in range(30):
        processor.process(frame, index, index / 15)
    processor.finish()

    page = client.get("/api/events", params={"job_id": job_id}).json()
    assert page["total"] == 1
    event = page["items"][0]
    assert event["reason"] == "stub"
    assert event["frame_index"] == 15
    assert client.get(event["snapshot_url"]).headers["content-type"] == "image/jpeg"
    clip = client.get(event["clip_url"])
    assert clip.status_code == 200 and len(clip.content) > 1000
    assert client.get(f"/api/jobs/{job_id}").json()["fall_count"] == 1
