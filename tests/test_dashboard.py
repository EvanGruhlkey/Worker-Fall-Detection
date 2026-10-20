import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest
import ultralytics
from sqlmodel import Session

from backend.db import FallEvent, Job, JobKind, JobStatus, ReviewStatus, get_engine, init_db
from backend.manager import JobManager
from backend.pipeline import assess_frame
from backend.routers.dashboard import uptime_ratio

# Keys of FallEvent, Stat and CameraInfo in ui/src/types.ts.
UI_EVENT_KEYS = {"id", "time", "name", "icon", "color", "confidence", "status"}
UI_STAT_KEYS = {"label", "value", "note", "tone", "icon"}
UI_CAMERA_KEYS = {"id", "name", "label", "fps", "online"}


def add_job(**fields):
    fields = {"kind": JobKind.camera, "name": "Camera 0", "source": "0", **fields}
    with Session(get_engine()) as session:
        job = Job(**fields)
        session.add(job)
        session.commit()
        session.refresh(job)
        return job


def add_event(job_id, **fields):
    fields = {
        "track_id": 3, "frame_index": 90, "video_time": 3.0, "confidence": 0.82,
        "reason": "test", "camera_name": "Camera 0", "location": "Bay 3",
        "coords": "1, 2", **fields,
    }
    with Session(get_engine()) as session:
        event = FallEvent(job_id=job_id, **fields)
        session.add(event)
        session.commit()
        session.refresh(event)
        return event


class FakeWorker:
    def __init__(self, source="0", fps=14.6, health="good"):
        self.source, self.fps, self.health = source, fps, health


@pytest.fixture
def live_camera(client, monkeypatch):
    """Pretend a camera job is running without opening a real camera."""
    job = add_job(
        status=JobStatus.running, label="Loading dock", location="Dock A", coords="3, 4",
        started_at=datetime.now(timezone.utc) - timedelta(minutes=30),
    )
    worker = FakeWorker()
    monkeypatch.setattr(
        client.app.state.manager, "active_cameras", lambda: {job.id: worker}
    )
    return job, worker


def test_event_matches_ui_fall_event_type(client):
    job = add_job()
    add_event(job.id)

    events = client.get("/api/dashboard").json()["events"]
    assert len(events) == 1
    event = events[0]
    assert UI_EVENT_KEYS <= event.keys()
    assert event["name"] == "Fall detected"
    assert event["icon"] == "fall"
    assert event["color"] == "var(--red)"
    assert event["confidence"] == 82
    assert event["status"] == "Unacknowledged"
    assert re.fullmatch(r"\d{1,2}:\d{2}:\d{2} [AP]M", event["time"])
    assert event["camera"] == "Camera 0" and event["location"] == "Bay 3"


def test_false_alarm_is_shown_as_warning(client):
    job = add_job()
    add_event(job.id, status=ReviewStatus.false_alarm)
    event = client.get("/api/dashboard").json()["events"][0]
    assert event["status"] == "False alarm"
    assert event["icon"] == "warn" and event["color"] == "var(--amber)"


def test_today_scope_hides_older_events(client):
    job = add_job()
    add_event(job.id, reason="today")
    add_event(job.id, reason="old", detected_at=datetime.now(timezone.utc) - timedelta(days=2))

    today = client.get("/api/dashboard").json()["events"]
    assert [e["reason"] for e in today] == ["today"]
    everything = client.get("/api/dashboard", params={"scope": "all"}).json()["events"]
    assert len(everything) == 2


def test_status_filter(client):
    job = add_job()
    add_event(job.id)
    add_event(job.id, status=ReviewStatus.resolved)
    events = client.get("/api/dashboard", params={"status": "resolved"}).json()["events"]
    assert [e["status"] for e in events] == ["Resolved"]


def test_stats_with_no_cameras(client, settings):
    body = client.get("/api/dashboard").json()
    stats = {s["label"]: s for s in body["stats"]}
    assert list(stats) == ["Active incidents", "Cameras online", "Detection uptime", "Camera health"]
    assert all(UI_STAT_KEYS <= s.keys() for s in stats.values())
    assert stats["Active incidents"]["value"] == "0"
    assert stats["Active incidents"]["tone"] == "green"
    assert stats["Cameras online"]["value"] == "0"
    assert stats["Detection uptime"]["value"] == "-"
    assert stats["Camera health"]["value"] == "Offline"

    assert UI_CAMERA_KEYS <= body["camera"].keys()
    assert body["camera"]["online"] is False
    assert body["location"] == {"name": settings.site_location, "coords": settings.site_coords}


def test_active_incidents_count_open_events(client):
    job = add_job()
    add_event(job.id)
    add_event(job.id, status=ReviewStatus.needs_review)
    add_event(job.id, status=ReviewStatus.resolved)
    incidents = client.get("/api/dashboard").json()["stats"][0]
    assert incidents["value"] == "2"
    assert incidents["tone"] == "red"
    assert incidents["note"] == "1 unacknowledged · 1 need review"


def test_live_camera_fills_camera_stats_and_location(client, live_camera):
    job, worker = live_camera
    body = client.get("/api/dashboard").json()
    stats = {s["label"]: s for s in body["stats"]}

    assert stats["Cameras online"]["value"] == "1"
    assert stats["Cameras online"]["tone"] == "green"
    assert stats["Detection uptime"]["value"] == "100.0%"
    assert stats["Camera health"]["value"] == "Good"

    camera = body["camera"]
    assert camera["id"] == "CAM-00"
    assert camera["label"] == "Loading dock"
    assert camera["fps"] == 15
    assert camera["online"] is True
    assert camera["stream_url"] == f"/api/cameras/{job.id}/stream"
    assert body["location"] == {"name": "Dock A", "coords": "3, 4"}

    worker.health = "obstructed"
    health = client.get("/api/dashboard").json()["stats"][3]
    assert health["value"] == "Obstructed" and health["tone"] == "red"


def test_uptime_counts_gaps_between_sessions(client):
    now = datetime.now(timezone.utc)
    add_job(
        status=JobStatus.stopped,
        started_at=now - timedelta(hours=10), finished_at=now - timedelta(hours=5),
    )
    uptime = client.get("/api/dashboard").json()["stats"][2]
    assert uptime["value"] == "50.0%"
    assert uptime["tone"] == "red"
    assert uptime["note"] == "Last 10.0 hours"


def test_uptime_ratio_merges_overlaps():
    t = datetime(2026, 1, 1, tzinfo=timezone.utc)
    h = timedelta(hours=1)
    assert uptime_ratio([(t, t + 2 * h), (t + h, t + 3 * h)], t, t + 4 * h) == 0.75
    assert uptime_ratio([(t - h, t + h)], t, t + 2 * h) == 0.5
    assert uptime_ratio([], t, t + h) == 0
    assert uptime_ratio([], t, t) is None


def test_review_accepts_dashboard_labels(client):
    job = add_job()
    event = add_event(job.id)
    response = client.patch(f"/api/events/{event.id}", json={"status": "Needs review"})
    assert response.status_code == 200
    assert response.json()["status"] == "needs_review"
    assert client.get("/api/dashboard").json()["events"][0]["status"] == "Needs review"


def test_csv_export(client):
    job = add_job()
    add_event(job.id, reviewer="Sam", status=ReviewStatus.resolved)
    response = client.get("/api/dashboard/events.csv")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "attachment" in response.headers["content-disposition"]
    header, row = response.text.strip().splitlines()
    assert header.startswith("id,detected_at_utc,local_time,event,camera,location")
    assert "Fall detected" in row and "Bay 3" in row and ",82," in row and "Resolved" in row


def test_camera_start_stores_label_and_location(client, monkeypatch):
    manager = client.app.state.manager
    monkeypatch.setattr(manager, "start_camera", lambda *args, **kwargs: None)
    response = client.post(
        "/api/cameras",
        json={
            "source": "1", "name": "Bay 3", "label": "Forklift lane",
            "location": "Plant 2", "coords": "5, 6",
        },
    )
    assert response.status_code == 201
    job = response.json()
    assert (job["name"], job["label"], job["location"], job["coords"]) == (
        "Bay 3", "Forklift lane", "Plant 2", "5, 6"
    )


def test_recovery_uses_last_frame_time(client, settings):
    last_frame = datetime.now(timezone.utc) - timedelta(hours=2)
    job = add_job(
        status=JobStatus.running, started_at=last_frame - timedelta(hours=1),
        last_frame_at=last_frame,
    )
    JobManager(settings).recover_interrupted_jobs()
    with Session(get_engine()) as session:
        recovered = session.get(Job, job.id)
    assert recovered.status == JobStatus.failed
    assert recovered.finished_at.replace(tzinfo=None) == last_frame.replace(tzinfo=None)


def test_old_database_is_migrated(tmp_path):
    path = tmp_path / "old.db"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE job (id INTEGER PRIMARY KEY, kind VARCHAR, name VARCHAR, source VARCHAR,
            status VARCHAR, frames_processed INTEGER, total_frames INTEGER, fps FLOAT,
            fall_count INTEGER, error VARCHAR, output_video VARCHAR, output_json VARCHAR,
            created_at DATETIME, started_at DATETIME, finished_at DATETIME);
        CREATE TABLE fallevent (id INTEGER PRIMARY KEY, job_id INTEGER REFERENCES job(id),
            track_id INTEGER, frame_index INTEGER, video_time FLOAT, detected_at DATETIME,
            confidence FLOAT, reason VARCHAR, status VARCHAR, reviewer VARCHAR, notes VARCHAR,
            reviewed_at DATETIME, snapshot_path VARCHAR, clip_path VARCHAR);
        INSERT INTO job VALUES (1, 'video', 'a.mp4', 'a.mp4', 'completed', 0, NULL, NULL, 2,
            NULL, NULL, NULL, '2026-10-18 10:00:00', NULL, NULL);
        INSERT INTO fallevent VALUES (1, 1, 3, 10, 1.0, '2026-10-18 10:00:01', 0.9, 'x',
            'pending', NULL, NULL, NULL, NULL, NULL);
        INSERT INTO fallevent VALUES (2, 1, 3, 20, 2.0, '2026-10-18 10:00:02', 0.9, 'x',
            'confirmed', NULL, NULL, NULL, NULL, NULL);
        """
    )
    connection.commit()
    connection.close()

    init_db(f"sqlite:///{path.as_posix()}")
    with Session(get_engine()) as session:
        first, second = session.get(FallEvent, 1), session.get(FallEvent, 2)
        assert first.status == ReviewStatus.unacknowledged
        assert second.status == ReviewStatus.resolved
        assert first.event_type == "fall"
        assert first.location is None
        assert session.get(Job, 1).label is None


def test_assess_frame():
    photo = cv2.imread(str(Path(ultralytics.__file__).parent / "assets" / "bus.jpg"))
    assert assess_frame(photo) == "good"
    assert assess_frame(cv2.GaussianBlur(photo, (51, 51), 0)) == "blurry"
    assert assess_frame(np.full((480, 640, 3), 20, np.uint8)) == "obstructed"
