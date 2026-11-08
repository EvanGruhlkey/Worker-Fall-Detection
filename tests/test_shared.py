"""Several stations sharing one database and one media bucket."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

from sqlmodel import Session

from backend.db import FallEvent, Job, JobKind, JobStatus, get_engine
from backend.location import current_site_location
from backend.manager import JobManager
from backend.storage import get_media, init_media, share


class FakeS3:
    """The few S3 client calls MediaStore makes, kept in memory."""

    def __init__(self, keys=()):
        self.objects = {key: {} for key in keys}

    def upload_file(self, filename, bucket, key, ExtraArgs=None):
        self.objects[key] = {"bucket": bucket, "filename": filename, **(ExtraArgs or {})}

    def get_paginator(self, _name):
        objects = self.objects

        class Paginator:
            def paginate(self, Bucket):
                yield {"Contents": [{"Key": key} for key in objects]}

        return Paginator()

    def generate_presigned_url(self, _operation, Params, ExpiresIn):
        return f"https://storage.example/{Params['Bucket']}/{Params['Key']}?expires={ExpiresIn}"


def add_job(**fields):
    with Session(get_engine()) as session:
        job = Job(kind=JobKind.camera, name="Camera 0", source="0", **fields)
        session.add(job)
        session.commit()
        session.refresh(job)
        return job


def use_bucket(settings, fake):
    """Swap the app's local-only storage for a bucket backed by `fake`."""
    get_media().shutdown()
    return init_media(replace(settings, storage_bucket="falls"), client=fake)


def test_recovery_leaves_other_stations_jobs_running(client, settings):
    started = datetime.now(timezone.utc) - timedelta(minutes=5)
    mine = add_job(station="test-station", status=JobStatus.running, started_at=started)
    legacy = add_job(station=None, status=JobStatus.running, started_at=started)
    theirs = add_job(station="other-laptop", status=JobStatus.running, started_at=started)

    JobManager(settings).recover_interrupted_jobs()

    with Session(get_engine()) as session:
        assert session.get(Job, mine.id).status == JobStatus.failed
        assert session.get(Job, legacy.id).status == JobStatus.failed
        assert session.get(Job, theirs.id).status == JobStatus.running


def test_each_station_keeps_its_own_location(client, settings):
    client.put("/api/location", json={"latitude": 29.7604, "longitude": -95.3698})
    assert current_site_location(settings) == ("29.7604, -95.3698", "29.7604, -95.3698")

    other = replace(settings, station_name="other-laptop")
    assert current_site_location(other) == (settings.site_location, settings.site_coords)

    client.put("/api/location", json={"latitude": 40.0, "longitude": -75.0})
    assert current_site_location(settings)[1] == "40.0000, -75.0000"


def test_new_media_is_uploaded_with_its_type(client, settings):
    fake = FakeS3()
    media = use_bucket(settings, fake)
    snapshot = settings.events_dir / "1" / "event_1.jpg"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(b"jpeg")

    share("events/1/event_1.jpg")
    media.shutdown()  # waits for the background upload

    assert fake.objects["events/1/event_1.jpg"]["ContentType"] == "image/jpeg"
    assert fake.objects["events/1/event_1.jpg"]["bucket"] == "falls"


def test_media_from_another_station_redirects_to_bucket(client, settings):
    use_bucket(settings, FakeS3())
    job = add_job(station="other-laptop")
    with Session(get_engine()) as session:
        event = FallEvent(
            job_id=job.id, frame_index=1, video_time=0.1, confidence=0.9, reason="test",
            snapshot_path="events/9/event_9.jpg", clip_path="events/9/event_9.mp4",
        )
        session.add(event)
        session.commit()
        event_id = event.id

    # Not on this station's disk, so the browser is sent to the bucket.
    response = client.get(f"/api/events/{event_id}/clip", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"].startswith("https://storage.example/falls/events/9/event_9.mp4")

    # A file this station has is served directly.
    local = settings.events_dir / "9" / "event_9.jpg"
    local.parent.mkdir(parents=True)
    local.write_bytes(b"jpeg")
    response = client.get(f"/api/events/{event_id}/snapshot", follow_redirects=False)
    assert response.status_code == 200
    assert response.content == b"jpeg"


def test_without_bucket_missing_media_is_404(client):
    job = add_job()
    with Session(get_engine()) as session:
        event = FallEvent(
            job_id=job.id, frame_index=1, video_time=0.1, confidence=0.9, reason="test",
            snapshot_path="events/9/event_9.jpg",
        )
        session.add(event)
        session.commit()
        event_id = event.id
    assert client.get(f"/api/events/{event_id}/snapshot").status_code == 404


def test_sync_uploads_only_what_the_bucket_lacks(client, settings):
    for name in ("events/1/event_1.jpg", "events/1/event_1.mp4", "output/job_1_a.mp4", "input/a.mp4"):
        path = settings.data_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    fake = FakeS3(keys=["events/1/event_1.jpg"])
    media = use_bucket(settings, fake)

    assert media.sync() == 2
    # Uploaded source videos stay on the station that processed them.
    assert set(fake.objects) == {"events/1/event_1.jpg", "events/1/event_1.mp4", "output/job_1_a.mp4"}
