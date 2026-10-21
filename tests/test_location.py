import io
import json
import os
from pathlib import Path

import cv2
import ultralytics
from sqlmodel import Session

from backend import location
from backend.config import load_env_file
from backend.db import Job, JobKind, get_engine
from backend.fall_detector import FallDetection
from backend.pipeline import FrameProcessor

HOUSTON = {"latitude": 29.7604, "longitude": -95.3698, "accuracy": 35}


def fake_geocoder(monkeypatch, name="Houston, Texas, United States"):
    calls = []

    def geocode(latitude, longitude, timeout=5):
        calls.append((latitude, longitude))
        return name

    monkeypatch.setattr(location, "reverse_geocode", geocode)
    return calls


def test_falls_back_to_settings_until_browser_reports(client, settings):
    assert client.get("/api/location").json() == {
        "name": settings.site_location, "coords": settings.site_coords,
    }


def test_browser_location_is_named_and_shown_on_dashboard(client, settings, monkeypatch):
    settings.reverse_geocode = True
    calls = fake_geocoder(monkeypatch)

    response = client.put("/api/location", json=HOUSTON)
    assert response.status_code == 200
    assert response.json() == {"name": "Houston, Texas, United States", "coords": "29.7604, -95.3698"}
    assert calls == [(29.7604, -95.3698)]
    assert client.get("/api/dashboard").json()["location"]["name"] == "Houston, Texas, United States"


def test_small_moves_reuse_the_name(client, settings, monkeypatch):
    settings.reverse_geocode = True
    calls = fake_geocoder(monkeypatch)
    client.put("/api/location", json=HOUSTON)
    # About 100 m away: same place, no new lookup.
    client.put("/api/location", json={**HOUSTON, "latitude": 29.7613})
    assert len(calls) == 1
    # About 40 km away: look it up again.
    fake_geocoder(monkeypatch, name="Katy, Texas, United States")
    moved = client.put("/api/location", json={**HOUSTON, "longitude": -95.7600}).json()
    assert moved["name"] == "Katy, Texas, United States"


def test_failed_lookup_shows_coordinates(client, settings, monkeypatch):
    settings.reverse_geocode = True
    fake_geocoder(monkeypatch, name=None)
    body = client.put("/api/location", json=HOUSTON).json()
    assert body == {"name": "29.7604, -95.3698", "coords": "29.7604, -95.3698"}
    # The next report retries the lookup even though the laptop hasn't moved.
    fake_geocoder(monkeypatch)
    assert client.put("/api/location", json=HOUSTON).json()["name"] == "Houston, Texas, United States"


def test_geocoding_can_be_turned_off(client, monkeypatch):
    calls = fake_geocoder(monkeypatch)
    body = client.put("/api/location", json=HOUSTON).json()  # reverse_geocode=False in conftest
    assert calls == []
    assert body["name"] == "29.7604, -95.3698"


def test_invalid_coordinates_are_rejected(client):
    assert client.put("/api/location", json={"latitude": 95, "longitude": 0}).status_code == 422


class FireOnce:
    def __init__(self):
        self.fired = False

    def update(self, people, timestamp):
        if self.fired or not people:
            return []
        self.fired = True
        return [FallDetection(track_id=people[0]["track_id"], confidence=0.9, reason="stub")]


def test_new_events_use_browser_location(client, settings, monkeypatch):
    settings.reverse_geocode = True
    fake_geocoder(monkeypatch)
    client.put("/api/location", json=HOUSTON)

    with Session(get_engine()) as session:
        job = Job(kind=JobKind.camera, name="Camera 01", source="0")
        session.add(job)
        session.commit()
        session.refresh(job)
        job_id = job.id

    frame = cv2.imread(str(Path(ultralytics.__file__).parent / "assets" / "bus.jpg"))
    processor = FrameProcessor(job_id, settings, detector=FireOnce())
    for index in range(3):
        processor.process(frame, index, index / 15)
    processor.finish()

    event = client.get("/api/events", params={"job_id": job_id}).json()["items"][0]
    assert event["location"] == "Houston, Texas, United States"
    assert event["coords"] == "29.7604, -95.3698"


def test_reverse_geocode_formats_city_state_country(monkeypatch):
    payload = {"address": {"city": "Houston", "county": "Harris County", "state": "Texas",
                           "country": "United States"}}

    class FakeResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(
        location.urllib.request, "urlopen",
        lambda request, timeout: FakeResponse(json.dumps(payload).encode()),
    )
    assert location.reverse_geocode(29.76, -95.37) == "Houston, Texas, United States"

    def offline(request, timeout):
        raise OSError("no network")

    monkeypatch.setattr(location.urllib.request, "urlopen", offline)
    assert location.reverse_geocode(29.76, -95.37) is None


def test_env_file_does_not_override_shell(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# comment\nFALL_TEST_A="from file"\nFALL_TEST_B=from file\n', encoding="utf-8")
    monkeypatch.delenv("FALL_TEST_A", raising=False)
    monkeypatch.setenv("FALL_TEST_B", "from shell")
    load_env_file(env)
    assert os.environ["FALL_TEST_A"] == "from file"
    assert os.environ["FALL_TEST_B"] == "from shell"
    monkeypatch.delenv("FALL_TEST_A")
