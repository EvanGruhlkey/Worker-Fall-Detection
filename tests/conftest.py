import pytest
from fastapi.testclient import TestClient

from backend.config import REPO_ROOT, Settings
from backend.main import create_app


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.delenv("FALL_DATABASE_URL", raising=False)
    return Settings(
        data_dir=tmp_path,
        model_path=REPO_ROOT / "models" / "yolo11n-pose.pt",
        clip_pre_seconds=0.5,
        clip_post_seconds=0.5,
        site_location="Plant 2, Houston, TX",
        site_coords="29.7604, -95.3698",
        camera_source=None,
        reverse_geocode=False,
    )


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as test_client:
        yield test_client
