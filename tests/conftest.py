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
    )


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as test_client:
        yield test_client
