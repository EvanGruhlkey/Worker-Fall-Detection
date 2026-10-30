import os

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlmodel import SQLModel

from backend.config import REPO_ROOT, Settings, database_url_from_env
from backend.db import close_db, create_db_engine, init_db
from backend.main import create_app


def _test_database_url():
    """The app's database with "_test" appended, so tests never touch real data."""
    if os.environ.get("FALL_TEST_DATABASE_URL"):
        return os.environ["FALL_TEST_DATABASE_URL"]
    name = make_url(database_url_from_env()).database
    return database_url_from_env(database=f"{name}_test")


@pytest.fixture(scope="session")
def test_database_url():
    url = _test_database_url()
    try:
        init_db(url)
    except RuntimeError as error:
        pytest.exit(
            f"{error}\nCreate the test database with `uv run scripts/create_db.py`.",
            returncode=1,
        )
    close_db()
    return url


@pytest.fixture
def database_url(test_database_url):
    """The test database, emptied before each test."""
    engine = create_db_engine(test_database_url)
    tables = ", ".join(f'"{table.name}"' for table in SQLModel.metadata.sorted_tables)
    with engine.begin() as connection:
        connection.execute(text(f"TRUNCATE {tables} RESTART IDENTITY CASCADE"))
    engine.dispose()
    return test_database_url


@pytest.fixture
def settings(tmp_path, database_url):
    return Settings(
        data_dir=tmp_path,
        model_path=REPO_ROOT / "models" / "yolo11n-pose.pt",
        clip_pre_seconds=0.5,
        clip_post_seconds=0.5,
        site_location="Plant 2, Houston, TX",
        site_coords="29.7604, -95.3698",
        camera_source=None,
        reverse_geocode=False,
        database_url=database_url,
    )


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as test_client:
        yield test_client
