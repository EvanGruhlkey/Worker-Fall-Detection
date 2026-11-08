import logging
import os
import tempfile
from pathlib import Path

import pixeltable_pgserver
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlmodel import SQLModel

from backend.config import REPO_ROOT, Settings
from backend.db import close_db, create_db_engine, init_db
from backend.main import create_app

# The tests never use the shared database from .env: they empty the database
# before every test. They start a private PostgreSQL (bundled with the
# pixeltable-pgserver package) whose files are kept between runs, so only the
# first run waits for it to be created.
TEST_PGDATA = Path(tempfile.gettempdir()) / "worker-fall-detection-test-pg"
# Its routine messages also run at exit, after pytest has closed the output.
logging.getLogger("pixeltable_pgserver").setLevel(logging.WARNING)


@pytest.fixture(scope="session")
def test_database_url():
    url = os.environ.get("FALL_TEST_DATABASE_URL")
    server = None
    if not url:
        server = pixeltable_pgserver.get_server(TEST_PGDATA, cleanup_mode="stop")
        url = make_url(server.get_uri()).set(drivername="postgresql+psycopg")
        url = url.render_as_string(hide_password=False)
    init_db(url)
    close_db()
    yield url
    if server:
        server.cleanup()


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
        model_path=REPO_ROOT / "models" / "pose_pipeline.json",
        clip_pre_seconds=0.5,
        clip_post_seconds=0.5,
        site_location="Plant 2, Houston, TX",
        site_coords="29.7604, -95.3698",
        camera_source=None,
        reverse_geocode=False,
        database_url=database_url,
        station_name="test-station",
        storage_bucket=None,
    )


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as test_client:
        yield test_client
