"""Transactional local storage, isolated from production incidents."""
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path


class Store:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.media = self.directory / "media"
        self.media.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "study.sqlite3"
        with self.transaction() as db:
            db.execute("CREATE TABLE IF NOT EXISTS config (id INTEGER PRIMARY KEY CHECK(id=1), body TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, token TEXT NOT NULL, body TEXT NOT NULL)")

    @contextmanager
    def transaction(self):
        # FastAPI may enter and finalize one request on different worker threads.
        # Each transaction owns its connection; it is never shared by requests.
        db = sqlite3.connect(self.path, timeout=15, check_same_thread=False)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA secure_delete=ON")
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def save(db, session):
        db.execute("UPDATE sessions SET body=? WHERE id=?", (json.dumps(session), session["id"]))
