"""Copy jobs and fall events from the old SQLite database into PostgreSQL (run once).

    uv run scripts/sqlite_to_postgres.py                    # copies data/falls.db
    uv run scripts/sqlite_to_postgres.py path/to/falls.db
    uv run scripts/sqlite_to_postgres.py --replace          # empty PostgreSQL first

Rows keep their ids, so snapshot and clip paths (relative to the data folder)
still point at the same files in data/events/. Databases from before the
dashboard integration are handled too: missing columns get their defaults and
old review statuses are renamed. The SQLite file is only read, never changed.
"""

import argparse
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import sqlalchemy as sa
from sqlmodel import Session, SQLModel, func, select

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from backend.config import Settings  # noqa: E402
from backend.db import FallEvent, Job, SiteLocation, close_db, init_db  # noqa: E402

# Parents before children, so foreign keys are satisfied.
MODELS = (Job, FallEvent, SiteLocation)
# Review statuses used before the dashboard integration.
STATUS_RENAMES = {"pending": "unacknowledged", "confirmed": "resolved"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("sqlite_path", nargs="?", type=Path, default=ROOT / "data" / "falls.db")
    parser.add_argument(
        "--replace", action="store_true",
        help="delete everything already in PostgreSQL before copying",
    )
    return parser.parse_args()


def parse_time(value):
    """SQLite stored UTC times as text, usually without a timezone."""
    if value is None or isinstance(value, datetime):
        moment = value
    else:
        moment = datetime.fromisoformat(value)
    if moment is not None and moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment


def convert(model, row):
    """Turn one SQLite row into a model instance, filling in newer columns."""
    columns = model.__table__.columns
    values = {}
    for key in row.keys():
        if key not in columns:
            continue
        value = row[key]
        if isinstance(columns[key].type, sa.DateTime):
            value = parse_time(value)
        elif isinstance(columns[key].type, sa.Enum) and value is not None:
            value = columns[key].type.enum_class(STATUS_RENAMES.get(value, value))
        if value is None and not columns[key].nullable:
            continue  # let the model's default apply (e.g. old rows without event_type)
        values[key] = value
    return model(**values)


def main():
    args = parse_args()
    if not args.sqlite_path.is_file():
        sys.exit(f"No SQLite database at {args.sqlite_path}")

    source = sqlite3.connect(f"file:{args.sqlite_path.as_posix()}?mode=ro", uri=True)
    source.row_factory = sqlite3.Row
    existing = {name for (name,) in source.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}

    engine = init_db(Settings().database_url)
    try:
        with Session(engine) as session:
            already = sum(session.exec(select(func.count()).select_from(m)).one() for m in MODELS)
            if already and not args.replace:
                sys.exit(
                    f"PostgreSQL already has {already} rows. Run again with --replace to "
                    "delete them and copy the SQLite data in their place."
                )
            if args.replace:
                tables = ", ".join(f'"{t.name}"' for t in SQLModel.metadata.sorted_tables)
                session.connection().execute(sa.text(f"TRUNCATE {tables} RESTART IDENTITY CASCADE"))

            for model in MODELS:
                table = model.__tablename__
                if table not in existing:
                    print(f"{table}: not in the SQLite file, skipped")
                    continue
                rows = source.execute(f'SELECT * FROM "{table}" ORDER BY id').fetchall()
                session.add_all(convert(model, row) for row in rows)
                session.flush()
                # Rows were inserted with their ids; move the id counter past them.
                session.connection().execute(sa.text(
                    f"SELECT setval(pg_get_serial_sequence('{table}', 'id'), "
                    f"COALESCE((SELECT MAX(id) FROM \"{table}\"), 0) + 1, false)"
                ))
                print(f"{table}: copied {len(rows)} rows")
            session.commit()
    finally:
        source.close()
        close_db()
    print(
        "Done. Snapshots and clips stay in data/events/; the copied rows point at them. "
        "The SQLite file is no longer used and can be deleted once you've checked the dashboard."
    )


if __name__ == "__main__":
    main()
