"""Copy jobs and fall events from another database into the configured one.

    uv run scripts/copy_data.py data/falls.db             # the old SQLite file
    uv run scripts/copy_data.py postgresql://fall:fall@localhost:5432/fall_detection
    uv run scripts/copy_data.py data/falls.db --replace   # empty the target first

The target is the database in .env (FALL_DATABASE_URL or FALL_DB_*), e.g. the
team's shared Supabase database. Rows keep their ids, so snapshot and clip
paths still point at the same files; run scripts/sync_media.py afterwards to
upload those files to shared storage. Older databases are handled too: missing
columns get their defaults and old review statuses are renamed. The source is
only read, never changed.
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import sqlalchemy as sa
from sqlalchemy.engine import make_url
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
    parser.add_argument("source", help="SQLite file, or a postgresql:// URL")
    parser.add_argument(
        "--replace", action="store_true",
        help="delete everything already in the target before copying",
    )
    return parser.parse_args()


def source_engine(source):
    if "://" not in source:
        path = Path(source)
        if not path.is_file():
            sys.exit(f"No SQLite database at {path}")
        return sa.create_engine(f"sqlite:///file:{path.resolve().as_posix()}?mode=ro&uri=true")
    url = make_url(source)
    if url.drivername == "postgresql":
        url = url.set(drivername="postgresql+psycopg")
    return sa.create_engine(url)


def parse_time(value):
    """SQLite stored UTC times as text, usually without a timezone."""
    moment = datetime.fromisoformat(value) if isinstance(value, str) else value
    if moment is not None and moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment


def convert(model, row):
    """Turn one source row into a model instance, filling in newer columns."""
    columns = model.__table__.columns
    values = {}
    for key, value in row.items():
        if key not in columns:
            continue
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
    settings = Settings()
    source = source_engine(args.source)
    if source.url == make_url(settings.database_url):
        sys.exit("The source is the database in .env; nothing to copy.")

    engine = init_db(settings.database_url)
    target = make_url(settings.database_url).render_as_string(hide_password=True)
    try:
        existing = set(sa.inspect(source).get_table_names())
        with source.connect() as reader, Session(engine) as session:
            already = sum(session.exec(select(func.count()).select_from(m)).one() for m in MODELS)
            if already and not args.replace:
                sys.exit(
                    f"{target} already has {already} rows. Run again with --replace to "
                    "delete them and copy the source's rows in their place."
                )
            if args.replace:
                tables = ", ".join(f'"{t.name}"' for t in SQLModel.metadata.sorted_tables)
                session.connection().execute(sa.text(f"TRUNCATE {tables} RESTART IDENTITY CASCADE"))

            for model in MODELS:
                table = model.__tablename__
                if table not in existing:
                    print(f"{table}: not in the source, skipped")
                    continue
                rows = reader.execute(sa.text(f'SELECT * FROM "{table}" ORDER BY id')).mappings().all()
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
        source.dispose()
        close_db()
    print(f"Done: copied into {target}. Upload their media with `uv run scripts/sync_media.py`.")


if __name__ == "__main__":
    main()
