# 10-30-2026 - PostgreSQL database

## Goal

Replace the SQLite file (`data/falls.db`) with a real database server that can
be browsed and edited with standard tools, survives schema changes without
losing data, and can be handed over to a customer: on their own server, a
managed cloud database, or the same kind of local install we use.

How to set it up and use it: [docs/database.md](../database.md).

Later the same day the database became shared between all stations (Supabase),
and the tests moved to a bundled private PostgreSQL; see
[10-30-2026-shared-database.md](10-30-2026-shared-database.md).

## Decisions

- **PostgreSQL** rather than MySQL: permissive license (no questions from a
  customer's legal team), a real timezone-aware timestamp type, and every cloud
  provider offers it managed. The code goes through SQLModel/SQLAlchemy, so
  queries didn't change.
- **No SQLite fallback.** Tests and development use PostgreSQL too, so what we
  test is what runs. The tests use a separate `<name>_test` database, emptied
  before each test.
- **No Docker.** PostgreSQL is installed natively (it runs as a service and
  starts with the computer). Docker Desktop can't reach laptop/USB webcams on
  Windows or macOS, and needs a paid license at larger companies.
- **Alembic migrations** replace the hand-written SQLite column upgrades. The
  backend applies them on startup; `tests/test_db.py` fails if `db.py` and the
  migrations disagree.
- **Times are `timestamptz`** and connections use UTC, so values come back
  timezone-aware. The `as_utc()` workaround for SQLite is gone.
- **Enums are stored as text** (`VARCHAR(32)`), not PostgreSQL enum types, so
  adding a status later is a code change only, with no `ALTER TYPE`.
- **Media stays on disk** in `data/`; rows store relative paths. Backups
  bundle the database dump with those folders so they can't get separated.

## Changes

| File | Change |
|---|---|
| `src/backend/db.py` | PostgreSQL engine (pre-ping, UTC), timestamptz and text-enum columns, migrations on startup, `close_db()` |
| `src/backend/migrations/` | Alembic environment and the first migration (`job`, `fallevent`, `sitelocation`) |
| `alembic.ini` | Lets `uv run alembic ...` run from the project root |
| `src/backend/config.py` | `database_url` built from `FALL_DB_*` or `FALL_DATABASE_URL` |
| `run.py` | Checks the database is reachable before starting, with a hint if not |
| `scripts/create_db.py` | Creates the user, the app and test databases, and the tables |
| `scripts/sqlite_to_postgres.py` | One-time copy of an old `data/falls.db` |
| `scripts/backup.py` | `backup` / `restore`: database dump plus media in one `.zip` |
| `tests/conftest.py`, `tests/test_db.py` | Tests run against `fall_detection_test`; migration and timezone checks |
