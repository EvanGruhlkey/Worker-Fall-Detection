# 10-30-2026 - Shared database and media storage (Supabase)

## Goal

Every teammate running the project should see the same falls. With a database
on each laptop (SQLite before, then a local PostgreSQL), each person only saw
the falls their own laptop recorded. Now every station (a computer running the
backend) records into one hosted PostgreSQL database and uploads its footage
to one storage bucket, both on a free Supabase project.

How to set it up and use it: [docs/database.md](../database.md). This builds
on [10-30-2026-postgres-database.md](10-30-2026-postgres-database.md).

## Decisions

- **Supabase** (hosted PostgreSQL plus S3-compatible storage, one account, free
  plan) over AWS (RDS and S3, about $20 a month after the credits run out, and
  more setup). The code uses plain PostgreSQL and the S3 protocol, so moving to
  AWS, the customer's own cloud, or on-site servers is a change to `.env`.
- **Each station still runs its own backend** (camera, AI, dashboard). Only
  the records and footage are shared.
- **Local first, then upload.** Media is written to `data/` and uploaded in a
  background thread, so the camera never waits for the network. Anything that
  failed to upload goes up on the next start (`MediaStore.sync`). A file a
  station doesn't have is served as a redirect to a one-hour signed bucket
  link, so footage never passes through another station.
- **Uploaded source videos stay local** (`data/input/`): they can be large, and
  only the annotated output is needed for review. This keeps within the 1 GB
  free storage.
- **Stations.** A shared database broke three single-laptop assumptions:
  - Startup recovery marked *every* running job as interrupted, which would
    have stopped teammates' live cameras on the dashboard. It now only
    recovers this station's jobs (and old untagged ones).
  - The site location was a single row. It's now one row per station.
  - Two stations could run the same migration at once. Migrations now take a
    PostgreSQL advisory lock.
- **Row level security on every table** (migration 0003). Supabase's web API
  serves `public` tables to anyone with the project's public key unless RLS is
  on. The backend connects as the tables' owner, which RLS doesn't apply to.
- **Small connection pools** (3, plus 2 extra): Supabase's free pooler allows
  15 connections across all stations. Prepared statements are off, so the
  transaction-mode pooler also works.
- **Tests use a private, bundled PostgreSQL** (`pixeltable-pgserver`), never
  the shared database, which they would empty. Nobody has to install
  PostgreSQL for anything. The same package provides `pg_dump`, `pg_restore`
  and `psql` (version 18) for backups.

## Changes

| File | Change |
|---|---|
| `src/backend/storage.py` | New. `MediaStore`: background uploads, startup sync, signed links |
| `src/backend/deps.py` | Files are served from `data/` if present, else redirected to the bucket |
| `src/backend/pipeline.py` | Snapshots, clips and annotated videos are uploaded once written |
| `src/backend/db.py` | `station` columns, migration lock, pool limits |
| `src/backend/migrations/versions/…0002…`, `…0003…` | Station columns; row level security |
| `src/backend/manager.py`, `routers/jobs.py`, `location.py` | Jobs tagged with their station; recovery and location per station |
| `src/backend/config.py` | `FALL_STATION_NAME`, `FALL_STORAGE_*` |
| `scripts/copy_data.py` | Was `sqlite_to_postgres.py`; now also copies from another PostgreSQL |
| `scripts/sync_media.py` | New. Uploads missing media and checks the storage settings |
| `scripts/backup.py` | Bundled PostgreSQL 18 tools; includes bucket-only media; restores into any PostgreSQL version; uploads restored media |
| `scripts/create_db.py` | Only for a self-hosted server now; no test database |
| `tests/conftest.py` | Private bundled PostgreSQL for the tests |
| `tests/test_shared.py` | New. Station recovery and location, uploads, redirects, sync |

## Known limitations

- **No internet, no recording.** A station that loses its connection keeps
  its media but can't write fall events until the connection is back. Before
  handover, decide whether the customer needs an offline buffer.
- **No login.** Anyone who can open a station's dashboard can see and review
  every station's falls.
- **Free-plan pause.** Supabase pauses a free project after a week without
  activity; it has to be restored from the Supabase website.
