# Database

The backend stores every processing run and every detected fall in a
**PostgreSQL** database. This guide covers setting it up, looking at and
editing the data, changing the tables, backing up, and moving the system to
another machine or to a customer's own database server.

- [What is stored where](#what-is-stored-where)
- [First-time setup](#first-time-setup)
- [Copying data from the old SQLite file](#copying-data-from-the-old-sqlite-file)
- [Viewing and editing the data](#viewing-and-editing-the-data)
- [Changing the tables (migrations)](#changing-the-tables-migrations)
- [Backup and restore](#backup-and-restore)
- [Handing over or moving the system](#handing-over-or-moving-the-system)
- [Troubleshooting](#troubleshooting)

---

## What is stored where

| Where | What |
|---|---|
| PostgreSQL database `fall_detection` | The tables below |
| `data/events/` | Snapshot (`.jpg`) and review clip (`.mp4`) of each fall |
| `data/input/` | Uploaded videos |
| `data/output/` | Annotated videos and pose JSON from analyzed uploads |

Images and videos stay as files; the database holds their paths, relative to
the `data/` folder. **The database and the `data/` folder belong together**:
back up and move both (the backup script does).

| Table | One row per | Main columns |
|---|---|---|
| `job` | Uploaded video or live camera session | `kind` (video/camera), `name`, `label`, `location`, `status`, `fall_count`, `started_at`, `last_frame_at`, `finished_at` |
| `fallevent` | Detected fall | `job_id`, `detected_at`, `camera_name`, `location`, `coords`, `confidence`, `reason`, `status` (unacknowledged / needs_review / resolved / false_alarm), `reviewer`, `notes`, `reviewed_at`, `snapshot_path`, `clip_path` |
| `sitelocation` | (single row) | Last location the dashboard's browser reported |
| `alembic_version` | (single row) | Which migration the tables are at. Don't edit it. |

All times are stored as `timestamptz` (with timezone) in UTC. The table
definitions are in `src/backend/db.py`.

---

## First-time setup

Do this once on each computer that runs the backend.

### 1. Install PostgreSQL

Any supported version works (14 or newer); we use 17.

- **Windows:** the installer from https://www.postgresql.org/download/windows/.
  Keep the defaults (port 5432), and **write down the password you choose for
  the `postgres` user**. Keep pgAdmin selected; it's a program for browsing the data.
  PostgreSQL then runs as a Windows service and starts with the computer.
- **macOS:** [Postgres.app](https://postgresapp.com/), or `brew install postgresql@17 && brew services start postgresql@17`.
  With Homebrew the admin user is your macOS user name with no password, so run step 3 with
  `--admin-user $(whoami)` and press Enter at the password prompt.
- **Linux (Ubuntu/Debian):** `sudo apt install postgresql`, then set an admin password with
  `sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'choose-one';"`.

### 2. Check the settings in `.env`

The backend connects with these (defaults shown, from `.env.example`):

| Setting | Default | Meaning |
|---|---|---|
| `FALL_DB_HOST` | `localhost` | Database server |
| `FALL_DB_PORT` | `5432` | Its port |
| `FALL_DB_NAME` | `fall_detection` | Database name (tests use `fall_detection_test`) |
| `FALL_DB_USER` | `fall` | User the backend logs in as |
| `FALL_DB_PASSWORD` | `fall` | Its password. **Change it on anything but a development laptop.** |
| `FALL_DATABASE_URL` | empty | A full URL instead of the five settings above, e.g. `postgresql+psycopg://user:pass@db.example.com:5432/fall_detection`. Use it for a managed or remote database. |

On a development laptop the defaults are fine: PostgreSQL only accepts
connections from the same computer unless you change its configuration.

### 3. Create the user and databases

```bash
uv run scripts/create_db.py
```

It asks for the `postgres` admin password from step 1 (it isn't saved), then
creates the `fall` user, the `fall_detection` and `fall_detection_test`
databases, and the tables.

**Expected:**

```
Created user 'fall'.
Created database 'fall_detection'.
Created database 'fall_detection_test'.
Tables are up to date. The backend will use fall@localhost:5432/fall_detection.
```

It's safe to run again; anything that exists is left alone. Then start the
system as usual with `uv run run.py`.

---

## Copying data from the old SQLite file

Before this change, events were stored in `data/falls.db` (SQLite). To bring
them into PostgreSQL, once:

```bash
uv run scripts/sqlite_to_postgres.py
```

**Expected:** `job: copied N rows`, `fallevent: copied N rows`, and so on.
Rows keep their ids, so their snapshots and clips in `data/events/` still match.
It refuses to run if PostgreSQL already has data; add `--replace` to delete
that data and copy the SQLite rows instead. After checking the dashboard,
`data/falls.db*` can be deleted.

---

## Viewing and editing the data

Use any PostgreSQL client and connect with the settings from `.env`
(host `localhost`, port `5432`, database `fall_detection`, user `fall`).

- **pgAdmin** (installed with PostgreSQL on Windows): add a server with those
  settings, then open *Databases → fall_detection → Schemas → public → Tables*.
  Right-click a table → *View/Edit Data* to see and edit rows.
- **DBeaver** (free, all platforms) or the **SQLTools** extension in VS Code also work.
- **psql** on the command line:

  ```bash
  psql -h localhost -U fall -d fall_detection
  ```

  ```sql
  -- Today's falls, newest first
  SELECT id, detected_at, camera_name, location, confidence, status
  FROM fallevent
  WHERE detected_at >= date_trunc('day', now())
  ORDER BY detected_at DESC;

  -- Mark one as a false alarm
  UPDATE fallevent SET status = 'false_alarm', reviewer = 'Sam', reviewed_at = now() WHERE id = 12;
  ```

Edits show up in the dashboard on its next refresh (a few seconds). Keep
`status` to one of the four values above, and don't change `id`, `job_id` or
the `*_path` columns by hand.

---

## Changing the tables (migrations)

The tables are created and updated by **Alembic migrations** in
`src/backend/migrations/versions/`. The backend applies any new migration when
it starts, so every database (yours, a teammate's, the customer's) ends up
with the same tables without losing data.

To add or change a column:

1. Edit the table class in `src/backend/db.py`.
2. Generate a migration from the difference:

   ```bash
   uv run alembic revision --autogenerate -m "add camera zone to job"
   ```

3. **Read the new file** in `src/backend/migrations/versions/`. Alembic
   guesses: a renamed column appears as a drop and an add (which would lose
   that column's data), so rewrite it as `op.alter_column(..., new_column_name=...)`.
   A new column that isn't nullable needs a `server_default` so existing rows get a value.
4. Apply it (`uv run alembic upgrade head`, or just restart the backend), and
   run `uv run pytest`. `tests/test_db.py::test_migrations_match_models`
   fails if `db.py` and the migrations disagree.
5. Commit the migration file with the `db.py` change.

Never edit a migration that has already been committed and run elsewhere;
add a new one instead.

---

## Backup and restore

```bash
uv run scripts/backup.py backup
```

Writes `backups/fall-backup-<date>-<time>.zip` containing `database.dump`
(made with PostgreSQL's `pg_dump`) and the `events/`, `input/` and `output/`
folders. `--no-videos` leaves out the uploaded and annotated videos (snapshots
and review clips are always included). The `backups/` folder isn't committed
to git; copy backups somewhere safe.

```bash
uv run scripts/backup.py restore backups/fall-backup-20261030-120000.zip
```

**Replaces everything** in the configured database with the backup's contents
and copies its files into `data/`. Stop the backend first. It asks before
doing anything (`--yes` skips the question). A backup from an older version
is upgraded to the current tables afterwards.

Both commands need `pg_dump` / `pg_restore`, which come with PostgreSQL. On
Windows the script finds them in `C:\Program Files\PostgreSQL\<version>\bin`;
elsewhere they must be on the PATH. They must be at least as new as the server.

To back up automatically, schedule `uv run scripts/backup.py backup` (Windows
Task Scheduler, or cron on Linux/macOS).

---

## Handing over or moving the system

The backend only needs a reachable PostgreSQL database, so the customer can
run it in whichever of these ways suits them:

| Option | How |
|---|---|
| **Same machine, local PostgreSQL** (like our laptops) | Install PostgreSQL, set a strong `FALL_DB_PASSWORD` in `.env`, run `scripts/create_db.py`. |
| **Their existing PostgreSQL server** | Their database administrator creates a user and an empty database owned by that user. Put the connection in `FALL_DATABASE_URL`. The backend creates the tables on first start. |
| **Managed cloud database** (AWS RDS, Azure Database for PostgreSQL, Google Cloud SQL) | Same as above with the cloud database's address. Add `?sslmode=require` to the URL if the provider requires encrypted connections. |

To move the data along with it:

1. On the old machine: `uv run scripts/backup.py backup`.
2. Copy the `.zip` to the new machine.
3. On the new machine, set up the database as in the table above, then:
   `uv run scripts/backup.py restore <file>.zip`.

`pg_dump` output restores into any PostgreSQL of the same or a newer version,
so the data can also be loaded by the customer's own tools without these scripts.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `run.py` says `Could not connect to the database ... Connection refused` | PostgreSQL isn't running. Windows: open *Services*, start `postgresql-x64-17`. macOS (Homebrew): `brew services start postgresql@17`. Linux: `sudo systemctl start postgresql`. |
| `password authentication failed for user "fall"` | `FALL_DB_PASSWORD` in `.env` doesn't match the user's password. Fix `.env`, or change the password in psql as `postgres`: `ALTER USER fall PASSWORD 'new';` |
| `database "fall_detection" does not exist` | Run `uv run scripts/create_db.py`. |
| `pytest` stops with `Could not connect to the database` | Tests need the `fall_detection_test` database: run `uv run scripts/create_db.py`. |
| `create_db.py` says `password authentication failed for user "postgres"` | That's the admin password from installing PostgreSQL. If it's lost, see "resetting the postgres password" in PostgreSQL's docs for your platform. |
| `pg_dump: error: aborting because of server version mismatch` | The `pg_dump` found is older than the server. Put the newer PostgreSQL's `bin` folder first on the PATH. |
