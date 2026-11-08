# Database and shared storage

Every computer that runs the backend (a **station**) records falls into one
shared **PostgreSQL** database, and uploads the snapshots, clips and annotated
videos to one shared **storage bucket**. So every station's dashboard shows
every fall, with its footage, no matter which laptop recorded it. The team uses
a free [Supabase](https://supabase.com) project for both.

- [How it fits together](#how-it-fits-together)
- [Setting up Supabase (once, for the whole team)](#setting-up-supabase-once-for-the-whole-team)
- [Setting up a station (each laptop)](#setting-up-a-station-each-laptop)
- [Copying in existing data](#copying-in-existing-data)
- [Viewing and editing the data](#viewing-and-editing-the-data)
- [Changing the tables (migrations)](#changing-the-tables-migrations)
- [Backup and restore](#backup-and-restore)
- [Handing over to the customer](#handing-over-to-the-customer)
- [Troubleshooting](#troubleshooting)

---

## How it fits together

```
 Station A ──┐                                  ┌── PostgreSQL: jobs, fall events, locations
 Station B ──┼── each runs its own backend ─────┤
 Station C ──┘   (camera + AI + dashboard)      └── Storage bucket: snapshots, clips, videos
```

| Where | What |
|---|---|
| Database table `job` | One row per uploaded video or live camera session, tagged with its `station` |
| Database table `fallevent` | One row per detected fall: time, camera, location, confidence, reason, review status, reviewer, notes, and the paths of its snapshot and clip |
| Database table `sitelocation` | Each station's last location, as reported by its dashboard's browser |
| Database table `alembic_version` | Which migration the tables are at. Don't edit it. |
| Bucket `events/<job id>/` | Snapshot (`.jpg`) and review clip (`.mp4`) of each fall |
| Bucket `output/` | Annotated videos and pose JSON from analyzed uploads |
| Station's `data/input/` | Uploaded source videos. These stay on the station that analyzed them. |

How the pieces behave:

- A station writes each file to its own `data/` folder first, then uploads it
  in the background, so the camera never waits for the network.
- When a dashboard asks for a file the station doesn't have, the backend sends
  the browser a link to the bucket that is valid for an hour.
- Files that couldn't be uploaded (no internet) stay in `data/` and are
  uploaded the next time that station's backend starts.
- **The database needs the internet.** If the connection drops, falls can't be
  recorded until it's back. The station only keeps its media files.
- Each station is named by `FALL_STATION_NAME` (default: the computer's
  network name). When a backend restarts it only marks *its own* unfinished
  jobs as interrupted, so it never touches a teammate's running camera.
- All times are stored with their timezone, in UTC. The tables are defined in
  `src/backend/db.py`, and the storage code is in `src/backend/storage.py`.

---

## Setting up Supabase (once, for the whole team)

One person does this. Use a **shared team email account**, not a personal
one, so the project can be handed over later. Supabase's page names may
differ slightly from these steps.

1. **Create the project.** Sign up at https://supabase.com (Free plan), then
   choose **New project**. Pick the region closest to the team. Click
   **Generate a password**, and **save it in your password manager**: it's the
   database password. Generated passwords contain only letters and numbers,
   which keeps the connection URL simple.
2. **Copy the database connection.** Click **Connect** at the top of the
   project, then **Connection string** → **URI**, under **Session pooler**.
   It looks like:

   ```
   postgresql://postgres.abcdefghij:[YOUR-PASSWORD]@aws-0-us-east-1.pooler.supabase.com:5432/postgres
   ```

   Use the **Session pooler**, not **Direct connection**. The direct one only
   works on networks with IPv6, which many home and campus networks lack.
   Replace `[YOUR-PASSWORD]` with the password from step 1, and change the
   start from `postgresql://` to `postgresql+psycopg://`. That's `FALL_DATABASE_URL`.
3. **Create the bucket.** Go to **Storage** → **New bucket**. Name it, for
   example, `fall-media`. Leave **Public bucket off**: footage is only ever
   handed out through short-lived links.
4. **Create the storage keys.** Go to **Storage** → **S3 Connection**
   (Project Settings → Storage on some versions):
   - Turn on the S3 connection if it's off.
   - Copy the **Endpoint** (`https://abcdefghij.supabase.co/storage/v1/s3`) and the **Region**.
   - Under **S3 Access Keys**, choose **New access key** and copy the
     **Access key ID** and **Secret access key**. The secret is shown only once.
5. **Share the values** with teammates through a private channel (a password
   manager's shared vault is best). They go in each station's `.env` (below).
   **Never commit them to git**, and don't paste them into issues or chats.

The first station to start creates the tables. There is nothing to set up
inside the database by hand.

**Free plan limits:** 500 MB database, 1 GB storage. Fall events are tiny; the
storage fills mainly with clips (about 1–2 MB each) and annotated videos.
Supabase **pauses a free project after a week with no activity**; open the
project's page and click **Restore** to wake it up.

---

## Setting up a station (each laptop)

1. `uv sync` (installs everything, including the PostgreSQL tools used for
   backups and tests).
2. In `.env` in the project root (created from `.env.example` by the first
   `uv run run.py`), fill in the values from the Supabase setup:

   | Setting | Value |
   |---|---|
   | `FALL_DATABASE_URL` | The Session pooler URL from step 2 |
   | `FALL_STORAGE_ENDPOINT` | The S3 endpoint |
   | `FALL_STORAGE_REGION` | The S3 region, e.g. `us-east-1` |
   | `FALL_STORAGE_BUCKET` | The bucket name, e.g. `fall-media` |
   | `FALL_STORAGE_KEY_ID` / `FALL_STORAGE_SECRET` | The S3 access key |
   | `FALL_STATION_NAME` | Optional: a readable name for this laptop, e.g. `alex-laptop`. Must be different on every station. |

   An `.env` from before these settings existed doesn't have the lines yet;
   copy them from `.env.example`.
3. Check the storage settings: `uv run scripts/sync_media.py`.
   **Expected:** `Bucket 'fall-media' has N files.`
4. Start as usual: `uv run run.py`. If the database can't be reached, it says
   so before starting anything.

With `FALL_DATABASE_URL` empty, the backend uses a PostgreSQL server you run
yourself (the `FALL_DB_*` settings, set up with `uv run scripts/create_db.py`).
With `FALL_STORAGE_BUCKET` empty, media stays on the station.

The automated tests never touch the shared database: `uv run pytest` starts a
private PostgreSQL of its own (bundled with the `pixeltable-pgserver` package)
and uses a stand-in for the bucket.

---

## Copying in existing data

Events recorded before the shared database existed are in a station's old
SQLite file (`data/falls.db`) or local PostgreSQL. Copy them in **once**, from
one station:

```bash
uv run scripts/copy_data.py data/falls.db
# or, from a local PostgreSQL:
uv run scripts/copy_data.py postgresql://fall:fall@localhost:5432/fall_detection

uv run scripts/sync_media.py        # upload their snapshots, clips and videos
```

**Expected:** `job: copied N rows`, `fallevent: copied N rows`, then
`Uploaded N files`. Rows keep their ids, so their files still match. The script
refuses to run if the shared database already has rows; `--replace` deletes
those first. **Be careful with `--replace` on the shared database**, because it
deletes everyone's data. Only one station can copy its old data this way,
because ids from two old databases would collide.

---

## Viewing and editing the data

- **In Supabase:** **Table Editor** shows every table and lets you edit rows
  like a spreadsheet. **SQL Editor** runs queries.
- **pgAdmin, DBeaver, or VS Code's SQLTools** work too. Use the host, port,
  user and password from `FALL_DATABASE_URL` (database `postgres`).

Useful queries:

```sql
-- Today's falls, newest first, with the station that recorded them
SELECT e.id, e.detected_at, j.station, e.camera_name, e.location, e.confidence, e.status
FROM fallevent e JOIN job j ON j.id = e.job_id
WHERE e.detected_at >= date_trunc('day', now())
ORDER BY e.detected_at DESC;

-- Mark one as a false alarm
UPDATE fallevent SET status = 'false_alarm', reviewer = 'Sam', reviewed_at = now() WHERE id = 12;
```

Edits show up on every dashboard within a few seconds. Keep `status` to
`unacknowledged`, `needs_review`, `resolved` or `false_alarm`, and don't
change `id`, `job_id` or the `*_path` columns by hand.

Supabase shows a security warning for tables without **row level security**
(RLS), because its web API would otherwise serve them to anyone with the
project's public key. RLS is turned on for every table (migration 0003), which
closes that API; the backend connects as the tables' owner and isn't affected.
Keep it on for any table you add.

---

## Changing the tables (migrations)

Tables are created and updated by **Alembic migrations** in
`src/backend/migrations/versions/`. Each backend applies new migrations when it
starts, one station at a time, so every database keeps the same tables without
losing data.

To add or change a column:

1. Edit the table class in `src/backend/db.py`.
2. Point `.env` at a database **you can afford to break** first: comment out
   `FALL_DATABASE_URL` to use a local server, or use a second Supabase
   project. Then generate a migration from the difference:

   ```bash
   uv run alembic revision --autogenerate -m "add camera zone to job"
   ```

3. **Read the new file** in `src/backend/migrations/versions/`. Alembic
   guesses: a renamed column appears as a drop and an add (which would lose
   that column's data), so rewrite it as `op.alter_column(..., new_column_name=...)`.
   A new column that isn't nullable needs a `server_default`, so existing rows
   get a value. For a new table, add
   `op.execute('ALTER TABLE "<name>" ENABLE ROW LEVEL SECURITY')`.
4. Run `uv run pytest`. `tests/test_db.py::test_migrations_match_models` fails
   if `db.py` and the migrations disagree.
5. Commit the migration with the `db.py` change. **Every station must pull it
   before running again**: once the shared database is upgraded, a station on
   older code may break.

Never edit a migration that has already run on the shared database; add a new
one instead.

---

## Backup and restore

The Supabase free plan has **no backups you can download**, so make your own:

```bash
uv run scripts/backup.py backup
```

This writes `backups/fall-backup-<date>-<time>.zip`, containing:
- `database.dump`, made with PostgreSQL's `pg_dump`;
- the media: this station's `events/`, `output/` and `input/` folders, plus any
  files that are only in the bucket.

`--no-videos` leaves out the uploaded and annotated videos; snapshots and
review clips are always included. The `backups/` folder isn't committed to git,
so copy backups somewhere safe. To back up automatically, schedule the command
(Windows Task Scheduler, or cron).

```bash
uv run scripts/backup.py restore backups/fall-backup-20261030-120000.zip
```

This **replaces everything** in the configured database with the backup's
contents, puts its files into `data/`, and uploads them to the bucket. **Stop
the backend on every station first.** It asks before doing anything (`--yes`
skips the question). A backup from an older version is upgraded to the
current tables afterwards.

`pg_dump`, `pg_restore` and `psql` come bundled with the `pixeltable-pgserver`
package (PostgreSQL 18), so nothing needs installing. They work with any
PostgreSQL server from version 9.2 up.

---

## Handing over to the customer

Everything is standard PostgreSQL and S3, so the customer can keep Supabase or
move to their own infrastructure. Only `.env` changes:

| Option | How |
|---|---|
| **Keep the Supabase project** | Transfer it to the customer's Supabase organization (Project Settings → General → Transfer project). Nothing else changes. Give them the current keys, then have them generate new ones. |
| **Their own Supabase, AWS, Azure or Google Cloud** | Create a PostgreSQL database and an S3-compatible bucket there (AWS: RDS and S3, with `FALL_STORAGE_ENDPOINT` left empty). Fill in `.env`, then restore a backup (below). |
| **Fully on-site, no cloud** | A PostgreSQL server on their network (set up with `scripts/create_db.py`). Either leave `FALL_STORAGE_BUCKET` empty so each station keeps its own media, or run an S3-compatible server such as MinIO. |

To move the data: `uv run scripts/backup.py backup` with the old `.env`, then
`uv run scripts/backup.py restore <file>.zip` with the new one. The database
dump also loads with the customer's own PostgreSQL tools.

Before handover, **rotate the secrets**: a new database password and new
storage keys, so former team members' copies stop working.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `run.py` says `Could not connect to the database` | Check the internet connection and `FALL_DATABASE_URL`. If the Supabase project was idle for a week, it's paused: open it on supabase.com and click **Restore**. |
| `password authentication failed` | The password in `FALL_DATABASE_URL` is wrong. If it contains symbols such as `@`, `:`, `/` or `#`, they must be URL-encoded (`@` → `%40`); or reset it in Supabase to a generated one (Project Settings → Database). |
| `could not translate host name` or `Network is unreachable` | You're using the **Direct connection** URL, which needs IPv6. Use the **Session pooler** URL. |
| `max clients reached` / `too many connections` | Too many stations or scripts are connected at once for the free plan. Stop unused backends; each one keeps up to 5 connections open. |
| Snapshot or clip from another station doesn't load | That station hasn't uploaded it yet: it may still be writing the clip, or it was offline (it uploads on its next start). Check its storage settings with `uv run scripts/sync_media.py`. |
| `scripts/sync_media.py` fails with `InvalidAccessKeyId` or `SignatureDoesNotMatch` | The `FALL_STORAGE_KEY_ID` / `FALL_STORAGE_SECRET` pair is wrong. Create a new access key in Supabase. |
| A teammate's live camera shows as interrupted | Two stations share a `FALL_STATION_NAME`. Give each a different one. |
