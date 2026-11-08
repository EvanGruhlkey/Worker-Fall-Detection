"""Back up or restore the database together with its snapshots, clips and videos.

    uv run scripts/backup.py backup                  # writes backups/fall-backup-<time>.zip
    uv run scripts/backup.py backup --no-videos      # skip uploaded and annotated videos
    uv run scripts/backup.py restore backups/fall-backup-20261030-120000.zip

A backup is one .zip holding ``database.dump`` (pg_dump's custom format) and the
files the rows point at: events/ (snapshots and review clips), and unless
--no-videos, input/ (this computer's uploaded videos) and output/ (annotated
videos). With shared storage configured, files only in the bucket (recorded by
other stations) are downloaded into the backup too. Moving the system to
another machine or database is a backup here and a restore there; see
docs/database.md.

Restoring REPLACES everything in the configured database, overwrites files of
the same name in the data folder, and uploads the media to shared storage.
Stop the backend on every station first.

pg_dump, pg_restore and psql come bundled (pixeltable-pgserver, PostgreSQL 18), which
can back up any server from PostgreSQL 9.2 to 18.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

from sqlalchemy.engine import make_url

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from backend.config import Settings  # noqa: E402
from backend.db import close_db, init_db  # noqa: E402
from backend.storage import SHARED_FOLDERS, MediaStore  # noqa: E402

DUMP_NAME = "database.dump"
MEDIA_FOLDERS = (*SHARED_FOLDERS, "input")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    backup = commands.add_parser("backup", help="write a backup .zip")
    backup.add_argument("--output", type=Path, default=ROOT / "backups", help="folder for the .zip")
    backup.add_argument(
        "--no-videos", action="store_true",
        help="leave out uploaded and annotated videos (snapshots and clips are kept)",
    )

    restore = commands.add_parser("restore", help="replace the database and files from a backup")
    restore.add_argument("archive", type=Path)
    restore.add_argument("--yes", action="store_true", help="don't ask for confirmation")
    return parser.parse_args()


def find_tool(name):
    """A PostgreSQL tool: the bundled PostgreSQL 18 copy, else one on the PATH."""
    import pixeltable_pgserver

    package = Path(pixeltable_pgserver.__file__).parent
    for folder in ("pginstall18", "pginstall"):
        for candidate in (package / folder / "bin" / name, package / folder / "bin" / f"{name}.exe"):
            if candidate.is_file():
                return str(candidate)
    found = shutil.which(name)
    if found:
        return found
    sys.exit(f"{name} not found; reinstall the dependencies with `uv sync`.")


def libpq_target(database_url):
    """(connection URL without the password, environment with PGPASSWORD) for pg tools."""
    url = make_url(database_url)
    env = {**os.environ}
    if url.password:
        env["PGPASSWORD"] = url.password
    plain = url.set(drivername="postgresql", password=None)
    return plain.render_as_string(hide_password=False), env


def run(command, env):
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"{Path(command[0]).name} failed:\n{result.stderr.strip()}")


def backup(args, settings):
    target, env = libpq_target(settings.database_url)
    folders = ("events",) if args.no_videos else MEDIA_FOLDERS
    media = MediaStore(settings)
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output / f"fall-backup-{datetime.now():%Y%m%d-%H%M%S}.zip"

    with tempfile.TemporaryDirectory() as scratch:
        dump = Path(scratch) / DUMP_NAME
        run([find_tool("pg_dump"), "--format=custom", "--no-owner", "--no-privileges",
             f"--file={dump}", f"--dbname={target}"], env)

        # Dump and videos are already compressed; storing them is just as small and faster.
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as zf:
            zf.write(dump, DUMP_NAME)
            stored = set()
            for folder in folders:
                for path in sorted((settings.data_dir / folder).rglob("*")):
                    if path.is_file():
                        key = path.relative_to(settings.data_dir).as_posix()
                        zf.write(path, f"data/{key}")
                        stored.add(key)
            if media.enabled:
                # Recorded on other stations: only in the bucket.
                download = Path(scratch) / "download"
                for key in sorted(media.remote_keys() - stored):
                    if key.split("/", 1)[0] in folders:
                        media.download(key, download)
                        zf.write(download, f"data/{key}")
                        stored.add(key)
            file_count = len(stored)
            zf.writestr("backup.json", json.dumps({
                "created": datetime.now().astimezone().isoformat(),
                "database": make_url(settings.database_url).database,
                "folders": list(folders),
                "files": file_count,
            }, indent=2))

    size_mb = archive.stat().st_size / 1e6
    print(f"Wrote {archive} ({size_mb:.1f} MB: database + {file_count} files from {', '.join(folders)}/)")


def restore(args, settings):
    if not args.archive.is_file():
        sys.exit(f"No backup at {args.archive}")
    database = make_url(settings.database_url).database
    if not args.yes:
        answer = input(
            f"This replaces ALL data in database '{database}' and overwrites files in "
            f"{settings.data_dir}. Is the backend stopped, and continue? [y/N] "
        )
        if answer.strip().lower() not in ("y", "yes"):
            sys.exit("Cancelled.")

    target, env = libpq_target(settings.database_url)
    data_dir = settings.data_dir.resolve()
    with zipfile.ZipFile(args.archive) as zf, tempfile.TemporaryDirectory() as scratch:
        if DUMP_NAME not in zf.namelist():
            sys.exit(f"{args.archive} is not a backup made by this script (no {DUMP_NAME}).")
        dump = Path(zf.extract(DUMP_NAME, scratch))
        # Turn the dump into SQL and run that, so it restores into older servers
        # too: PostgreSQL 18's tools add a setting servers before 17 reject.
        script = Path(scratch) / "restore.sql"
        run([find_tool("pg_restore"), "--clean", "--if-exists", "--no-owner", "--no-privileges",
             f"--file={script}", str(dump)], env)
        lines = script.read_text(encoding="utf-8").splitlines(keepends=True)
        script.write_text(
            "".join(line for line in lines if not line.startswith("SET transaction_timeout")),
            encoding="utf-8",
        )
        run([find_tool("psql"), "--no-psqlrc", "--quiet", "--set=ON_ERROR_STOP=1",
             "--single-transaction", f"--file={script}", f"--dbname={target}"], env)

        restored = 0
        for member in zf.infolist():
            if member.is_dir() or not member.filename.startswith("data/"):
                continue
            path = (data_dir / member.filename.removeprefix("data/")).resolve()
            if not path.is_relative_to(data_dir):
                continue  # never write outside the data folder
            path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            restored += 1

    media = MediaStore(settings)
    uploaded = media.sync()
    media.shutdown()

    # A backup from an older version gets any newer tables and columns.
    init_db(settings.database_url)
    close_db()
    print(f"Restored database '{database}' and {restored} files into {data_dir}.")
    if media.enabled:
        print(f"Uploaded {uploaded} of them to shared storage '{settings.storage_bucket}'.")


def main():
    args = parse_args()
    settings = Settings()
    if args.command == "backup":
        backup(args, settings)
    else:
        restore(args, settings)


if __name__ == "__main__":
    main()
