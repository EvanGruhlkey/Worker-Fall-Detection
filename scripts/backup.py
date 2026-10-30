"""Back up or restore the database together with its snapshots, clips and videos.

    uv run scripts/backup.py backup                  # writes backups/fall-backup-<time>.zip
    uv run scripts/backup.py backup --no-videos      # skip uploaded and annotated videos
    uv run scripts/backup.py restore backups/fall-backup-20261030-120000.zip

A backup is one .zip holding ``database.dump`` (pg_dump's custom format) and the
data folders the rows point at: events/ (snapshots and review clips), and
unless --no-videos, input/ (uploaded videos) and output/ (annotated videos).
Moving the system to another machine or database is a backup here and a
restore there; see docs/database.md.

Restoring REPLACES everything in the configured database and overwrites files
of the same name in the data folder. Stop the backend first.

Needs PostgreSQL's pg_dump and pg_restore (installed with PostgreSQL), at least
as new as the database server.
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

DUMP_NAME = "database.dump"
MEDIA_FOLDERS = ("events", "input", "output")


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
    """pg_dump/pg_restore from PATH, or from a standard PostgreSQL install on Windows."""
    found = shutil.which(name)
    if found:
        return found
    installs = sorted(
        Path(os.environ.get("ProgramFiles", r"C:\Program Files"), "PostgreSQL").glob(f"*/bin/{name}.exe"),
        key=lambda path: int(path.parts[-3]) if path.parts[-3].isdigit() else 0,
    )
    if installs:
        return str(installs[-1])
    sys.exit(f"{name} not found. It comes with PostgreSQL; add PostgreSQL's bin folder to PATH.")


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
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output / f"fall-backup-{datetime.now():%Y%m%d-%H%M%S}.zip"

    with tempfile.TemporaryDirectory() as scratch:
        dump = Path(scratch) / DUMP_NAME
        run([find_tool("pg_dump"), "--format=custom", "--no-owner", "--no-privileges",
             f"--file={dump}", f"--dbname={target}"], env)

        file_count = 0
        # Dump and videos are already compressed; storing them is just as small and faster.
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as zf:
            zf.write(dump, DUMP_NAME)
            for folder in folders:
                for path in sorted((settings.data_dir / folder).rglob("*")):
                    if path.is_file():
                        zf.write(path, f"data/{path.relative_to(settings.data_dir).as_posix()}")
                        file_count += 1
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
        run([find_tool("pg_restore"), "--clean", "--if-exists", "--no-owner", "--no-privileges",
             "--single-transaction", f"--dbname={target}", str(dump)], env)

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

    # A backup from an older version gets any newer tables and columns.
    init_db(settings.database_url)
    close_db()
    print(f"Restored database '{database}' and {restored} files into {data_dir}.")


def main():
    args = parse_args()
    settings = Settings()
    if args.command == "backup":
        backup(args, settings)
    else:
        restore(args, settings)


if __name__ == "__main__":
    main()
