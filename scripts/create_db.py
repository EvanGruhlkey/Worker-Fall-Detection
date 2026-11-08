"""Create the user and database on a self-hosted PostgreSQL server (run once).

Not needed with a hosted database such as the team's Supabase project, which
is set with FALL_DATABASE_URL instead.

    uv run scripts/create_db.py

Asks for the password of the PostgreSQL admin user (``postgres``, chosen when
PostgreSQL was installed), then creates:

- the user FALL_DB_USER with password FALL_DB_PASSWORD (from .env),
- the database FALL_DB_NAME, owned by that user, with the tables created.

Anything that already exists is left as it is, so it is safe to run again.
The admin password is only used for this script and is not saved. Set
FALL_DB_ADMIN_PASSWORD to skip the prompt. See docs/database.md.
"""

import argparse
import getpass
import os
import sys
from pathlib import Path

import psycopg
from psycopg import sql

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from backend.config import database_url_from_env  # noqa: E402  (also loads .env)
from backend.db import close_db, init_db  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--admin-user", default="postgres", help="PostgreSQL admin user (default: postgres)")
    return parser.parse_args()


def main():
    args = parse_args()
    if os.environ.get("FALL_DATABASE_URL"):
        sys.exit(
            "FALL_DATABASE_URL is set, so the database is hosted elsewhere (e.g. Supabase) "
            "and needs no setup here. Unset it to set up a local server with FALL_DB_*."
        )

    host = os.environ.get("FALL_DB_HOST", "localhost")
    port = int(os.environ.get("FALL_DB_PORT", "5432"))
    user = os.environ.get("FALL_DB_USER", "fall")
    password = os.environ.get("FALL_DB_PASSWORD", "fall")
    name = os.environ.get("FALL_DB_NAME", "fall_detection")

    admin_password = os.environ.get("FALL_DB_ADMIN_PASSWORD") or getpass.getpass(
        f"Password for PostgreSQL user '{args.admin_user}' on {host}:{port}: "
    )
    try:
        admin = psycopg.connect(
            host=host, port=port, user=args.admin_user, password=admin_password,
            dbname="postgres", autocommit=True,
        )
    except psycopg.OperationalError as error:
        sys.exit(f"Could not connect as '{args.admin_user}': {error}")

    with admin:
        if admin.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (user,)).fetchone():
            print(f"User '{user}' already exists (password left unchanged).")
        else:
            admin.execute(
                sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(
                    sql.Identifier(user), sql.Literal(password)
                )
            )
            print(f"Created user '{user}'.")

        if admin.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone():
            print(f"Database '{name}' already exists.")
        else:
            admin.execute(
                sql.SQL("CREATE DATABASE {} OWNER {} ENCODING 'UTF8'").format(
                    sql.Identifier(name), sql.Identifier(user)
                )
            )
            print(f"Created database '{name}'.")

    # Create the tables now, as the app's own user, to check its login works.
    try:
        init_db(database_url_from_env())
    except RuntimeError as error:
        sys.exit(f"{error}\nIf the user already existed, check FALL_DB_PASSWORD in .env.")
    close_db()
    print(f"Tables are up to date. The backend will use {user}@{host}:{port}/{name}.")


if __name__ == "__main__":
    main()
