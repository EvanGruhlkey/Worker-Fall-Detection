"""Alembic environment: migrates the database the backend is configured for.

The backend passes its own connection in ``config.attributes["connection"]``;
the ``alembic`` command connects using the settings in .env instead.
"""

import logging

from alembic import context
from sqlmodel import SQLModel

from backend.config import database_url_from_env
from backend.db import create_db_engine  # also registers the tables

config = context.config
target_metadata = SQLModel.metadata


def run_migrations(connection):
    context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    connection = config.attributes.get("connection")
    if connection is not None:
        run_migrations(connection)
        return

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    engine = create_db_engine(config.get_main_option("sqlalchemy.url") or database_url_from_env())
    try:
        with engine.begin() as connection:
            run_migrations(connection)
    finally:
        engine.dispose()


if context.is_offline_mode():
    raise SystemExit("Offline (--sql) migrations aren't supported; connect to a database.")
run_migrations_online()
