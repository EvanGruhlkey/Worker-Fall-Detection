"""Row level security on every table

Supabase serves tables in the public schema over its web API to anyone with
the project's public (anon) key, unless row level security is on. With it on
and no policies, that API sees nothing. The backend connects as the tables'
owner, which row level security doesn't apply to, so it is unaffected; so is
any other PostgreSQL server.

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-30
"""

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None

TABLES = ("job", "fallevent", "sitelocation", "alembic_version")


def upgrade():
    for table in TABLES:
        op.execute(f'ALTER TABLE "{table}" ENABLE ROW LEVEL SECURITY')


def downgrade():
    for table in TABLES:
        op.execute(f'ALTER TABLE "{table}" DISABLE ROW LEVEL SECURITY')
