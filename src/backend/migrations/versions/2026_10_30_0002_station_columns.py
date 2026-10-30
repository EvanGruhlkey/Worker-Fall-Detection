"""Station columns, for several computers sharing one database

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-30
"""

import sqlalchemy as sa
import sqlmodel
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("job", sa.Column("station", sqlmodel.AutoString(), nullable=True))
    op.create_index("ix_job_station", "job", ["station"])
    op.add_column("sitelocation", sa.Column("station", sqlmodel.AutoString(), nullable=True))
    op.create_unique_constraint("sitelocation_station_key", "sitelocation", ["station"])
    # The single location row used to be inserted as id 1 by hand, which left
    # the id counter behind; new per-station rows would collide with it.
    op.execute(
        "SELECT setval(pg_get_serial_sequence('sitelocation', 'id'), "
        "COALESCE((SELECT MAX(id) FROM sitelocation), 0) + 1, false)"
    )


def downgrade():
    op.drop_constraint("sitelocation_station_key", "sitelocation", type_="unique")
    op.drop_column("sitelocation", "station")
    op.drop_index("ix_job_station", table_name="job")
    op.drop_column("job", "station")
