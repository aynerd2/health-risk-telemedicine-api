"""add user.consent_given_at

Revision ID: a3c1d9e5f2b7
Revises: ef78ccfb0247
Create Date: 2026-10-05 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a3c1d9e5f2b7'
down_revision: Union[str, None] = 'ef78ccfb0247'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Nullable: accounts registered before consent was collected have no timestamp.
    op.add_column('user', sa.Column('consent_given_at', sa.DateTime(), nullable=True))


def downgrade() -> None:
    op.drop_column('user', 'consent_given_at')
