"""Add last_processed column to documents

Revision ID: 365fce923d28
Revises: 16d8c51c51ae
Create Date: 2024-10-09 13:56:17.946612

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '365fce923d28'
down_revision: Union[str, None] = '16d8c51c51ae'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create new tables with the desired schema
    op.create_table('documents_new',
        sa.Column('id', sa.Integer(), nullable=False, primary_key=True),
        sa.Column('organization', sa.String(), nullable=True),
        sa.Column('document_type', sa.String(), nullable=True),
        sa.Column('category', sa.String(), nullable=True),
        sa.Column('clientele', sa.String(), nullable=True),
        sa.Column('knowledge_type', sa.String(), nullable=True),
        sa.Column('language', sa.String(), nullable=True),
        sa.Column('filepath', sa.String(), nullable=True, unique=True),
        sa.Column('last_processed', sa.DateTime(), nullable=True)
    )
    
    op.create_table('content_new',
        sa.Column('id', sa.Integer(), nullable=False, primary_key=True),
        sa.Column('doc_id', sa.Integer(), sa.ForeignKey('documents_new.id'), nullable=True),
        sa.Column('content', sa.String(), nullable=True)
    )

    # Copy data from old tables to new tables
    op.execute('INSERT INTO documents_new SELECT id, organization, document_type, category, clientele, knowledge_type, language, filepath, NULL FROM documents')
    op.execute('INSERT INTO content_new SELECT id, doc_id, content FROM content')

    # Drop old tables
    op.drop_table('content')
    op.drop_table('documents')

    # Rename new tables to original names
    op.rename_table('documents_new', 'documents')
    op.rename_table('content_new', 'content')


def downgrade() -> None:
    # Create tables with the old schema
    op.create_table('documents_old',
        sa.Column('id', sa.Integer(), nullable=True, primary_key=True),
        sa.Column('organization', sa.String(), nullable=True),
        sa.Column('document_type', sa.String(), nullable=True),
        sa.Column('category', sa.String(), nullable=True),
        sa.Column('clientele', sa.String(), nullable=True),
        sa.Column('knowledge_type', sa.String(), nullable=True),
        sa.Column('language', sa.String(), nullable=True),
        sa.Column('filepath', sa.String(), nullable=True)
    )
    
    op.create_table('content_old',
        sa.Column('id', sa.Integer(), nullable=True, primary_key=True),
        sa.Column('doc_id', sa.Integer(), nullable=True),
        sa.Column('content', sa.String(), nullable=True)
    )

    # Copy data from current tables to old schema tables
    op.execute('INSERT INTO documents_old SELECT id, organization, document_type, category, clientele, knowledge_type, language, filepath FROM documents')
    op.execute('INSERT INTO content_old SELECT id, doc_id, content FROM content')

    # Drop current tables
    op.drop_table('content')
    op.drop_table('documents')

    # Rename old schema tables to original names
    op.rename_table('documents_old', 'documents')
    op.rename_table('content_old', 'content')
