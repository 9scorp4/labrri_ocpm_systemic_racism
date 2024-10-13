"""Initial migration

Revision ID: 16d8c51c51ae
Revises: 
Create Date: 2024-10-09 10:28:33.671121

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '16d8c51c51ae'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    # Create documents table
    op.create_table('documents',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('organization', sa.String(), nullable=True),
        sa.Column('document_type', sa.String(), nullable=True),
        sa.Column('category', sa.String(), nullable=True),
        sa.Column('clientele', sa.String(), nullable=True),
        sa.Column('knowledge_type', sa.String(), nullable=True),
        sa.Column('language', sa.String(), nullable=True),
        sa.Column('filepath', sa.String(), nullable=True),
        sa.Column('last_processed', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('filepath')
    )

    # Create content table
    op.create_table('content',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('doc_id', sa.Integer(), nullable=True),
        sa.Column('content', sa.String(), nullable=True),
        sa.ForeignKeyConstraint(['doc_id'], ['documents.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

    # Create database_updates table
    op.create_table('database_updates',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=True),
        sa.Column('new_documents', sa.Integer(), nullable=True),
        sa.Column('updated_documents', sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )

    # Create topics table
    op.create_table('topics',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('label', sa.String(), nullable=True),
        sa.Column('words', sa.String(), nullable=True),
        sa.Column('coherence_score', sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )

    # Create document_topics table
    op.create_table('document_topics',
        sa.Column('doc_id', sa.Integer(), nullable=False),
        sa.Column('topic_id', sa.Integer(), nullable=False),
        sa.Column('relevance_score', sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(['doc_id'], ['documents.id'], ),
        sa.ForeignKeyConstraint(['topic_id'], ['topics.id'], ),
        sa.PrimaryKeyConstraint('doc_id', 'topic_id')
    )


def downgrade():
    op.drop_table('document_topics')
    op.drop_table('topics')
    op.drop_table('database_updates')
    op.drop_table('content')
    op.drop_table('documents')