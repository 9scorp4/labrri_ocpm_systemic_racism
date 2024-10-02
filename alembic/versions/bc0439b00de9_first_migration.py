"""First migration

Revision ID: bc0439b00de9
Revises: 
Create Date: 2024-09-29 15:32:19.963258

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'bc0439b00de9'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create new tables with the desired structure
    op.execute('''
        CREATE TABLE new_content (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id INTEGER,
            content VARCHAR
        )
    ''')
    
    op.execute('''
        CREATE TABLE new_documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            organization VARCHAR,
            document_type VARCHAR,
            category VARCHAR,
            clientele VARCHAR,
            knowledge_type VARCHAR,
            language VARCHAR,
            filepath VARCHAR
        )
    ''')
    
    # Copy data from old tables to new tables
    op.execute('''
        INSERT INTO new_content (doc_id, content)
        SELECT doc_id, content FROM content
    ''')
    
    op.execute('''
        INSERT INTO new_documents (id, organization, document_type, category, clientele, knowledge_type, language, filepath)
        SELECT id, organization, document_type, category, clientele, knowledge_type, language, filepath FROM documents
    ''')
    
    # Drop old tables
    op.drop_table('content')
    op.drop_table('documents')
    
    # Rename new tables to original names
    op.rename_table('new_content', 'content')
    op.rename_table('new_documents', 'documents')


def downgrade() -> None:
    # Create tables with the original structure
    op.execute('''
        CREATE TABLE old_content (
            doc_id INTEGER,
            content TEXT
        )
    ''')
    
    op.execute('''
        CREATE TABLE old_documents (
            id INTEGER,
            organization TEXT,
            document_type TEXT,
            category TEXT,
            clientele TEXT,
            knowledge_type TEXT,
            language TEXT,
            filepath TEXT
        )
    ''')
    
    # Copy data from current tables to old structure
    op.execute('''
        INSERT INTO old_content (doc_id, content)
        SELECT doc_id, content FROM content
    ''')
    
    op.execute('''
        INSERT INTO old_documents (id, organization, document_type, category, clientele, knowledge_type, language, filepath)
        SELECT id, organization, document_type, category, clientele, knowledge_type, language, filepath FROM documents
    ''')
    
    # Drop current tables
    op.drop_table('content')
    op.drop_table('documents')
    
    # Rename old structure tables to original names
    op.rename_table('old_content', 'content')
    op.rename_table('old_documents', 'documents')