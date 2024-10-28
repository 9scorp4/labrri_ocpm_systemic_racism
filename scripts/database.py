import os
from loguru import logger
from pathlib import Path
from sqlalchemy import create_engine, text, func, select, join
from sqlalchemy.pool import QueuePool
from sqlalchemy.orm import sessionmaker, joinedload, aliased
from sqlalchemy.exc import SQLAlchemyError
from tenacity import retry, stop_after_attempt, wait_exponential
import pandas as pd
from alembic import command
from alembic.config import Config
from contextlib import contextmanager
from datetime import datetime

from .models import Base, Document, Content, DatabaseUpdate, DocumentTopic, Topic

class Database:
    def __init__(self, db_url=None):
        """Initialize database connection with PostgreSQL.
        
        Args:
            db_url (str, optional): Database URL. If not provided, constructs from environment variables.
        """
        if not db_url:
            # Construct database URL from environment variables
            db_user = os.getenv('DB_USER')
            db_password = os.getenv('DB_PASSWORD')
            db_host = os.getenv('DB_HOST', 'localhost')
            db_port = os.getenv('DB_PORT', '5432')
            db_name = os.getenv('DB_NAME', 'labrri_ocpm_systemic_racism')
            
            if not all([db_user, db_password]):
                raise ValueError("Database credentials must be provided in environment variables")
            
            db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

        try:
            # Configure PostgreSQL-specific engine parameters
            self.engine = create_engine(
                db_url,
                poolclass=QueuePool,
                pool_size=20,
                max_overflow=30,
                pool_timeout=60,
                pool_pre_ping=True,  # Enable connection health checks
                connect_args={
                    "keepalives": 1,
                    "keepalives_idle": 30,
                    "keepalives_interval": 10,
                    "keepalives_count": 5
                }
            )
            self.Session = sessionmaker(bind=self.engine)
            logger.info("PostgreSQL database connection successful")
            
            # Test the connection
            self.test_connection()
            
        except SQLAlchemyError as e:
            logger.error(f"Database connection failed: {str(e)}")
            raise

    def test_connection(self):
        """Test the database connection by executing a simple query."""
        try:
            with self.session_scope() as session:
                session.execute(text("SELECT 1"))
            logger.info("Database connection test successful")
        except SQLAlchemyError as e:
            logger.error(f"Database connection test failed: {str(e)}")
            raise

    @contextmanager
    def session_scope(self):
        """Provide a transactional scope around a series of operations."""
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            logger.error(f"Session error: {str(e)}")
            session.rollback()
            raise
        finally:
            session.close()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def execute_with_retry(self, operation):
        try:
            return operation()
        except SQLAlchemyError as e:
            logger.error(f"Database operation failed. Error: {e}")
            raise

    def get_document_by_path(self, pdf_path):
        with self.session_scope() as session:
            return session.query(Document).filter(Document.filepath == str(Path(pdf_path))).first()
        
    def update_document(self, doc_id, organization, document_type, category, clientele, knowledge_type, language, content):
        with self.session_scope() as session:
            doc = session.query(Document).filter(Document.id == doc_id).first()
            if doc:
                doc.organization = organization
                doc.document_type = document_type
                doc.category = category
                doc.clientele = clientele
                doc.knowledge_type = knowledge_type
                doc.language = language
                if doc.content:
                    doc.content.content = content
                else:
                    doc.content = Content(content=content)
                logger.info(f"Document {doc_id} updated successfully.")
            else:
                logger.error(f"Document {doc_id} not found.")

    def update_document_content(self, filepath, content):
        with self.session_scope() as session:
            document = session.query(self.Document).filter_by(filepath=str(Path(filepath))).first()
            if document:
                if document.content:
                    document.content.content = content
                else:
                    document.content = self.Content(content=content)
                logger.info(f"Updated content for document: {filepath}")
            else:
                logger.warning(f"Document not found in database: {filepath}")                    

    def log_database_updates(self, new_docs, updated_docs):
        with self.session_scope() as session:
            update = DatabaseUpdate(
                timestamp=datetime.now(),
                new_documents=new_docs,
                updated_documents=updated_docs
            )
            session.add(update)
            logger.info(f"Logged database update: {new_docs} new documents, {updated_docs} updated documents.")

    def get_documents_needing_ocr(self, min_content_length, min_concatenated_length):
        with self.session_scope() as session:
            query = session.query(Document.filepath).join(Content).filter(
                (Content.content == None) |
                (Content.content == '') |
                (func.length(Content.content) < min_content_length) |
                (func.length(func.replace(Content.content, ' ', '')) < min_concatenated_length)
            )
            return [doc.filepath for doc in query.all()]

    def fetch_all(self, lang=None):
        """Fetch all documents from the database.

        Args:
            lang (str, optional): Filter documents by language. Defaults to None.

        Returns:
            list: List of tuples containing the document id and content.
        """
        with self.session_scope() as session:
            query = session.query(Document).options(joinedload(Document.content))
            if lang:
                query = query.filter(Document.language == lang)
            results = [(doc.id, doc.content.content) for doc in query.all() if doc.content]
            logger.info(f"Fetched {len(results)} documents from the database.")
            if not results:
                logger.warning("No documents found in the database.")
            return results
    
    def fetch_all_docs_with_metadata(self):
        doc = aliased(Document)
        content = aliased(Content)

        query = (
            select(
                doc.id,
                doc.filepath,
                doc.organization,
                doc.document_type,
                doc.category,
                doc.clientele,
                doc.knowledge_type,
                doc.language,
                content.content,
            )
            .select_from(
                join(doc, content, doc.id == content.doc_id)
            )
            .order_by(doc.id)
        )

        with self.session_scope() as session:
            result = session.execute(query).all()

        documents = {}
        for row in result:
            for row in result:
                doc_id = row.id
                if doc_id not in documents:
                    documents[doc_id] = {
                        'id': doc_id,
                        'filepath': row.filepath,
                        'organization': row.organization,
                        'document_type': row.document_type,
                        'category': row.category,
                        'clientele': row.clientele,
                        'knowledge_type': row.knowledge_type,
                        'language': row.language,
                        'content': row.content
                    }
        return list(documents.values())

    def fetch_single(self, doc_id):
        """Fetch a single document from the database.

        Args:
            doc_id (int or tuple): The document id to fetch. If a tuple is provided, the first element of the tuple will be used.

        Returns:
            tuple or None: A tuple containing the document id and content, or None if the document is not found.
        """
        with self.session_scope() as session:
            logger.debug(f"Fetching document with id {doc_id}")
            if isinstance(doc_id, tuple):
                doc_id = doc_id[0]
            doc = session.query(Document).filter(Document.id == doc_id).first()
            if doc:
                logger.info(f"Document {doc_id} found in the database.")
                return (doc.id, doc.content.content if doc.content else None)
            else:
                logger.warning(f"Document {doc_id} not found in the database.")
                return None

    def df_from_query(self, query):
        try:
            return pd.read_sql(query, self.engine)
        except SQLAlchemyError as e:
            logger.error(f"Error executing query: {e}")
            raise

    def clear_previous_data(self):
        with self.session_scope() as session:
            session.query(Content).delete()
            session.query(Document).delete()
            logger.info("Previous data cleared successfully.")

    def insert_data(self, organization, document_type, category, clientele, knowledge_type, language, pdf_path, content):
        with self.session_scope() as session:
            new_doc = Document(
                organization=organization,
                document_type=document_type,
                category=category,
                clientele=clientele,
                knowledge_type=knowledge_type,
                language=language,
                filepath=str(Path(pdf_path))
            )
            new_content = Content(content=content)
            new_doc.content = new_content
            session.add(new_doc)
            logger.info(f"Data inserted successfully for document: {organization}")

    def run_migrations(self, alembic_cfg_path):
        try:
            alembic_cfg = Config(alembic_cfg_path)
            command.upgrade(alembic_cfg, "head")
            logger.info("Database migrations completed successfully.")
        except Exception as e:
            logger.error(f"Error running database migrations: {e}")
            raise

    def check_data_integrity(self):
        with self.session_scope() as session:
            orphaned_content = session.query(Content).filter(
                ~Content.doc_id.in_(session.query(Document.id))
            ).count()

            missing_content = session.query(Document).filter(
                ~Document.id.in_(session.query(Content.doc_id))
            ).count()

            if orphaned_content:
                logger.warning(f"Found {orphaned_content} orphaned content entries.")
            if missing_content:
                logger.warning(f"Found {missing_content} documents without content.")

            return orphaned_content == 0 and missing_content == 0

    def cleanup_orphaned_data(self):
        with self.session_scope() as session:
            deleted = session.query(Content).filter(
                ~Content.doc_id.in_(session.query(Document.id))
            ).delete(synchronize_session=False)
            logger.info(f"Cleaned up {deleted} orphaned content entries.")

    def csv_to_xlsx(self, csv_path, xlsx_path):
        try:
            df = pd.read_csv(csv_path)
            df.to_excel(xlsx_path, index=False)
            logger.info(f"CSV file {csv_path} successfully converted to XLSX file {xlsx_path}.")
        except Exception as e:
            logger.error(f"Error converting CSV to XLSX: {e}")
            raise

    def add_topic(self, label, words, coherence_score):
        with self.session_scope() as session:
            topic = Topic(label=label, words=words, coherence_score=coherence_score)
            session.add(topic)
            session.flush()
            return topic.id
    
    def add_document_topic(self, doc_id, topic_id, relevance_score):
        with self.session_scope() as session:
            doc_topic = DocumentTopic(doc_id=doc_id, topic_id=topic_id, relevance_score=relevance_score)
            session.add(doc_topic)
    
    def get_document_topics(self, doc_id):
        with self.session_scope() as session:
            return session.query(Topic).join(DocumentTopic).filter(DocumentTopic.doc_id == doc_id).all()

    def get_all_topics(self):
        with self.session_scope() as session:
            return session.query(Topic).all()
        
    def get_last_update_time(self):
        with self.session_scope() as session:
            last_update = session.query(DatabaseUpdate).order_by(DatabaseUpdate.timestamp.desc()).first()
            return last_update.timestamp if last_update else datetime.min
        
    def needs_processing(self, doc_id, last_update):
        with self.session_scope() as session:
            doc = session.query(Document).filter(Document.id == doc_id).first()
            return doc.last_processed is None or doc.last_processed < last_update
        
    def batch_update_document_topics(self, updates):
        def operation():
            with self.session_scope() as session:
                for doc_id, topics in updates:
                    # Delete existing topics for the document
                    session.query(DocumentTopic).filter(DocumentTopic.doc_id == doc_id).delete()
                    
                    # Add new topics
                    for label, words, coherence_score in topics:
                        topic = Topic(label=label, words=','.join(words), coherence_score=coherence_score)
                        session.add(topic)
                        session.flush()  # To get the topic id
                        doc_topic = DocumentTopic(doc_id=doc_id, topic_id=topic.id)
                        session.add(doc_topic)
                
                session.commit()
        
        return self.execute_with_retry(operation)

    def update_last_processing_time(self, doc_id, topics):
        with self.session_scope() as session:
            session.query(DocumentTopic).filter(DocumentTopic.doc_id == doc_id).delete()

            for label, words, coherence_score in topics:
                topic = Topic(label=label, words=','.join(words), coherence_score=coherence_score)
                session.add(topic)
                session.flush()

                doc_topic = DocumentTopic(doc_id=doc_id, topic_id=topic.id)
                session.add(doc_topic)
            
            doc = session.query(Document).filter(Document.id == doc_id).first()
            if doc:
                doc.last_processed = datetime.now()
            else:
                logger.warning(f"Document with id {doc_id} not found in the database.")
            
            logger.info(f"Updated topics for document {doc_id}.")

    def get_document_topics(self, doc_id):
        with self.session_scope() as session:
            topics = session.query(Topic).join(DocumentTopic).filter(DocumentTopic.doc_id == doc_id).all()
            # Detach the topics from the session
            return [
                {
                    'id': topic.id,
                    'label': topic.label,
                    'words': topic.words,
                    'coherence_score': topic.coherence_score
                } for topic in topics
            ]

    def clear_document_topics(self, doc_ids):
        with self.session_scope() as session:
            session.query(DocumentTopic).filter(DocumentTopic.doc_id.in_(doc_ids)).delete(synchronize_session=False)
            logger.info(f"Cleared existing topics for {len(doc_ids)} documents.")

    def get_unique_categories(self):
        query = "SELECT DISTINCT category FROM documents ORDER BY category"
        with self.engine.connect() as connection:
            result = connection.execute(text(query))
            categories = [row[0] for row in result]
        return categories

    def __del__(self):
        """Cleanup database resources."""
        if hasattr(self, 'engine'):
            self.engine.dispose()
            logger.info("Database connection closed")