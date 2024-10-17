import os
import sys
from loguru import logger
from pathlib import Path
from datetime import datetime

from scripts.database import Database
from scripts.topic_analysis.analysis import Analysis
from exceptions import AnalysisError, DatabaseError, VectorizationError

# Define constants
LOGS_DIR = Path('logs')
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOGS_FILE = LOGS_DIR / f"populate_topics_{timestamp}.log"
PDF_LIST = Path('data/pdf_list.csv')
BATCH_SIZE = 5  # Adjust based on your system's capabilities

# Initialize database
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_PATH = rf"postgresql://{DB_USER}:{DB_PASSWORD}@localhost:5432/labrri_ocpm_systemic_racism"

# Ensure the logs directory exists
if not LOGS_DIR.exists():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Configure loguru
logger.remove()
logger.add(
    sys.stderr,
    format="{time:YYYY-MM-DD at HH:mm:ss} | {process} | {level} | {message}",
    level="INFO",
    enqueue=True,
    backtrace=True,
    diagnose=True
)
logger.add(
    LOGS_FILE,
    format="{time:YYYY-MM-DD at HH:mm:ss} | {process} | {level} | {message}",
    level="DEBUG",
    rotation="10 MB",
    compression="zip",
    enqueue=True,
    backtrace=True,
    diagnose=True
)

def process_single_document(doc):
    db = Database(DB_PATH)
    analysis = Analysis(db=DB_PATH, lang='bilingual')
    try:
        logger.debug(f"Processing document {doc[0]}")
        result = analysis.analyze_docs([doc[1]], method='lda', num_topics=10)
        logger.debug(f"Analysis result for document {doc[0]}: {result}")
        return (doc[0], result)
    except (AnalysisError, DatabaseError, VectorizationError) as e:
        logger.error(f"Error processing document {doc[0]}: {str(e)}", exc_info=True)
    except Exception as e:
        logger.error(f"Error processing document {doc[0]}: {str(e)}", exc_info=True)
    return None

def populate_topics(doc_ids='all'):
    logger.info("Starting topic population process")
    
    try:
        db = Database(DB_PATH)
        analysis = Analysis(db=DB_PATH, lang='bilingual')

        if doc_ids == 'all':
            docs = db.fetch_all()
        elif isinstance(doc_ids, list):
            docs = [db.fetch_single(doc_id) for doc_id in doc_ids]
        elif isinstance(doc_ids, int):
            docs = [db.fetch_single(doc_ids)]
        else:
            raise ValueError("Invalid doc_ids argument")
        
        if not docs:
            logger.info("No documents found in the database")
            return
        
        logger.info(f"Processing {len(docs)} documents")

        results = []
        for doc in docs:
            if doc is not None and len(doc) >= 2:
                result = process_single_document(doc)
                if result:
                    results.append(result)
            else:
                logger.warning(f"Skipping invalid document: {doc}")

        logger.info(f"Processed {len(results)} documents")

        # Clear existing topics for these documents
        doc_ids = [doc[0] for doc in results]
        db.clear_document_topics(doc_ids)

        # Insert new topics
        for doc_id, topics in results:
            for topic_label, topic_words, coherence_score in topics:
                topic_id = db.add_topic(topic_label, ','.join(topic_words), coherence_score)
                db.add_document_topic(doc_id, topic_id, 1.0)
        
        logger.info("Topic population process completed successfully")
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}", exc_info=True)
    finally:
        logger.info("Script execution completed")

if __name__ == "__main__":
    try:
        populate_topics(doc_ids='all')
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}", exc_info=True)
    finally:
        logger.info("Script execution completed")