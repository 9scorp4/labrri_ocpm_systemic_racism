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

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Function call timed out")

def process_single_document(doc):
    db = Database(DB_PATH)
    analysis = Analysis(db=DB_PATH, lang='bilingual')
    try:
        logger.debug(f"Processing document {doc[0]}")
        result = analysis.analyze_docs(doc[1], method='lda', num_topics=20)
        logger.debug(f"Analysis result for document {doc[0]}: {result}")
        return (doc[0], result)
    except (AnalysisError, DatabaseError, VectorizationError) as e:
        logger.error(f"Error processing document {doc[0]}: {str(e)}", exc_info=True)
    except Exception as e:
        logger.error(f"Error processing document {doc[0]}: {str(e)}", exc_info=True)
    return None

def populate_topics():
    logger.info("Starting topic population process")
    
    try:
        db = Database(DB_PATH)
        analysis = Analysis(db=DB_PATH, lang='bilingual')
        
        # Get all documents
        docs = db.fetch_all()
        
        if not docs:
            logger.error("No documents found in the database.")
            return
        
        # Get the last update time
        last_update = db.get_last_update_time()
        logger.info(f"Last update time: {last_update}")
        
        # Filter documents that need processing
        docs_to_process = [doc for doc in docs if db.needs_processing(doc[0], last_update)]
        
        logger.info(f"Processing {len(docs_to_process)} documents")
        
        if docs_to_process:
            try:
                all_docs_content = [doc[1] for doc in docs_to_process]
                results = analysis.analyze_docs(
                    all_docs_content,
                    method='lda',
                    num_topics=100,
                    coherence_threshold=-3.5,
                    min_topics=15,
                    max_topics=35
                )
                logger.info(f"Analyzed documents and generated {len(results)} topics")

                csv_file = analysis.save_topics_to_csv(results)
                logger.info(f"Saved topics to {csv_file}")

                updates = []
                for doc, doc_content in docs_to_process:
                    doc_topics = [topic for topic in results if any(word in doc_content for word in topic[1])]
                    updates.append((doc, doc_topics))
                
                logger.info(f"Updating topics for {len(updates)} documents")
                db.batch_update_document_topics(updates)

                # Update last processing time
                db.update_last_processing_time(docs_to_process[-1][0], results)
                logger.info(f"Updated last processing time for document {docs_to_process[-1][0]}")
            except Exception as e:
                logger.error(f"Error processing documents: {str(e)}", exc_info=True)
        else:
            logger.info("No documents need processing")
        
        logger.info("Topic population process completed")
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}", exc_info=True)
    finally:
        logger.info("Script execution completed")

if __name__ == "__main__":
    try:
        populate_topics()
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}", exc_info=True)
    finally:
        logger.info("Script execution completed")