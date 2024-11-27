import os
import sys
from pathlib import Path
from datetime import datetime
from loguru import logger

from scripts.database import Database
from scripts.document_processing import ProcessDocuments

# Constants
REPO_DIR = Path(os.path.dirname(__file__))
LOGS_DIR = REPO_DIR / "logs"
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOGS_FILE = LOGS_DIR / f"update_database_{timestamp}.log"
MIN_CONTENT_LENGTH = 400
MIN_CONCATENATED_LENGTH = 100
PDF_LIST = Path('data/pdf_list.csv')

# Initialize database
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_PATH = rf"postgresql://{DB_USER}:{DB_PASSWORD}@localhost:5432/labrri_ocpm_systemic_racism"

# Initialize logging directory
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

# Initialize database
db = Database(DB_PATH)

def update_database():
    if not PDF_LIST.exists():
        logger.error(f"File {PDF_LIST} does not exist.")
        raise FileNotFoundError(PDF_LIST)

    # Output XLSX file path
    xlsx_path = PDF_LIST.with_suffix('.xlsx')

    # Convert CSV to XLSX
    logger.info(f"Converting {PDF_LIST} to {xlsx_path}...")
    db.csv_to_xlsx(PDF_LIST, xlsx_path)

    # Process the batch of PDF documents
    logger.info(f"Processing {PDF_LIST}...")
    process_documents = ProcessDocuments(PDF_LIST, MIN_CONTENT_LENGTH, MIN_CONCATENATED_LENGTH, language='bilingual')
    new_docs, updated_docs = process_documents.pdf_batch()
    
    logger.info(f"Finished processing {PDF_LIST}.")
    logger.info(f"New documents added: {new_docs}")
    logger.info(f"Documents updated: {updated_docs}")

    # Log all database updates for auditing purposes
    db.log_database_updates(new_docs, updated_docs)

    # Check data integrity
    if db.check_data_integrity():
        logger.info("Data integrity check passed.")
    else:
        logger.warning("Data integrity check failed. Running cleanup...")
        db.cleanup_orphaned_data()

if __name__ == "__main__":
    update_database()