import os
from pathlib import Path
from datetime import datetime
from loguru import logger
from starlette.applications import Starlette
from starlette.staticfiles import StaticFiles

from scripts.database import Database
from scripts.document_processing import ProcessDocuments

# Constants
REPO_DIR = Path(os.path.dirname(__file__))
LOGS_DIR = REPO_DIR / "logs"
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOGS_FILE = LOGS_DIR / f"update_database_{timestamp}.log"
MIN_CONTENT_LENGTH = 400
MIN_CONCATENATED_LENGTH = 100
PDF_LIST = r"data\pdf_list.csv"
DB_FILE = r"data\database.db"

# Initialize logging directory
if not LOGS_DIR.exists():
    LOGS_DIR.mkdir(parents=True)
    print(f"Created {LOGS_DIR} directory.")

# Configure loguru
logger.add(
    LOGS_FILE,
    rotation="1 day",
    retention="1 week",
    level="DEBUG",
    backtrace=True,
    diagnose=True
)

# Initialize static files folder
app = Starlette()
static_dir = Path(os.path.dirname(__file__)) / "static"
if not static_dir.exists():
    os.makedirs(static_dir)

# Initialize static files
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Initialize database
db = Database(DB_FILE)

# Initialize functions
csv_to_xlsx = db.csv_to_xlsx
process_documents = ProcessDocuments(PDF_LIST, MIN_CONTENT_LENGTH, MIN_CONCATENATED_LENGTH)

def update_database():
    if not Path(PDF_LIST).exists():
        logger.error(f"File {PDF_LIST} does not exist.")
        raise FileNotFoundError(PDF_LIST)

    # Output XLSX file path
    xlsx_path = PDF_LIST.replace(".csv", ".xlsx")

    # Convert CSV to XLSX
    logger.info(f"Converting {PDF_LIST} to {xlsx_path}...")
    csv_to_xlsx(PDF_LIST, xlsx_path)

    # Process the batch of PDF documents
    logger.info(f"Processing {PDF_LIST}...")
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