# -*- coding: utf-8 -*-

import os
from pathlib import Path
from datetime import datetime
import logging
from starlette.applications import Starlette
from starlette.staticfiles import StaticFiles

# Initialize static files folder
app = Starlette()
static_dir = Path(os.path.dirname(__file__)) / "static"
if not static_dir.exists():
    os.makedirs(static_dir)

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

# Initialize logging file
logging.basicConfig(
    filename=LOGS_FILE,
    level=logging.DEBUG,
    format="%(asctime)s:%(levelname)s:%(message)s",
)

# Initialize static files
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Initialize functions
csv_to_xlsx = Database(DB_FILE).csv_to_xlsx
process_documents = ProcessDocuments(PDF_LIST, MIN_CONTENT_LENGTH, MIN_CONCATENATED_LENGTH)

if not Path(PDF_LIST).exists():
    logging.error(f"File {PDF_LIST} does not exist.", exc_info=True)
    raise FileNotFoundError(PDF_LIST)
else:
    # Output XLSX file path
    xlsx_path = PDF_LIST.replace(".csv", ".xlsx")

    # Convert CSV to XLSX
    print(f"Converting {PDF_LIST} to {xlsx_path}...")
    csv_to_xlsx(PDF_LIST, xlsx_path)

    # Process the batch of PDF documents
    print(f"Processing {PDF_LIST}...")
    process_documents(PDF_LIST, MIN_CONTENT_LENGTH, MIN_CONCATENATED_LENGTH)
    logging.info(f"Finished processing {PDF_LIST}.")