from loguru import logger
from pathlib import Path 
from datetime import datetime
from scripts.database import Database
from scripts.topic_analysis_main import topic_analysis

# Define constants
LOGS_DIR = Path('logs')
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOGS_FILE = LOGS_DIR / f"populate_topics_{timestamp}.log"
PDF_LIST = Path('data/pdf_list.csv')
DB_PATH = Path('data/database.db')

logger.add(
    LOGS_FILE,
    rotation="1 day",
    retention="7 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
)

def populate_topics():
    logger.info("Starting topic population process")
    db = Database('data/database.db')
    
    # Get all documents
    docs = db.fetch_all()
    
    if not docs:
        logger.error("No documents found in the database.")
        return
    
    # Perform topic analysis for all documents
    topic_analysis(lang='bilingual', mode='all', method='lda', num_topics=10)
    
    logger.info("Topic population process completed")

if __name__ == "__main__":
    populate_topics()