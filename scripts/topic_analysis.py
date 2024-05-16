import os
import logging
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from scripts.topic_analysis.text_processing import Process
from scripts.topic_analysis.analysis import Analysis

"""import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('average_perceptron_tagger')
nltk.download('punkt')
nltk.download('perluniprops')
nltk.download('snowball_data')"""

# Set up logging
MOD_DIR = Path(__file__).resolve().parent
LOGS_DIR = MOD_DIR / "logs"
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOGS_FILE = LOGS_DIR / f"topic_analysis_{timestamp}.log"

logging.basicConfig(
    filename=LOGS_FILE,
    level=logging.DEBUG,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

def main(lang=None):
    try:
        text_processor = Process(lang)
        analysis = Analysis(db='data\database.db', lang=lang)

        docs = list(analysis.fetch())

        with tqdm(total=len(docs), desc="Processing Sentences", unit="sentences", unit_scale=True, mininterval=0.5) as pbar:
            processed_docs = text_processor.docs_parallel(docs, lang=lang, pbar=pbar)
        
        analysis.docs_batch(processed_docs)
    
    except FileNotFoundError as e:
        logging.error(f"Failed to fetch data from database. Error: {e}", exc_info=True)
    except Exception as e:
        logging.error(f"Failed to process data. Error: {e}", exc_info=True)
    else:
        logging.info("Script completed successfully.")