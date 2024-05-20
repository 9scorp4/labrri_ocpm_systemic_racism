import os
import logging
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from scripts.topic_analysis.text_processing import Process
from scripts.topic_analysis.analysis import Analysis

def main(lang=None):
    try:
        if lang is None:
            raise ValueError("lang parameter is required")
        
        text_processor = Process(lang)
        analysis = Analysis(db=r'data\database.db', lang=lang)

        docs = list(analysis.fetch())
        if not docs:
            logging.warning("No documents found in database")
            return

        with tqdm(total=len(docs), desc="Processing Sentences", unit="sentences", unit_scale=True, mininterval=0.5) as pbar:
            processed_docs = text_processor.docs_parallel(docs, lang=lang, pbar=pbar)
        
        analysis.docs_batch(processed_docs)
    
    except FileNotFoundError as e:
        logging.error(f"Failed to fetch data from database. Error: {e}", exc_info=True)
    except Exception as e:
        logging.error(f"Failed to process data. Error: {e}", exc_info=True)
        raise
    else:
        logging.info("Script completed successfully.")
