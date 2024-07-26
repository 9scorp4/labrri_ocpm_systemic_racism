import os
import logging
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from scripts.topic_analysis.text_processing import Process
from scripts.topic_analysis.analysis import Analysis
from scripts.database import Database

def main(lang, mode='all', document_id=None):
    if lang is None:
        raise ValueError("lang parameter is required")
    
    db = os.path.join(Path(__file__).parent.parent, 'data', 'database.db')
    database = Database(db)
    analysis = Analysis(db=db, lang=lang)
    
    try:
        process = Process(lang=lang)
    except Exception as e:
        logging.error(f"Error initializing Process: {str(e)}")
        return

    if mode == 'single' and document_id is not None:
        doc = database.fetch_single(document_id)
        if doc is None:
            logging.error(f"Document {document_id} not found in the database.")
            return
        
        with tqdm(total=1, desc=f"Processing Document {document_id}") as pbar:
            procesed_doc = process.single_doc(doc[1], lang)
            results = analysis.process_documents(([doc[0], procesed_doc]))
            pbar.update(1)

        print(f"Document {document_id} Results:")
        if results:
            for topic_idx, topic in enumerate(results):
                print(f"Topic {topic_idx + 1}: {', '.join(topic)}")
        else:
            print("No signficant topics found in the document.")

    elif mode == 'all':
        docs = database.fetch_all()
        if not docs:
            logging.error("No documents found in the database.")
            return
        
        with tqdm(total=len(docs), desc="Processing Documents") as pbar:
            for doc in docs:
                processed_docs = process.docs_parallel([doc[1] for doc in docs], lang, pbar)
                results = analysis.process_documents(list(zip([doc[0] for doc in docs], processed_docs)))

        print('All Documents Results:')
        if results:
            for topic_idx, topic in enumerate(results):
                print(f"Topic {topic_idx + 1}: {', '.join(topic)}")
        else:
            print("No signficant topics found in the documents.")
    
    else:
        logging.error(f"Invalid mode or missing document ID for single mode.")