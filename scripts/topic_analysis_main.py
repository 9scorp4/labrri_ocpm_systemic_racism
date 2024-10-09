import os
from loguru import logger
from datetime import datetime
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from scripts.topic_analysis.ta_text_processing import Process
from scripts.topic_analysis.analysis import Analysis
from scripts.database import Database
from scripts.data_processing_utils import create_topic_dataframes

def topic_analysis(lang, mode='all', document_id=None, method='lda', num_topics=5):
    if lang not in ['fr', 'en', 'bilingual']:
        raise ValueError("lang must be 'fr', 'en', or 'bilingual'")
    
    topic_df = None
    words_df = None

    try:    
        db = os.path.join(Path(__file__).parent.parent, 'data', 'database.db')
        database = Database(db)
        analysis = Analysis(db=db, lang=lang)
    
        try:
            process = Process(lang=lang)
        except Exception as e:
            logger.error(f"Error initializing Process: {str(e)}")
            return None, None

        if mode == 'single' and document_id is not None:
            doc = database.fetch_single(document_id)
            if doc is None:
                logger.error(f"Document {document_id} not found in the database.")
                return
            
            with tqdm(total=1, desc=f"Processing Document {document_id}") as pbar:
                procesed_doc = process.single_doc(doc[1], lang)
                results = analysis.analyze_docs([doc], method=method, num_topics=num_topics)
                pbar.update(1)

            for label, words, coherence_score in results:
                topic_id = database.add_topic(label, words, coherence_score)
                db.add_document_topic(document_id, topic_id, 1.0)
        elif mode == 'all':
            docs = database.fetch_all()
            if not docs:
                logger.error("No documents found in the database.")
                return
            
            with tqdm(total=len(docs), desc="Processing Documents") as pbar:
                for doc in docs:
                    processed_docs = process.docs_parallel([doc[1] for doc in docs], lang, pbar)
                    results = analysis.analyze_docs(processed_docs, method=method, num_topics=num_topics)
                    pbar.update(len(docs))

            for label, words, coherence_score in results:
                topic_id = database.add_topic(label, words, coherence_score)
                for doc in docs:
                    relevance_score = calculate_relevance(doc, words)
                    database.add_document_topic(doc[0], topic_id, relevance_score)
        else:
            logger.error(f"Invalid mode or missing document ID for single mode.")
            return None, None

    except Exception as e:
        logger.error(f"Error processing topic analysis: {str(e)}")
        return None, None

    return topic_df, words_df