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

            if results:
                logger.info(f"Document {document_id} Results:")
                topic_df, words_df = create_topic_dataframes(results)

                if topic_df is not None and not topic_df.empty:
                    logger.info(f"Topic DataFrame:")
                    logger.info(topic_df.to_string())
                    topic_df.to_csv(f'results/topic_analysis/topic_analysis_{document_id}_topics.csv', index=False)
                else:
                    logger.warning("Topic DataFrame is empty.")

                if words_df is not None and not words_df.empty:
                    logger.info(f"Word DataFrame:")
                    logger.info(words_df.to_string())
                    words_df.to_csv(f'results/topic_analysis/topic_analysis_{document_id}_words.csv', index=False)
                else:
                    logger.warning("Word DataFrame is empty.")

                topic_df.to_csv(f'results/topic_analysis/topic_analysis_{document_id}_topics.csv', index=False)
                words_df.to_csv(f'results/topic_analysis/topic_analysis_{document_id}_words.csv', index=False)
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

            logger.info('All Documents Results:')
            if results:
                topic_df, words_df = create_topic_dataframes(results)

                if topic_df is not None and not topic_df.empty:
                    logger.info(f"Topic DataFrame:")
                    logger.info(topic_df.to_string())
                    topic_df.to_csv(f'results/topic_analysis/topic_analysis_all_{lang}_topics.csv', index=False)
                else:
                    logger.warning("Topic DataFrame is empty.")

                if words_df is not None and not words_df.empty:
                    logger.info(f"Word DataFrame:")
                    logger.info(words_df.to_string())
                    words_df.to_csv(f'results/topic_analysis/topic_analysis_all_{lang}_words.csv', index=False)
                else:
                    logger.warning("Word DataFrame is empty.")
            else:
                logger.warning("Noe results returned from topic analysis.")
                topic_df, words_df = None, None
        else:
            logger.error(f"Invalid mode or missing document ID for single mode.")
            return None, None

    except Exception as e:
        logger.error(f"Error processing topic analysis: {str(e)}")
        return None, None

    return topic_df, words_df