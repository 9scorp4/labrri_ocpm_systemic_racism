import os
import logging
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from scripts.topic_analysis.text_processing import Process
from scripts.topic_analysis.analysis import Analysis

def main(lang, mode='batch', document_id=None):
    try:
        if lang is None:
            raise ValueError("lang parameter is required")
        
        analysis = Analysis(db=r'data\database.db', lang=lang)
        text_processor = Process(lang)

        if mode == 'batch':
            docs = analysis.fetch_all()
            if not docs:
                logging.warning("No documents found in the database")

            with tqdm(total=len(docs), desc="Processing documents", unit="doc", unit_scale=True, mininterval=0.5) as pbar:
                processed_docs = []
                for doc_id, content in docs:
                    processed_content = text_processor.docs_parallel(content, lang, pbar)
                    if processed_content:
                        processed_docs.append((doc_id, processed_content))
                    pbar.update(1)
            
            analysis.process_documents(processed_docs)

        elif mode == 'single':
            if document_id == None:
                raise ValueError("document_id parameter is required for single mode")
            
            doc = analysis.fetch_single(document_id)
            if doc:
                doc_id, content = doc
                processed_content = text_processor.single_doc(content, lang)
                if processed_content:
                    results = analysis.process_documents((doc_id, processed_content))
                    print(f"Single Document (ID: {doc_id}) Results:\n{results}")
                else:
                    logging.error(f"Failed to process document with ID {doc_id}")
            else:
                logging.error(f"Document with ID {document_id} not found in the database")

        else:
            raise ValueError("Invalid mode. Must be 'batch' or 'single'")

    except Exception as e:
        logging.error(f"Failed to process data: Error: {e}", exc_info=True)
        raise
    else:
        logging.info("Data processing completed successfully")