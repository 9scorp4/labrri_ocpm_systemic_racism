# -*- coding: utf-8 -*-

import re
import csv
import logging
from pathlib import Path

from scripts.text_processing import ProcessText
from scripts.database import Database

class ProcessDocuments:

    def __init__(self, pdf_list_path, min_content_length, min_concatenated_length):
        """
        Initialize the ProcessDocuments object with the provided parameters.

        Args:
            pdf_list_path (str): Path to the list of PDF documents.
            min_content_length (int): Minimum content length for processing.
            min_concatenated_length (int): Minimum concatenated length for processing.
        """
        if pdf_list_path is None:
            raise ValueError("pdf_list_path cannot be None")
        if min_content_length is None:
            raise ValueError("min_content_length cannot be None")
        if min_concatenated_length is None:
            raise ValueError("min_concatenated_length cannot be None")

        self.db = Database(r"data\database.db")
        self.pdf_list_path = pdf_list_path
        self.min_content_length = min_content_length
        self.min_concatenated_length = min_concatenated_length
        self.process_text = ProcessText()

    def pdf_single(self, pdf_path, organization, document_type, category, clientele, knowledge_type, language):
        """
        Process a single PDF document. Used as a helper function within the batch processing function.
        
        Args:
            pdf_path (str): Path to the PDF document.
            organization (str): Name of the organization.
            document_type (str): Type of the document.
            category (str): Category of the organization.
            clientele (str): Clientele the organization addresses.
            knowledge_type (str): Type of knowledge mobilized.
            language (str): Language of the document.
        """

        # Check if required parameters are not None
        if any(param is None for param in [pdf_path, organization, document_type, category, clientele, knowledge_type, language]):
            raise ValueError("Some of the required parameters are None")

        # Document processing
        logging.info(f"Extracting text from PDF: {pdf_path}")
        try:
            content = self.process_text.extract_from_pdf(pdf_path)
        except Exception as e:
            logging.error(f"Failed to extract text from PDF: {e}", exc_info=True)
            return

        logging.info(f"Cleaning and preprocessing...")
        try:
            cleaned_content = self.process_text.clean(content)
        except Exception as e:
            logging.error(f"Failed to clean and preprocess text: {e}", exc_info=True)
            return

        try:
            self.db.insert_data(
                organization,
                document_type,
                category,
                clientele,
                knowledge_type,
                language,
                str(pdf_path),
                cleaned_content,
            )
            logging.info("Operation successful. Proceeding to next document.")
        except Exception as e:
            logging.error(f"Operation failed. Error: {e}", exc_info=True)

    def pdf_batch(self, pdf_list_path):
        """
        Process a batch of PDF documents.
        
        Args:
            pdf_list_path: path to the list of PDF documents
        """

        if not pdf_list_path:
            raise ValueError("pdf_list_path cannot be None")

        try:
            self.db.clear_previous_data()

            # Read a list of PDF files
            with open(pdf_list_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    pdf_file = row.get('Path')
                    if not pdf_file:
                        logging.error(f"Invalid PDF file: {pdf_file}")
                        continue
                    
                    pdf_path = Path(pdf_file)
                    if not pdf_path.exists() or pdf_path.suffix.lower() != '.pdf':
                        logging.error(f"Invalid PDF file: {pdf_path}")
                        continue
                    
                    logging.info(f"Processing document: {pdf_path}")
                    print(f"Processing document: {pdf_path}")
                    
                    # Pass information from CSV row
                    organization = row.get('Organization')
                    document_type = row.get('Document Type')
                    category = row.get('Category')
                    clientele = row.get('Clientele')
                    knowledge_type = row.get('Knowledge Type')
                    language = row.get('Language')

                    # Process and insert data
                    self.pdf_single(pdf_path, organization, document_type, category, clientele, knowledge_type, language)

        except Exception as e:
            logging.error(f"Operation failed. Error: {e}", exc_info=True)
        
        logging.info("Batch processing complete.")

    def ocr(self, min_content_length, min_concatenated_length):
        """
        Check the database for documents with missing or incomplete content,
        and apply OCR to those documents.
        The function will filter documents based on the following criteria:
        - Documents with concatenated words without spaces (filter step 1).
        - Documents with content length shorter than the specified minimum
          (filter step 2).
       
        Args:
            min_content_length: minimum content length for processing
            min_concatenated_length: minimum concatenated length for processing
        """
        print("Checking for documents with missing or incomplete content...")

        try:
            cursor = self.db.conn.cursor()

            # Filter step 1: Checking documents with concatenated words without spaces
            cursor.execute("""
                SELECT filepath, content
                FROM documents
                INNER JOIN content ON documents.id = content.doc_id
            """)
            rows = cursor.fetchall()

            concatenated_documents = []
            for filepath, content in rows:
                if content is None:
                    continue
                # Find sequences of uppercase and lowercase letters
                sequences = re.findall(r'[A-ZÀ-ȕ][a-zà-ȕ]+', content)
                # Filter sequences longer than the threshold
                lengthy_sequences = [seq for seq in sequences if len(seq) >= min_concatenated_length]
                if lengthy_sequences:
                    concatenated_documents.append(filepath)

            # Filter step 2: Checking documents with unreadable content
            cursor.execute("""
                        SELECT filepath
                        FROM documents
                        INNER JOIN content ON documents.id = content.doc_id
                        WHERE LENGTH(TRIM(content.content)) < ?
                    """, (min_content_length,))
            rows = cursor.fetchall()

            unreadable_documents = [row[0] for row in rows]

            # Filter documents that are in both lists
            documents_to_ocr = list(set(concatenated_documents).union(set(unreadable_documents)))

            # Show results
            if not documents_to_ocr:
                print("No documents with missing, unreadable or incomplete content found.")
                logging.info("No documents with missing, unreadable or incomplete content found.")
            else:
                print(f"Found {len(documents_to_ocr)} documents with missing or incomplete content.")
                logging.info(f"Found {len(documents_to_ocr)} documents with missing or incomplete content.")

                for filepath in documents_to_ocr:
                    self.process_text.ocr_pdf(filepath)
                
                print("OCR processing complete.")

        except Exception as e:
            logging.error(f"Operation failed. Error: {e}", exc_info=True)
                
    
    def run(self, pdf_list_path, min_content_length, min_concatenated_length):
        """
        Run the document processing pipeline.

        This function processes a batch of PDF documents and applies OCR recognition to documents with missing, unreadable, or incomplete content.

        Args:
            pdf_list_path (str): The path to the list of PDF documents.
            min_content_length (int): The minimum content length for processing.
            min_concatenated_length (int): The minimum concatenated length for processing.

        Returns:
            None
        """
        # Check if the pdf_list_path is None
        if pdf_list_path is None:
            raise ValueError("pdf_list_path cannot be None")

        # Check if the min_content_length is None
        if min_content_length is None:
            raise ValueError("min_content_length cannot be None")

        # Check if the min_concatenated_length is None
        if min_concatenated_length is None:
            raise ValueError("min_concatenated_length cannot be None")

        # Batch process the remaining documents
        self.pdf_batch(pdf_list_path)

        # Apply OCR recognition to documents with missing, unreadable or incomplete content
        try:
            self.ocr(min_content_length, min_concatenated_length)
        except Exception as e:
            logging.error(f"OCR processing failed. Error: {e}", exc_info=True)
    
    def __call__(self, pdf_list_path, min_content_length, min_concatenated_length):
        """
        Call the `run` method of the `ProcessDocuments` class with the provided parameters.

        Args:
            pdf_list_path (str): The path to the list of PDF documents.
            min_content_length (int): The minimum content length for processing.
            min_concatenated_length (int): The minimum concatenated length for processing.

        Returns:
            None
        """
        if pdf_list_path is None:
            raise ValueError("pdf_list_path cannot be None")
        if min_content_length is None:
            raise ValueError("min_content_length cannot be None")
        if min_concatenated_length is None:
            raise ValueError("min_concatenated_length cannot be None")

        try:
            self.run(pdf_list_path, min_content_length, min_concatenated_length)
        except Exception as e:
            logging.error(f"Failed to run the document processing pipeline. Error: {e}", exc_info=True)
