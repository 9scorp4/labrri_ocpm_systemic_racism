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
            pdf_list_path: path to the list of PDF documents
            min_content_length: minimum content length for processing
            min_concatenated_length: minimum concatenated length for processing
        """
        self.db = Database(r"data\database.db")
        self.pdf_list_path = pdf_list_path
        self.min_content_length = min_content_length
        self.min_concatenated_length = min_concatenated_length
        self.process_text = ProcessText()

    def pdf_single(self, pdf_path, organization, document_type, category, clientele, knowledge_type, language):
        """
        Process a single PDF document. Used as a helper function within the batch processing function.
        
        Args:
            conn: database connection
            pdf_path: path to the PDF document
            organization: name of the organization
            document_type: type of the document
            category: category of the organization
            clientele: clientele the organization addresses
            knowledge_type: type of knowledge mobilized
            language: language of the document
        """

        # Document processing
        logging.info(f"Extracting text from PDF: {pdf_path}")
        content = ProcessText().extract_from_pdf(pdf_path)

        logging.info(f"Cleaning and preprocessing...")
        cleaned_content = ProcessText().clean(content)

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
            conn: database connection
            pdf_list_path: path to the list of PDF documents
        """
    
        try:
            self.db.clear_previous_data()

            # Read a list of PDF files
            with open(pdf_list_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    pdf_file = row['Path']
                    pdf_path = Path(pdf_file)

                    if pdf_path.exists() and pdf_path.suffix.lower() == '.pdf':
                        logging.info(f"Processing document: {pdf_path}")
                        print(f"Processing document: {pdf_path}")
                        # Pass information from CSV row
                        organization = row['Organization']
                        document_type = row['Document Type']
                        category = row['Category']
                        clientele = row['Clientele']
                        knowledge_type = row['Knowledge Type']
                        language = row['Language']

                        # Process and insert data
                        self.pdf_single(pdf_path, organization, document_type, category, clientele, knowledge_type, language)

                    else:
                        logging.error(f"Invalid PDF file: {pdf_path}")

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

        unreadable_documents = []
        for filepath in rows:
            filepath = filepath[0]
            unreadable_documents.append(filepath)

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
        # Batch process the remaining documents
        self.pdf_batch(pdf_list_path)
        # Apply OCR recognition to documents with missing, unreadable or incomplete content
        self.ocr(min_content_length, min_concatenated_length)
    
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
        self.run(pdf_list_path, min_content_length, min_concatenated_length)
