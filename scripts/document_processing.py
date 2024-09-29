import os
import re
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from loguru import logger
import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
import docx
from PIL import Image
import textract

from scripts.database import Database
from scripts.text_processing import ProcessText

class ProcessDocuments:
    def __init__(self, pdf_list_path, min_content_length, min_concatenated_length):
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

        # Configure loguru
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logger.add(f"logs/document_processing_{timestamp}.log", rotation="500 MB")

    def extract_text(self, file_path):
        """Extract text from various document formats."""
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.pdf':
            return self.extract_from_pdf(file_path)
        elif file_extension == '.docx':
            return self.extract_from_docx(file_path)
        elif file_extension in ['.txt', '.rtf']:
            return self.extract_from_text(file_path)
        else:
            logger.warning(f"Unsupported file format: {file_extension}")
            return ""

    def extract_from_pdf(self, pdf_path):
        """Extract text from a PDF document with improved OCR."""
        text = ""
        try:
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    text += page.get_text()
                
                if not text.strip():  # If no text was extracted, use OCR
                    images = convert_from_path(pdf_path)
                    for image in images:
                        text += pytesseract.image_to_string(image, lang='eng+fra')  # Assuming English and French
        except Exception as e:
            logger.error(f"Failed to extract text from PDF: {pdf_path}. Error: {str(e)}")
        return text

    def extract_from_docx(self, docx_path):
        """Extract text from a DOCX document."""
        try:
            doc = docx.Document(docx_path)
            return " ".join([para.text for para in doc.paragraphs])
        except Exception as e:
            logger.error(f"Failed to extract text from DOCX: {docx_path}. Error: {str(e)}")
            return ""

    def extract_from_text(self, text_path):
        """Extract text from TXT or RTF documents."""
        try:
            return textract.process(text_path).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to extract text from text file: {text_path}. Error: {str(e)}")
            return ""

    def process_single_document(self, file_path, metadata):
        """Process a single document and insert into database."""
        logger.info(f"Processing document: {file_path}")
        
        content = self.extract_text(file_path)
        cleaned_content = self.process_text.clean(content)
        
        try:
            self.db.insert_data(
                metadata['Organization'],
                metadata['Document Type'],
                metadata['Category'],
                metadata['Clientele'],
                metadata['Knowledge Type'],
                metadata['Language'],
                str(file_path),
                cleaned_content,
            )
            logger.info(f"Successfully processed and inserted: {file_path}")
        except Exception as e:
            logger.error(f"Failed to insert data for {file_path}. Error: {str(e)}")

    def process_documents_parallel(self, document_list):
        """Process documents in parallel."""
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []
            for doc in document_list:
                future = executor.submit(self.process_single_document, doc['Path'], doc)
                futures.append(future)
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Documents"):
                future.result()  # This will raise any exceptions that occurred

    def run(self):
        """Run the document processing pipeline."""
        logger.info("Starting document processing pipeline")
        
        try:
            self.db.clear_previous_data()
            
            import csv
            with open(self.pdf_list_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                document_list = list(reader)
            
            self.process_documents_parallel(document_list)
            
            self.apply_ocr_where_needed()
            
            logger.info("Document processing pipeline completed successfully")
        except Exception as e:
            logger.error(f"Document processing pipeline failed. Error: {str(e)}")

    def apply_ocr_where_needed(self):
        """Apply OCR to documents with missing or incomplete content."""
        logger.info("Checking for documents requiring OCR")
        
        cursor = self.db.conn.cursor()
        cursor.execute("""
            SELECT filepath, content
            FROM documents
            INNER JOIN content ON documents.id = content.doc_id
        """)
        docs = cursor.fetchall()
        
        docs_needing_ocr = []
        for filepath, content in docs:
            if content is None or len(content) < self.min_content_length:
                docs_needing_ocr.append(filepath)
            elif re.search(r'[A-ZÀ-ȕ][a-zà-ȕ]{%d,}' % self.min_concatenated_length, content):
                docs_needing_ocr.append(filepath)
        
        for filepath in tqdm(docs_needing_ocr, desc="Applying OCR"):
            self.process_text.ocr_pdf(filepath)
        
        logger.info(f"OCR applied to {len(docs_needing_ocr)} documents")

    def __call__(self, pdf_list_path, min_content_length, min_concatenated_length):
        self.pdf_list_path = pdf_list_path
        self.min_content_length = min_content_length
        self.min_concatenated_length = min_concatenated_length
        self.run()