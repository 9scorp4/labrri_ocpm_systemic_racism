from pathlib import Path
import csv
from loguru import logger
import fitz
from pdf2image import convert_from_path
import pytesseract
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy.orm.exc import DetachedInstanceError

from scripts.text_processing import ProcessText
from scripts.database import Database
from scripts.models import Document, Content

class ProcessDocuments:
    def __init__(self, pdf_list_path, min_content_length, min_concatenated_length):
        if not pdf_list_path:
            raise ValueError("pdf_list_path cannot be None")
        if min_content_length is None:
            raise ValueError("min_content_length cannot be None")
        if min_concatenated_length is None:
            raise ValueError("min_concatenated_length cannot be None")

        self.db = Database(r"data\database.db")
        self.pdf_list_path = Path(pdf_list_path)
        self.min_content_length = min_content_length
        self.min_concatenated_length = min_concatenated_length
        self.process_text = ProcessText()

    def pdf_single(self, pdf_path, organization, document_type, category, clientele, knowledge_type, language):
        logger.info(f"Processing document: {pdf_path}")
        try:
            content = self.process_text.extract_from_pdf(pdf_path)
            cleaned_content = self.process_text.clean(content)

            with self.db.session_scope() as session:
                existing_doc = session.query(self.db.Document).filter_by(filepath=str(Path(pdf_path))).first()

                if existing_doc:
                    existing_doc.organization = organization
                    existing_doc.document_type = document_type
                    existing_doc.category = category
                    existing_doc.clientele = clientele
                    existing_doc.knowledge_type = knowledge_type
                    existing_doc.language = language
                    if existing_doc.content:
                        existing_doc.content.content = cleaned_content
                    else:
                        existing_doc.content = Content(content=cleaned_content)
                    logger.info(f"Document {pdf_path} updated successfully.")
                    return "updated"
                else:
                    new_doc = Document(
                        organization=organization,
                        document_type=document_type,
                        category=category,
                        clientele=clientele,
                        knowledge_type=knowledge_type,
                        language=language,
                        filepath=str(Path(pdf_path))
                    )
                    new_doc.content = Content(content=cleaned_content)
                    session.add(new_doc)
                    logger.info(f"Document {pdf_path} processed and inserted successfully.")
                    return "new"
        except DetachedInstanceError:
            logger.error(f"DetachedInstanceError for document {pdf_path}. The session might have expired")
            return "error"
        except Exception as e:
            logger.error(f"Error processing document {pdf_path}: {e}")
            return "error"
                    

    def pdf_batch(self):
        if not self.pdf_list_path.exists():
            raise FileNotFoundError(f"File {self.pdf_list_path} does not exist.")

        new_docs = 0
        updated_docs = 0

        try:
            with open(self.pdf_list_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                with ThreadPoolExecutor() as executor:
                    futures = []
                    for row in reader:
                        pdf_path = Path(row.get('Path', ''))
                        if not pdf_path.exists() or pdf_path.suffix.lower() != '.pdf':
                            logger.error(f"Invalid PDF file: {pdf_path}")
                            continue
                        futures.append(executor.submit(
                            self.pdf_single,
                            pdf_path,
                            row.get('Organization'),
                            row.get('Document Type'),
                            row.get('Category'),
                            row.get('Clientele'),
                            row.get('Knowledge Type'),
                            row.get('Language')
                        ))
                    for future in as_completed(futures):
                        result = future.result()
                        if result == "new":
                            new_docs += 1
                        elif result == "updated":
                            updated_docs += 1
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
        logger.info("Batch processing complete.")
        return new_docs, updated_docs

    def ocr(self):
        logger.info("Checking for documents with missing or incomplete content...")
        documents_to_ocr = self.db.get_documents_needing_ocr(self.min_content_length, self.min_concatenated_length)
        
        if not documents_to_ocr:
            logger.info("No documents with missing, unreadable or incomplete content found.")
            return

        logger.info(f"Found {len(documents_to_ocr)} documents with missing or incomplete content.")
        for filepath in documents_to_ocr:
            try:
                self.process_text.ocr_pdf(filepath)
                logger.info(f"OCR completed for {filepath}.")
            except Exception as e:
                logger.error(f"Error appling OCR to {filepath}: {e}")
        logger.info("OCR processing complete.")

    def extract_text_from_pdf(self, filepath):
        try:
            with fitz.open(filepath) as doc:
                text = ""
                for page in doc:
                    text += page.get_text()
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF {filepath}: {e}")
            return ""

    def apply_ocr(self, filepath):
        try:
            images = convert_from_path(filepath)
            text = ""
            for image in images:
                image_text = pytesseract.image_to_string(image)
                text += image_text + "\n"
            return text
        except Exception as e:
            logger.error(f"Error applying OCR to {filepath}: {e}")
            return ""

    def update_document_content(self, filepath, content):
        with self.db.session_scope() as session:
            document = session.query(self.db.Document).filter_by(filepath=filepath).first()
            if document:
                if document.content:
                    document.content.content = content
                else:
                    document.content = self.db.Content(content=content)
                logger.info(f"Updated content for document: {filepath}")
            else:
                logger.warning(f"Document not found in database: {filepath}")

    def run(self):
        logger.info("Starting document processing pipeline")
        new_docs, updated_docs = self.pdf_batch()
        self.ocr()
        logger.info("Document processing pipeline completed")
        return new_docs, updated_docs

    def __call__(self):
        return self.run()