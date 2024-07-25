import re
import logging
from pathlib import Path
from tqdm import tqdm
import fitz
import pytesseract
from docx import Document
from pdf2image import convert_from_path

from scripts.database import Database

class ProcessText:
    """
    Class for text processing.
    """

    def __init__(self):
        try:
            self.db = Database("data\database.db")
            self.conn = self.db.conn
        except Exception as e:
            logging.error(f"Error initializing database: {e}", exc_info=True)
            raise

    def extract_from_pdf(self, pdf_path):
        """
        Extract text from a PDF document.

        Args:
            pdf_path: path to the PDF document

        Returns:
            content: extracted text
        """
        if not pdf_path:
            raise ValueError("pdf_path cannot be None")

        text = ""

        try:
            with fitz.open(pdf_path) as doc:
                if not doc:
                    raise ValueError("Empty document")

                for page_number in range(doc.page_count):
                    page = doc.load_page(page_number)
                    if not page:
                        raise ValueError(f"Failed to load page {page_number}")

                    text += page.get_text("text")
        except Exception as e:
            logging.error(f"Operation failed. Error: {e}", exc_info=True)
            raise
        return text
    
    def ocr_pdf(self, pdf_path):
        """
        Apply OCR to a PDF document and updates the corresponding content in the database.

        Args:
            pdf_path: path to the PDF document
            conn: database connection
        """
        if not pdf_path:
            raise ValueError("pdf_path cannot be None")

        print(f"Applying OCR to {pdf_path}...")
        logging.info(f"Applying OCR to {pdf_path}...")

        # Create a cursor
        cursor = self.conn.cursor()

        # Retrieve the document_id from the documents table based on the filepath
        cursor.execute("SELECT id FROM documents WHERE filepath = ?", (pdf_path,))
        document_id = cursor.fetchone()

        if not document_id:
            print(f"No document found for {pdf_path}.")
            logging.info(f"No document found for {pdf_path}.")
            return

        document_id = document_id[0]

        # Convert PDF to images
        try:
            images = convert_from_path(pdf_path)
        except Exception as e:
            logging.error(f"Operation failed. Error: {e}", exc_info=True)
            raise

        # Initialize outer progress bar for the entire process
        ocr_progress_bar = tqdm(total=len(images), desc="Processing pages", unit="page")

        extracted_text = []
        for i, image in enumerate(images):
            # Upgrade the progress bar
            ocr_progress_bar.update(1)

            text = pytesseract.image_to_string(image)
            extracted_text.append(text)

        # Close the progress bar
        ocr_progress_bar.close()
        
        logging.info(f"Text extracted from {pdf_path}")
        logging.debug(f"Extracted text: {extracted_text}")

        # Update the content in the database
        try:
            # Concatenate all extracted text into a single string
            combined_text = '\n'.join(extracted_text)

            # Update the content in the database
            cursor.execute("UPDATE content SET content = ? WHERE doc_id = ?", (combined_text, document_id))

            # Commit the changes
            self.conn.commit()
            logging.debug(f"Content updated in the database.")
        except Exception as e:
            logging.error(f"Operation failed. Error: {e}", exc_info=True)
            self.conn.rollback()

        # Initialize the progress bar for saving text to a Word document
        word_progress_bar = tqdm(total=1, desc="Writing text to Word document", unit="doc")

        # Create a Word document
        document = Document()
        for text in extracted_text:
            document.add_paragraph(text)
        
        # Save document
        pdf_path = Path(pdf_path)
        output_txt_path = pdf_path.with_suffix(".docx")
        document.save(output_txt_path)

        # Close the progress bar
        word_progress_bar.close()

        print(f"OCR extraction completed for document ID: {document_id}. Word document saved to {output_txt_path}.")
        logging.info(f"OCR extraction completed for document ID: {document_id}. Word document saved to {output_txt_path}.")

    def clean(self, text):
        """
        Clean extracted text.

        Args:
            text (str): extracted text

        Returns:
            str: cleaned text
        """
        try:
            if not isinstance(text, str):
                raise ValueError("text must be a string")

            logging.debug(f"Initial text: {text}")

            # Remove multiple consecutive spaces
            cleaned_text = re.sub(r'\s+', ' ', text)

            # Remove unnecessary line breaks
            cleaned_text = re.sub(r'\n', ' ', cleaned_text)

            # Replace consecutive newlines with a single newline
            cleaned_text = re.sub(r'\n+', '\n', cleaned_text)

            # Remove non-alphanumeric characters (except those needed in French)
            cleaned_text = re.sub(r'[^\w\s.;:?!-]', ' ', cleaned_text).strip()

            # Convert to lowercase
            cleaned_text = cleaned_text.lower()

            logging.debug(f"Cleaned text: {cleaned_text}")
            return cleaned_text
        except Exception as e:
            logging.error(f"Operation failed. Error: {e}", exc_info=True)
            raise
