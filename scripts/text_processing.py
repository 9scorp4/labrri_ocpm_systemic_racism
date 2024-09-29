import re
from pathlib import Path
from loguru import logger
import fitz
import pytesseract
from docx import Document
from pdf2image import convert_from_path
import spacy
from unidecode import unidecode

from scripts.database import Database

class ProcessText:
    """
    Class for text processing with improved cleaning and anonymization.
    """

    def __init__(self):
        try:
            self.db = Database("data/database.db")
            self.conn = self.db.conn
            self.nlp_fr = spacy.load("fr_core_news_sm")
            self.nlp_en = spacy.load("en_core_web_sm")
        except Exception as e:
            logger.error(f"Error initializing ProcessText: {e}")
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
            logger.error(f"Operation failed. Error: {e}")
            raise
        return text
    
    def ocr_pdf(self, pdf_path):
        """
        Apply OCR to a PDF document and updates the corresponding content in the database.

        Args:
            pdf_path: path to the PDF document
        """
        if not pdf_path:
            raise ValueError("pdf_path cannot be None")

        logger.info(f"Applying OCR to {pdf_path}...")

        cursor = self.conn.cursor()

        cursor.execute("SELECT id FROM documents WHERE filepath = ?", (pdf_path,))
        document_id = cursor.fetchone()

        if not document_id:
            logger.info(f"No document found for {pdf_path}.")
            return

        document_id = document_id[0]

        try:
            images = convert_from_path(pdf_path)
        except Exception as e:
            logger.error(f"Operation failed. Error: {e}")
            raise

        extracted_text = []
        for image in images:
            text = pytesseract.image_to_string(image)
            extracted_text.append(text)

        logger.info(f"Text extracted from {pdf_path}")
        logger.debug(f"Extracted text: {extracted_text}")

        try:
            combined_text = '\n'.join(extracted_text)
            cursor.execute("UPDATE content SET content = ? WHERE doc_id = ?", (combined_text, document_id))
            self.conn.commit()
            logger.debug(f"Content updated in the database.")
        except Exception as e:
            logger.error(f"Operation failed. Error: {e}")
            self.conn.rollback()

        document = Document()
        for text in extracted_text:
            document.add_paragraph(text)
        
        output_txt_path = Path(pdf_path).with_suffix(".docx")
        document.save(output_txt_path)

        logger.info(f"OCR extraction completed for document ID: {document_id}. Word document saved to {output_txt_path}.")

    def clean(self, text: str, lang: str = 'fr') -> str:
        """
        Clean extracted text with improved noise removal and language-specific processing.

        Args:
            text: extracted text
            lang: language of the text ('fr' for French, 'en' for English)

        Returns:
            cleaned text
        """
        try:
            if not isinstance(text, str):
                raise ValueError("text must be a string")

            logger.debug(f"Initial text: {text}")

            # Remove multiple consecutive spaces
            cleaned_text = re.sub(r'\s+', ' ', text)

            # Remove unnecessary line breaks
            cleaned_text = re.sub(r'\n', ' ', cleaned_text)

            # Replace consecutive newlines with a single newline
            cleaned_text = re.sub(r'\n+', '\n', cleaned_text)

            # Remove non-alphanumeric characters (except those needed in French and English)
            cleaned_text = re.sub(r'[^\w\s.;:?!-]', ' ', cleaned_text).strip()

            # Convert to lowercase
            cleaned_text = cleaned_text.lower()

            # Language-specific cleaning
            if lang == 'fr':
                cleaned_text = self.clean_french(cleaned_text)
            elif lang == 'en':
                cleaned_text = self.clean_english(cleaned_text)

            # Light anonymization using NER
            cleaned_text = self.anonymize(cleaned_text, lang)

            logger.debug(f"Cleaned text: {cleaned_text}")
            return cleaned_text
        except Exception as e:
            logger.error(f"Operation failed. Error: {e}")
            raise

    def clean_french(self, text: str) -> str:
        """
        Apply French-specific text cleaning.

        Args:
            text: text to clean

        Returns:
            cleaned text
        """
        # Remove common French contractions
        contractions = {
            "l'": "l ", "d'": "d ", "j'": "j ", "m'": "m ", "t'": "t ", "s'": "s ",
            "n'": "n ", "qu'": "qu ", "jusqu'": "jusque ", "lorsqu'": "lorsque "
        }
        for contraction, replacement in contractions.items():
            text = text.replace(contraction, replacement)

        # Handle accents
        text = unidecode(text)

        return text

    def clean_english(self, text: str) -> str:
        """
        Apply English-specific text cleaning.

        Args:
            text: text to clean

        Returns:
            cleaned text
        """
        # Remove common English contractions
        contractions = {
            "n't": " not", "'s": " is", "'m": " am", "'re": " are",
            "'ll": " will", "'d": " would", "'ve": " have"
        }
        for contraction, replacement in contractions.items():
            text = text.replace(contraction, replacement)

        return text

    def anonymize(self, text: str, lang: str) -> str:
        """
        Perform light anonymization using Named Entity Recognition.

        Args:
            text: text to anonymize
            lang: language of the text ('fr' for French, 'en' for English)

        Returns:
            anonymized text
        """
        nlp = self.nlp_fr if lang == 'fr' else self.nlp_en
        doc = nlp(text)

        anonymized_text = text
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG']:
                anonymized_text = anonymized_text.replace(ent.text, f"[{ent.label_}]")

        return anonymized_text