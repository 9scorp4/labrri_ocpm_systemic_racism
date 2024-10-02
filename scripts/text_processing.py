import re
from pathlib import Path
from loguru import logger
import fitz
import pytesseract
from docx import Document as DocxDocument
from pdf2image import convert_from_path
import spacy
from unidecode import unidecode
import numpy as np
import cv2

from scripts.database import Database

class ProcessText:
    """
    Class for text processing with improved cleaning and anonymization.
    """

    def __init__(self):
        try:
            self.db = Database("data/database.db")
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
            logger.info(f"Attempting to extract text from {pdf_path}")
            with fitz.open(pdf_path) as doc:
                if not doc:
                    logger.warning(f"Empty document: {pdf_path}")
                    return ""

                logger.info(f"Document has {len(doc)} pages")
                for page_number in range(len(doc)):
                    page = doc.load_page(page_number)
                    if not page:
                        logger.warning(f"Failed to load page {page_number} from {pdf_path}")
                        continue

                    page_text = page.get_text("text")
                    text += page_text
                    logger.debug(f"Extracted {len(page_text)} characters from page {page_number}")

            if not text.strip():
                logger.warning(f"No text extracted from {pdf_path}. Attempting OCR.")
                text = self.ocr_pdf(pdf_path)

        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            # Attempt OCR as a fallback
            logger.info(f"Attempting OCR as fallback for {pdf_path}")
            text = self.ocr_pdf(pdf_path)

        logger.info(f"Total extracted text length: {len(text)}")
        return text

    def ocr_pdf(self, pdf_path):
        if not pdf_path:
            raise ValueError("pdf_path cannot be None")
        
        logger.info(f"Attempting OCR on {pdf_path}")

        try:
            images = convert_from_path(pdf_path, dpi=300)
        except Exception as e:
            logger.error(f"Error converting PDF {pdf_path} to images: {e}")

        extracted_text = []
        for i, image in enumerate(images):
            img_np = np.array(image)
            gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
            threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            text = pytesseract.image_to_string(threshold, lang='fra+eng')
            extracted_text.append(text)

            logger.info(f"Extracted text from image {i+1}")
        
        logger.info(f"Text extracted from all pages of {pdf_path}")

        combined_text = '\n'.join(extracted_text)

        self.db.update_document_content(pdf_path, combined_text)

        output_txt_path = Path(pdf_path).with_suffix('.txt')
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(combined_text)

        logger.info(f"OCR extraction completed for {pdf_path}. Text saved to {output_txt_path}")
        return combined_text

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