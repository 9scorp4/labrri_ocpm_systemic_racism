import logging
import sqlite3
from langdetect import detect_langs
from langdetect import LangDetectException
from scripts.topic_analysis.tools import Tools
from scripts.topic_analysis.noise_remover import NoiseRemover
from scripts.topic_analysis.documents import Documents, French, English, Bilingual
from scripts.topic_analysis.text_processing import Process

class Analysis:
    """
    Class for performing topic analysis on documents.

    Attributes:
        db (str): Path to the database file.
        lang (str): Language of the documents.
        tools (Tools): Object for performing various tasks.
        nlp (spacy.lang.Language): Language model for tokenization and
            part-of-speech tagging.
    """
    def __init__(self, db=r'data\database.db', lang=None):
        self.db = db
        self.lang = lang
        self.tools = Tools(lang)
        self.documents = self._get_documents_instance()

    def _get_documents_instance(self):
        if self.lang == 'fr':
            return French(self, self.lang)
        elif self.lang == 'en':
            return English(self, self.lang)
        elif self.lang == 'bilingual':
            return Bilingual(self, self.lang)
        else:
            raise ValueError(f"Unsupported language: {self.lang}")

    def fetch_all(self):
        """Fetch all documents from the database."""
        try:
            with sqlite3.connect(self.db) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT doc_id, content FROM content")
                return cursor.fetchall()
        except Exception as e:
            logging.error(f"Error fetching documents from the database: {e}", exc_info=True)
            return []
        
    def fetch_single(self, doc_id):
        """Fetch a single document from the database by ID."""
        try:
            with sqlite3.connect(self.db) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT doc_id, content FROM content WHERE doc_id=?", (doc_id,))
                return cursor.fetchone()
        except Exception as e:
            logging.error(f"Error fetching document {doc_id} from the database: {e}", exc_info=True)
            return None

    def process_documents(self, docs):
        if not isinstance(docs, list):
            docs = [docs]
        
        # Preprocess the documents to ensure they are in the correct format
        preprocessed_docs = []

        for doc in docs:
            if isinstance(doc, tuple) and len(doc) == 2:
                preprocessed_docs.append(doc)
            elif isinstance(doc, list) and len(doc) > 0:
                doc_content = ' '.join(str(item) for item in doc)
                preprocessed_docs.append((None, doc_content))
            else:
                preprocessed_docs.append((None, str(doc)))

        return self.documents.analyze(preprocessed_docs)