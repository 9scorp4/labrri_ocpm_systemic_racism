import logging
import sqlite3
from langdetect import detect_langs
from langdetect import LangDetectException
from scripts.topic_analysis.tools import Tools
from scripts.topic_analysis.noise_remover import NoiseRemover
from scripts.topic_analysis.documents import Documents

class Analysis:
    def __init__(self, db='data\database.db', lang=None):
        self.db = db
        self.lang = lang
        self.tools = Tools(lang)

        # Load language model
        if lang == 'fr':
            self.nlp = self.tools.load_spacy('fr')
        elif lang == 'en':
            self.nlp = self.tools.load_spacy('en')
        else:
            self.nlp = None

    def fetch(self, batch_size=1000):
        try:
            conn = sqlite3.connect(self.db)
            cursor = conn.cursor()

            docs = []
            for row in cursor.execute(f"SELECT c.content FROM documents d INNER JOIN content c ON d.id = c.doc_id WHERE d.language = '{self.lang}'"):
                docs.append(row[0])
                if len(docs) >= batch_size:
                    yield from docs
                    docs = []
            
            if docs:
                yield from docs
            
            conn.close()
            logging.info("Data fetched successfully.")
        except sqlite3.Error as e:
            logging.error(f"Failed to fetch data from database. Error: {e}", exc_info=True)

    def batch_detect_langs(self, sentences):
        langs = {}
        try:
            detected_langs = detect_langs(' '.join(sentences))
            for sentence, lang_info in zip(sentences, detected_langs):
                lang = lang_info.lang
                langs[sentence] = lang
        except LangDetectException as e:
            logging.error(f"Failed to detect language. Error: {e}", exc_info=True)

        return langs
    
    def batch_detect_lang(self, sentences):
        langs = {}
        detected_langs = self.batch_detect_langs(sentences)
        for sentence, lang_info in zip(sentences, detected_langs):
            try:
                lang, _ = str(lang_info).split(':')
                langs[sentence] = lang
            except ValueError:
                langs[sentence] = 'unknown'
        return langs
    
    def docs_batch(self, documents_batch):
        try:
            if not documents_batch:
                logging.info("No documents fetched. Exiting analysis.")
                return
            logging.info(f"Sample document before noise removal:\n{documents_batch[0]}")

            # Remove noise
            noise_remover = NoiseRemover(lang=self.lang)
            cleaned_documents = noise_remover.clean(documents_batch, lang=self.lang)

            logging.debug(f"Sample document after noise removal:\n{cleaned_documents[0]}")

            # Analyze cleaned documents
            doc_analyzer = Documents(topic_analysis=self, lang=self.lang)
            doc_analyzer.analyze(cleaned_documents)
        except Exception as e:
            logging.error(f"Failed to analyze documents. Error: {e}", exc_info=True)