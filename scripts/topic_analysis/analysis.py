import logging
import sqlite3
from langdetect import detect_langs
from langdetect import LangDetectException
from scripts.topic_analysis.tools import Tools
from scripts.topic_analysis.noise_remover import NoiseRemover
from scripts.topic_analysis.documents import Documents

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
        r"""
        Initialize the Analysis object.

        Args:
            db (str, optional): Path to the database file. Defaults to
                'data\database.db'.
            lang (str, optional): Language of the documents. Defaults to None.

        Raises:
            ValueError: If the language argument is None.
            RuntimeError: If the language model cannot be loaded.
        """
        # Set the database file path
        self.db = db

        # Set the language of the documents
        self.lang = lang

        # Check if the language argument is None
        if lang is None:
            raise ValueError("Language argument cannot be None")

        # Initialize the Tools object
        self.tools = Tools(lang)

        # Load the language model
        try:
            # Load the French language model if the language is 'fr'
            if lang == 'fr':
                self.nlp = self.tools.load_spacy('fr')
            # Load the English language model if the language is 'en'
            elif lang == 'en':
                self.nlp = self.tools.load_spacy('en')
            # If the language is neither 'fr' nor 'en', set the language model to None
            else:
                self.nlp = None
        # If the language model cannot be loaded, raise a RuntimeError
        except Exception as e:
            raise RuntimeError("Failed to load spacy model") from e

    def fetch(self, batch_size=1000):
        """
        Fetches documents from the database in batches.

        Args:
            batch_size (int, optional): Number of documents to fetch per batch.
                Defaults to 1000.

        Yields:
            str: Document content.

        Raises:
            RuntimeError: If there is an error connecting to the database or fetching data.
        """
        try:
            # Connect to the database
            conn = sqlite3.connect(self.db)
            cursor = conn.cursor()

            # Check if the database connection is successful
            if not conn:
                raise RuntimeError("Failed to connect to database.")

            docs = []
            # Check if the database cursor is not None
            if cursor is not None:
                # Construct the query to fetch documents
                query = f"SELECT c.content FROM documents d INNER JOIN content c ON d.id = c.doc_id WHERE d.language = '{self.lang}'"
                for row in cursor.execute(query):
                    docs.append(row[0])
                    # If the batch size is reached, yield the documents and clear the list
                    if len(docs) >= batch_size:
                        yield from docs
                        docs = []
                
                # If there are any remaining documents, yield them
                if docs:
                    yield from docs
            
            # Close the database connection
            conn.close()
            logging.info("Data fetched successfully.")
        except sqlite3.Error as e:
            # Log the error and raise a RuntimeError
            logging.error(f"Failed to fetch data from database. Error: {e}", exc_info=True)
            raise RuntimeError("Failed to fetch data from database.") from e

    def batch_detect_langs(self, sentences):
        """
        Detects the language of a list of sentences.

        Args:
            sentences (list): List of sentences.

        Returns:
            dict: Mapping of sentences to their corresponding languages.

        Raises:
            ValueError: If the sentences parameter is None.
            ValueError: If the length of sentences and detected_langs does not match.
            ValueError: If lang_info.lang is None.
            RuntimeError: If there is an error detecting the language.
        """
        langs = {}  # dictionary to store sentence-language mappings
        try:
            if sentences is None:
                raise ValueError("Sentences parameter cannot be None")

            # Detect the language for each sentence
            detected_langs = detect_langs(' '.join(sentences))

            # Check if the length of sentences and detected_langs matches
            if len(sentences) != len(detected_langs):
                raise ValueError("Length of sentences and detected_langs does not match")

            # Iterate over the sentences and languages
            for sentence, lang_info in zip(sentences, detected_langs):
                lang = lang_info.lang
                # Check if lang_info.lang is None
                if lang is None:
                    raise ValueError("lang_info.lang cannot be None")
                # Add the sentence-language mapping to the dictionary
                langs[sentence] = lang
        except LangDetectException as e:
            # Log the error and raise a RuntimeError
            logging.error(f"Failed to detect language. Error: {e}", exc_info=True)
            raise RuntimeError("Failed to detect language") from e

        return langs
    
    def batch_detect_lang(self, sentences):
        """
        Detects the language of a list of sentences.

        Args:
            sentences (list): List of sentences.

        Returns:
            dict: Mapping of sentences to their corresponding languages.
                If the language cannot be determined, the language is set to 'unknown'.

        Raises:
            ValueError: If the sentences parameter is None.
        """
        langs = {}  # dictionary to store sentence-language mappings

        if sentences is None:
            raise ValueError("Sentences parameter cannot be None")

        # Detect the language for each sentence
        detected_langs = self.batch_detect_langs(sentences)

        # Iterate over the sentences and languages
        for sentence, lang_info in zip(sentences, detected_langs):
            try:
                # Split the lang_info string into language and confidence score
                lang, _ = str(lang_info).split(':')
                # Add the sentence-language mapping to the dictionary
                langs[sentence] = lang
            except (ValueError, AttributeError):
                # If the lang_info cannot be split or is None, set the language to 'unknown'
                langs[sentence] = 'unknown'

        return langs
    
    def docs_batch(self, documents_batch):
        """
        Process a batch of documents. 

        Args:
            documents_batch (list): List of documents to be processed.

        Raises:
            Exception: If an error occurs during the processing of the documents.
        """
        try:
            # Check if documents_batch is empty
            if not documents_batch:
                logging.info("No documents fetched. Exiting analysis.")
                return

            # Log a sample document before noise removal
            logging.info(f"Sample document before noise removal:\n{documents_batch[0]}")

            # Remove noise
            noise_remover = NoiseRemover(lang=self.lang)
            cleaned_documents = noise_remover.clean(documents_batch, lang=self.lang)

            # Check if cleaned_documents is empty
            if not cleaned_documents:
                logging.warning("No cleaned documents fetched. Exiting analysis.")
                return

            # Log a sample document after noise removal
            logging.debug(f"Sample document after noise removal:\n{cleaned_documents[0]}")

            # Analyze cleaned documents
            doc_analyzer = Documents(topic_analysis=self, lang=self.lang)
            doc_analyzer.analyze(cleaned_documents)
        except Exception as e:
            # Log the error and re-raise the exception
            logging.error(f"Failed to analyze documents. Error: {e}", exc_info=True)
            raise
