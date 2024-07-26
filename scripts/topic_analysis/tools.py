import nltk
from nltk.corpus import stopwords
import spacy
import logging

nltk.download('stopwords')

class Tools:
    def __init__(self, lang=None):
        """
        Initialize the Tools class.

        This class is used to load the spacy model and the stopwords for the specified language.
        Args:
            lang (str): The language to use. Required.

        Raises:
            ValueError: If the language is not specified.
            Exception: If the spacy model for the language cannot be loaded.
        """
        logging.info('Initializing Tools')
        if lang is None:
            raise ValueError("Language must be specified")

        # Initialize spaCy models and stopwords
        self.lang = lang
        self.nlp = {}
        self.stopwords_lang = {}

        try:
            if lang == 'bilingual' or lang == 'fr':
                self.nlp['fr'] = spacy.load('fr_core_news_sm')
                self.stopwords_lang['fr'] = set(stopwords.words('french'))
            if lang == 'bilingual' or lang == 'en':
                self.nlp['en'] = spacy.load('en_core_web_sm')
                self.stopwords_lang['en'] = set(stopwords.words('english'))
        except Exception as e:
            logging.error(f"Failed to load spacy model. Error: {e}", exc_info=True)
            raise e
        
        self.load_additional_stopwords()

    def load_additional_stopwords(self):
        """
        Load additional stopwords from the stopwords.txt file and add them to the stopwords for each language.

        The stopwords.txt file should contain one stopword per line, with no empty lines or leading/trailing white space.
        Each stopword should be in lowercase.

        If the stopwords.txt file is not found, a warning is logged and no additional stopwords are loaded.

        If any other error occurs while loading the stopwords, an error is logged and the exception is re-raised.
        """
        try:
            # Load additional stopwords from file
            with open(r'scripts/topic_analysis/stopwords.txt', 'r', encoding='utf-8') as f:
                additional_stopwords = set(line.strip().lower() for line in f if line.strip())

            # Add additional stopwords to stopwords for each language
            for lang in self.stopwords_lang:
                self.stopwords_lang[lang] |= additional_stopwords

            # Log the number of additional stopwords added
            logging.info(f"Added {len(additional_stopwords)} additional stopwords")

        except FileNotFoundError:
            # Log a warning if the stopwords file is not found
            logging.warning("No stopwords.txt file found. Skipping addition of additional stopwords.")

        except Exception as e:
            # Log an error and re-raise the exception if any other error occurs
            logging.error(f"Failed to load additional stopwords. Error: {e}", exc_info=True)

    def load_spacy(self, lang):
        if lang not in self.nlp:
            raise ValueError(f"Invalid language: {lang}")
        return self.nlp[lang]

    def lemmatize(self, token, lang):
        if lang not in self.nlp:
            logging.error(f"NLP model not loaded for language: {lang}")
            return token
        
        try:
            doc = self.nlp[lang](token)
            return doc[0].lemma_
        except Exception as e:
            logging.error(f"Failed to lemmatize token. Error: {e}", exc_info=True)
            return token