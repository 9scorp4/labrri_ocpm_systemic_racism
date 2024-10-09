import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.tokenize import word_tokenize, TweetTokenizer
import spacy
from loguru import logger
import os

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

class Tools:
    """
    Initialize tools for a given language.

    Args:
        lang (str): Language code ('fr', 'en', or 'bilingual')
    """
    def __init__(self, lang=None):
        """
        Initialize tools for a given language.

        :param lang: str, language code ('fr', 'en', or 'bilingual')
        """
        logger.info(f"Initializing tools for {lang}...")
        if lang is None:
            raise ValueError("Language must be specified")
        
        # Initialize spaCy models and stopwords
        self.lang = lang
        self.nlp = {}
        self.stopwords_lang = {}
        self.lemmatizers = {}
        self.stemmers = {}

        try:
            if lang == 'bilingual' or lang == 'fr':
                # Initialize French spaCy model
                self.nlp['fr'] = spacy.load('fr_core_news_md')
                # Load French stopwords
                self.stopwords_lang['fr'] = set(stopwords.words('french'))
                # Initialize French lemmatizer (using spaCy model)
                self.lemmatizers['fr'] = self.nlp['fr']
                # Initialize French stemmer
                self.stemmers['fr'] = SnowballStemmer('french')
                logger.info("French model loaded successfully.")
            if lang == 'bilingual' or lang == 'en':
                # Initialize English spaCy model
                self.nlp['en'] = spacy.load('en_core_web_md')
                # Load English stopwords
                self.stopwords_lang['en'] = set(stopwords.words('english'))
                # Initialize English lemmatizer (using WordNetLemmatizer)
                self.lemmatizers['en'] = WordNetLemmatizer()
                # Initialize English stemmer
                self.stemmers['en'] = SnowballStemmer('english')
                logger.info("English model loaded successfully.")

            if lang == 'bilingual':
                self.stopwords_lang['bilingual'] = self.stopwords_lang['fr'].union(self.stopwords_lang['en'])
                logger.info("Bilingual model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load spacy model. Error: {e}", exc_info=True)
            raise e
        
        # Load additional stopwords from file
        self.load_additional_stopwords()
        # Initialize TweetTokenizer
        self.tweet_tokenizer = TweetTokenizer()

        logger.info("Tools initialized successfully.")

    def load_additional_stopwords(self):
        """
        Load additional stopwords from file and add them to the existing stopwords
        for each language.

        The file should contain one word per line. The words are added to the
        existing stopwords for each language.

        If the file is not found, a warning is logged.

        :return: None
        """
        try:
            # File path is relative to this file
            stopwords_path = os.path.join(os.path.dirname(__file__), 'stopwords.txt')
            logger.info(f"Loading additional stopwords from {stopwords_path}...")
            with open(stopwords_path, 'r', encoding='utf-8') as f:
                # Read the file and strip and lower-case each line
                additional_stopwords = set(line.strip().lower() for line in f if line.strip())

            # Add the additional stopwords to the existing stopwords for each language
            for lang in self.stopwords_lang:
                self.stopwords_lang[lang] = self.stopwords_lang[lang].union(additional_stopwords)

            logger.info(f"Loaded {len(additional_stopwords)} additional stopwords")
        except FileNotFoundError:
            # If the file is not found, log a warning
            logger.warning("Additional stopwords file not found")
        except Exception as e:
            # If an error occurs, log an error and raise the exception
            logger.error(f"Failed to load additional stopwords. Error: {e}", exc_info=True)
            raise
    def load_spacy(self, lang):
        """
        Load the spaCy model for a given language.

        Args:
            lang (str): Language code ('fr', 'en', or 'bilingual')

        Returns:
            spacy.Language: spaCy model for the given language

        Raises:
            ValueError: If the language is not supported
        """
        if lang not in self.nlp:
            raise ValueError(f"Invalid language: {lang}")
        # Return the spaCy model for the given language
        return self.nlp[lang]
    
    def lemmatize(self, token, lang):
        """
        Lemmatize a given token using the lemmatizer for the given language.

        Args:
            token (str): Token to lemmatize
            lang (str): Language code ('fr', 'en', or 'bilingual')

        Returns:
            str: Lemmatized token

        Raises:
            ValueError: If the language is not supported
        """
        if lang == 'bilingual':
            try:
                # Try French lemmatization first, then English if it fails
                return self.lemmatizers['fr'](token)[0].lemma_
            except:
                return self.lemmatizers['en'].lemmatize(token)
        elif lang in self.lemmatizers:
            if lang == 'fr':
                return self.lemmatizers[lang](token)[0].lemma_
            else:
                return self.lemmatizers[lang].lemmatize(token)
        else:
            logger.error(f"Lemmatizer not found for language: {lang}")
            return token
        
    def stem(self, token, lang):
        """
        Stem a given token using the stemmer for the given language.

        Args:
            token (str): Token to stem
            lang (str): Language code ('fr', 'en', or 'bilingual')

        Returns:
            str: Stemmed token

        Raises:
            ValueError: If the language is not supported
        """
        if lang not in self.stemmers:
            logger.error(f"Stemmer not found for language: {lang}")
            return token
        
        try:
            # Use the SnowballStemmer for the given language
            return self.stemmers[lang].stem(token)
        except Exception as e:
            logger.error(f"Failed to stem token: {token}. Error: {e}", exc_info=True)
            # If an error occurs, return the original token
            return token
    
    def tokenize(self, text, method='standard'):
        """
        Tokenize a given text using a chosen method.

        Args:
            text (str): Text to tokenize
            method (str): Tokenization method ('standard' or 'tweet')

        Returns:
            List[str]: Tokens

        Raises:
            ValueError: If the method is not supported
        """
        try:
            if method == 'standard':
                # Use standard NLTK word tokenization
                return word_tokenize(text)
            elif method == 'tweet':
                # Use TweetTokenizer for tokenization
                return self.tweet_tokenizer.tokenize(text)
            else:
                raise ValueError(f"Invalid tokenization method: {method}. Using 'standard' instead.")
                return word_tokenize(text)
        except Exception as e:
            logger.error(f"Failed to tokenize text. Error: {e}", exc_info=True)
            return []

    def is_stopword(self, token, lang):
        return token.lower() in self.stopwords_lang.get(lang, set())
    
    def preprocess(self, text, lang, tokenize_method='standard', use_lemmatization=True):
        """
        Preprocess a given text.

        Args:
            text (str): Text to preprocess
            lang (str): Language of the text
            tokenize_method (str, optional): Tokenization method. Defaults to 'standard'.
            use_lemmatization (bool, optional): Whether to use lemmatization. Defaults to True.

        Returns:
            List[str]: Preprocessed tokens
        """
        tokens = self.tokenize(text, method=tokenize_method)
        processed_tokens = []
        for token in tokens:
            if not self.is_stopword(token, lang):
                if use_lemmatization:
                    # Use lemmatization to reduce tokens to their base form
                    processed_token = self.lemmatize(token, lang)
                else:
                    # Use stemming to reduce tokens to their base form
                    processed_token = self.stem(token, lang)
                processed_tokens.append(processed_token)
        return processed_tokens
