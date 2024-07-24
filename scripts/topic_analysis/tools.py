import nltk
from nltk.corpus import stopwords
import spacy
import logging

nltk.download('stopwords')

class Tools:
    def __init__(self, lang=None):
        """
        Initialize the Tools class.

        Args:
            lang (str): The language to use. Required.

        Raises:
            ValueError: If the language is not specified.
            Exception: If the spacy model for the language cannot be loaded.
        """
        logging.info('Initializing Tools')
        self.lang = lang
        if lang is None:
            raise ValueError("Language must be specified")
        
        # Load stopwords for the language
        self.stopwords_lang = self.stopwords(lang)

        try:
            # Load the spacy model for the language
            self.nlp = self.load_spacy(lang)
            
            # If bilingual, also load the spacy models for French and English
            if lang == 'bilingual':
                self.nlp = {
                    'fr': self.load_spacy('fr'),
                    'en': self.load_spacy('en')
                }
        except Exception as e:
            # Log and raise an exception if the spacy model cannot be loaded
            logging.error(f"Failed to load spacy model. Error: {e}", exc_info=True)
            raise e
            
            
    
    def stopwords(self, lang):
        """
        Load the stopwords for the specified language.

        Args:
            lang (str): The language to load stopwords for.

        Returns:
            set: Set of stopwords for the specified language.
        """
        # Define the stopwords for each language
        stopwords_dict = {
            'fr': set(stopwords.words('french')),
            'en': set(stopwords.words('english')),
            'bilingual': set(stopwords.words('french')) | set(stopwords.words('english'))
        }

        # Load additional stopwords from file
        try:
            # Open the stopwords.txt file
            with open(r"scripts\topic_analysis\stopwords.txt", 'r', encoding='utf-8') as f:
                # Initialize an empty set to store additional stopwords
                additional_stopwords = set()
                # Read each line in the file and add the word to the set if it's not empty
                for line in f:
                    word = line.strip().lower()
                    if word:
                        additional_stopwords.add(word)
                # If the stopwords for the specified language exist,
                # add the additional stopwords to it
                if stopwords_dict.get(lang) is not None:
                    stopwords_dict[lang] |= additional_stopwords
                    # Log the number of additional stopwords added
                    logging.info(f"Added {len(additional_stopwords)} additional stopwords")
                else:
                    # Log a warning if the language is not found in the stopwords dictionary
                    logging.warning(f"Language '{lang}' not found in stopwords_dict. Skipping addition of additional stopwords.")
        except FileNotFoundError:
            # Log a warning if the stopwords.txt file is not found
            logging.warning("No stopwords.txt file found. Skipping addition of additional stopwords.")
        except Exception as e:
            # Log an error if there is an exception while loading the stopwords.txt file
            logging.error(f"Failed to load stopwords.txt. Error: {e}", exc_info=True)

        # Return the set of stopwords for the specified language
        return {word.lower() for word in stopwords_dict.get(lang, [])}

    def load_spacy(self, lang):
        """
        Load the spaCy model for the specified language.

        Args:
            lang (str): The language to load the spaCy model for.
                Valid values are 'fr' for French, 'en' for English, and 'bilingual' for both.

        Returns:
            spacy.Language: The loaded spaCy model for the specified language.
                If 'bilingual' is specified, a dictionary is returned with the models for French and English.

        Raises:
            ValueError: If an invalid language is specified.
            OSError: If the spaCy model cannot be loaded.
        """
        try:
            # Load the spaCy model for the specified language
            if lang == 'fr':
                return spacy.load('fr_core_news_sm')
            elif lang == 'en':
                return spacy.load('en_core_web_sm')
            elif lang == 'bilingual':
                # Load the spaCy models for French and English
                return {
                    'fr': spacy.load('fr_core_news_sm'),
                    'en': spacy.load('en_core_web_sm')
                }
            else:
                raise ValueError(f"Invalid language: {lang}")
        except OSError as e:
            # Log and raise an exception if the spaCy model cannot be loaded
            logging.error(f"Failed to load spacy model. Error: {e}", exc_info=True)

    def lemmatize(self, token, lang):
        """
        Lemmatize a token using the specified language.

        Args:
            token (str): The token to lemmatize.
            lang (str): The language to use for lemmatization.
                Valid values are 'fr' for French, 'en' for English, and 'bilingual' for both.

        Returns:
            dict: A dictionary containing the lemmatized token for each language.
                If lemmatization fails, the original token is returned.
        """
        if not self.nlp:
            # Log an error if the NLP model is not loaded
            logging.error("NLP model not loaded")
            return token
        
        try:
            lang = lang or self.lang
            lemmatized_token = {}
            if lang == 'bilingual':
                # Lemmatize the token for French and English
                if 'fr' not in self.nlp or 'en' not in self.nlp:
                    # Log an error if the NLP models for French and English are not loaded
                    logging.error("NLP models for French and English not loaded")
                    return token
                
                lemmatized_token['fr'] = self.nlp['fr'](token).lemma_
                lemmatized_token['en'] = self.nlp['en'](token).lemma_
                
                fr_ratio = len([token for token in lemmatized_token['fr'] if token.is_alpha]) / len(lemmatized_token['fr'])
                en_ratio = len([token for token in lemmatized_token['en'] if token.is_alpha]) / len(lemmatized_token['en'])
                
                if fr_ratio > 0.3 or en_ratio > 0.3:
                    # If the lemmatized token contains more than 30% alphabetic characters, return the lemmatized token
                    lemmatized_token = {
                        'fr': ' '.join([token.lemma_ for token in lemmatized_token['fr']]),
                        'en': ' '.join([token.lemma_ for token in lemmatized_token['en']])
                    }
                elif en_ratio <= 0.09:
                    # If the lemmatized token for English contains less than 9% alphabetic characters, return the lemmatized token for French only
                    lemmatized_token = {
                        'fr': ' '.join([token.lemma_ for token in lemmatized_token['fr']]),
                        'en': ''
                    }
                elif fr_ratio <= 0.09:
                    # If the lemmatized token for French contains less than 9% alphabetic characters, return the lemmatized token for English only
                    lemmatized_token = {
                        'fr': '',
                        'en': ' '.join([token.lemma_ for token in lemmatized_token['en']])
                    }
            elif lang == 'fr':
                # Lemmatize the token for French
                doc = self.nlp(token)
                lemmatized_token = {
                    'fr': doc[0].lemma_ if doc else '',
                    'en': ''
                }
            elif lang == 'en':
                # Lemmatize the token for English
                doc = self.nlp(token)
                lemmatized_token = {
                    'fr': '',
                    'en': doc[0].lemma_ if doc else ''
                }
            return lemmatized_token
        except Exception as e:
            # Log an error if there is an exception during lemmatization
            logging.error(f"Failed to lemmatize token: {token}. Error: {e}", exc_info=True)
            return token
