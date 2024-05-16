import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import spacy
import logging

class Tools:
    def __init__(self, lang=None):
        logging.info('Initializing Tools')
        self.lang = lang
        self.stopwords_lang = self.stopwords(lang)

        try:
            if lang == 'bilingual':
                self.nlp = {
                    'fr': self.load_spacy('fr'),
                    'en': self.load_spacy('en')
                }
            else:
                self.nlp = self.load_spacy(lang)
        except Exception as e:
            logging.error(f"Failed to load spacy model. Error: {e}", exc_info=True)
    
    def stopwords(self, lang):
        # Define the stopwords for each language
        stopwords_dict = {
            'fr': set(stopwords.words('french')),
            'en': set(stopwords.words('english')),
            'bilingual': set(stopwords.words('french')) | set(stopwords.words('english'))
        }

        # Load additional stopwords from file
        try:
            with open(r"scripts\topic_analysis\stopwords.txt", 'r', encoding='utf-8') as f:
                additional_stopwords = set()
                for line in f:
                    word = line.strip().lower()
                    if word:
                        additional_stopwords.add(word)
                for key in stopwords_dict.keys():
                    stopwords_dict[key] |= additional_stopwords
                logging.info(f"Added {len(additional_stopwords)} additional stopwords")
        except FileNotFoundError:
            logging.warning("No stopwords.txt file found. Skipping addition of additional stopwords.")
        except Exception as e:
            logging.error(f"Failed to load stopwords.txt. Error: {e}", exc_info=True)

        return {word.lower() for word in stopwords_dict.get(lang, [])}

    def load_spacy(self, lang):
        try:
            if lang == 'fr':
                return spacy.load('fr_core_news_sm')
            elif lang == 'en':
                return spacy.load('en_core_web_sm')
            elif lang == 'bilingual':
                return {
                    'fr': spacy.load('fr_core_news_sm'),
                    'en': spacy.load('en_core_web_sm')
                }
            else:
                raise ValueError(f"Invalid language: {lang}")
        except Exception as e:
            logging.error(f"Failed to load spacy model. Error: {e}", exc_info=True)

    def lemmatize(self, token, lang):
        if not self.nlp:
            logging.error("NLP model not loaded")
            return token
        
        try:
            lemmatized_token = {}
            if lang == 'bilingual':
                # Split the token into French and English parts
                if 'fr' in self.nlp and 'en' in self.nlp:
                    lemmatized_token['fr'] = self.nlp['fr'](token).lemma_
                    lemmatized_token['en'] = self.nlp['en'](token).lemma_
                    # Check the percentage ratio of French and English lemmas
                    fr_ratio = len([token for token in lemmatized_token['fr'] if token.is_alpha]) / len(lemmatized_token['fr'])
                    en_ratio = len([token for token in lemmatized_token['en'] if token.is_alpha]) / len(lemmatized_token['en'])
                    # If either English and French tokens are present, return both texts
                    if fr_ratio > 0.3 or en_ratio > 0.3:
                        # If both English and French tokens are present, return both texts
                        lemmatized_token = {
                            'fr': ' '.join([token.lemma_ for token in lemmatized_token['fr']]),
                            'en': ' '.join([token.lemma_ for token in lemmatized_token['en']])
                        }
                    elif en_ratio <= 0.09:
                        # If only French tokens are present, return only French text
                        lemmatized_token = {
                            'fr': ' '.join([token.lemma_ for token in lemmatized_token['fr']]),
                            'en': ''
                        }
                    elif fr_ratio <= 0.09:
                        # If only English tokens are present, return only English text
                        lemmatized_token = {
                            'fr': '',
                            'en': ' '.join([token.lemma_ for token in lemmatized_token['en']])
                        }
                else:
                    logging.error("NLP model not loaded")
                    return token
            elif lang == 'fr':
                doc = self.nlp(token)
                lemmatized_token = {
                    'fr': doc[0].lemma_,
                    'en': ''
                }
                return lemmatized_token
            elif lang == 'en':
                doc = self.nlp(token)
                lemmatized_token = {
                    'fr': '',
                    'en': doc[0].lemma_
                }
                return lemmatized_token
        except IndexError:
            logging.warning(f"Failed to lemmatize token: {token}")
            return token
        except Exception as e:
            logging.error(f"Failed to lemmatize token: {token}. Error: {e}", exc_info=True)
            return token