import logging
import string
from nltk.corpus import stopwords
from spacy.lang.fr import French
from spacy.lang.en import English

class NoiseRemover:
    def __init__(self, lang):
        if lang == None:
            raise ValueError("Language must be specified")
        
        # Load spacy model
        try:
            self.nlp_fr = French()
            self.nlp_en = English()
        except Exception as e:
            raise ValueError(f"Failed to load spacy model. Error: {e}", exc_info=True)
        
        # Combine stopwords for both languages
        self.stopwords_fr = set(stopwords.words('french'))
        self.stopwords_en = set(stopwords.words('english'))
        self.additional_stopwords = set()

        try:
            with open(r'scripts\topic_analysis\stopwords.txt', 'r', encoding='utf-8') as f:
                for line in f:
                    self.additional_stopwords.add(line.strip().lower())
        except FileNotFoundError:
            raise FileNotFoundError("No stopwords.txt file found. Skipping addition of additional stopwords.")
        
        if lang == 'bilingual':
            self.stopwords_lang = {
                'fr': self.stopwords_fr | self.additional_stopwords,
                'en': self.stopwords_en | self.additional_stopwords
            }
        elif lang == 'fr':
            self.stopwords_lang = self.stopwords_fr | self.additional_stopwords
        elif lang == 'en':
            self.stopwords_lang = self.stopwords_en | self.additional_stopwords
        else:
            self.stopwords_lang = {}
        
        if self.stopwords_lang:
            logging.info(f"Added {len(self.stopwords_lang)} additional stopwords")
        else:
            logging.warning("No additional stopwords found. Skipping addition of additional stopwords.")
    
    def clean_docs(self, docs, lang=None):
        processed_docs = []
        logging.debug(f"Cleaning {len(docs)} documents")

        if lang == 'bilingual':
            processed_docs = self.bilingual_docs(docs)
        elif lang == 'fr':
            processed_docs = self.fr_docs(docs)
        elif lang == 'en':
            processed_docs = self.en_docs(docs)
        
        return processed_docs
    
    def bilingual_docs(self, docs):
        logging.info("Cleaning bilingual documents")
        logging.debug(f"Documents to clean: {len(docs)}")

        logging.debug("Flattening documents")
        flattened_docs = [item for sublist in docs for item in sublist]
        logging.debug(f"Flattened documents: {len(flattened_docs)}")

        logging.debug("Merging documents")
        merged_docs = {'fr': [], 'en': []}
        try:
            # Merge list of dictionaries into a single dictionary
            for doc in flattened_docs:
                logging.debug(f"Merging document: {doc}")
                if doc is None:
                    raise ValueError("Document is None")
                for key, value in doc.items():
                    if key in merged_docs:
                        merged_docs[key].append(value)
                    else:
                        merged_docs[key] = [value]
            logging.debug(f"Merged documents: {merged_docs}")

            logging.debug("Translating punctuation")
            translation_table = str.maketrans('', '', string.punctuation)

            merged_docs['fr'] = [value.translate(translation_table) for value in merged_docs['fr']]
            merged_docs['en'] = [value.translate(translation_table) for value in merged_docs['en']]
            logging.debug(f"Translated punctuation: {merged_docs}")

            logging.debug("Filtering out empty strings")
            merged_docs = {
                'fr': [value for value in merged_docs['fr'] if value],
                'en': [value for value in merged_docs['en'] if value]
            }
            logging.debug(f"Filtered empty strings: {merged_docs}")

            stopwords_lang = self.stopwords_lang
            logging.debug("Fitlering stopwords")
            merged_docs['fr'] = [value for value in merged_docs['fr'] if value.lower() not in stopwords_lang['fr']]
            merged_docs['en'] = [value for value in merged_docs['en'] if value.lower() not in stopwords_lang['en']]
            logging.debug(f"Filtered stopwords: {merged_docs}")

            logging.debug("Removing duplicates")
            merged_docs = {
                'fr': list(set(merged_docs['fr'])),
                'en': list(set(merged_docs['en']))
            }
            logging.debug(f"Removed duplicates: {merged_docs}")

        except Exception as e:
            logging.error(f"Failed to clean documents. Error: {e}", exc_info=True)
            return []
        
        logging.info("Documents cleaned successfully!")
        return merged_docs
    
    def fr_docs(self, docs):
        logging.debug(f"Cleaning {len(docs)} documents")
        try:
            for doc in docs:
                tokens = self.nlp_fr(doc)
        except Exception as e:
            logging.error(f"Failed to clean documents. Error: {e}", exc_info=True)
            return []
        
        filtered_tokens = [token.text.lower() for token in tokens if token.text.lower() not in self.stopwords_fr and token.text not in string.punctuation and not token.text.isdigit()]
        logging.debug(f"Filtered tokens: {filtered_tokens}")

        processed_doc = ' '.join(filtered_tokens)
        if processed_doc and len(processed_doc) > 0:
            return [processed_doc]
        
    def en_docs(self, docs):
        logging.debug(f"Cleaning {len(docs)} documents")
        try:
            for doc in docs:
                tokens = self.nlp_en(doc)
        except Exception as e:
            logging.error(f"Failed to clean documents. Error: {e}", exc_info=True)
            return []
        
        filtered_tokens = [token.text.lower() for token in tokens if token.text.lower() not in self.stopwords_en and token.text not in string.punctuation and not token.text.isdigit()]
        logging.debug(f"Filtered tokens: {filtered_tokens}")

        processed_doc = ' '.join(filtered_tokens)
        if processed_doc and len(processed_doc) > 0:
            return [processed_doc]