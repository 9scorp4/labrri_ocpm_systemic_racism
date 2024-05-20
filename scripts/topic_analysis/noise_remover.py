import logging
import string
from nltk.corpus import stopwords
from spacy.lang.fr import French
from spacy.lang.en import English

class NoiseRemover:
    def __init__(self, lang):
        """
        Initialize the NoiseRemover class.

        Args:
            lang (str): The language to be used for noise removal.

        Raises:
            ValueError: If language is not specified.
        """
        # Check if language is specified
        if lang is None:
            raise ValueError("Language must be specified")
        
        # Load spacy model
        try:
            # Load French and English models
            self.nlp_fr = French()
            self.nlp_en = English()
        except Exception as e:
            # Raise error if model loading fails
            raise ValueError(f"Failed to load spacy model. Error: {e}")
        
        # Combine stopwords for both languages
        self.stopwords_fr = set(stopwords.words('french'))
        self.stopwords_en = set(stopwords.words('english'))
        self.additional_stopwords = set()

        try:
            # Read additional stopwords from file
            stopwords_file = r'scripts\topic_analysis\stopwords.txt'
            with open(stopwords_file, 'r', encoding='utf-8') as f:
                for line in f:
                    self.additional_stopwords.add(line.strip().lower())
        except FileNotFoundError:
            # Log warning if stopwords file is not found
            logging.warning("No stopwords.txt file found. Skipping addition of additional stopwords.")
        
        # Set stopwords for the specified language
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
        
        # Log the number of additional stopwords added
        if self.stopwords_lang:
            logging.info(f"Added {len(self.stopwords_lang)} additional stopwords")
        else:
            logging.warning("No additional stopwords found. Skipping addition of additional stopwords.")
    
    def clean_docs(self, docs, lang=None):
        """
        Cleans the documents based on the specified language.

        Args:
            docs (list): The list of documents to be cleaned.
            lang (str): The language of the documents. Default is None.

        Returns:
            list: The cleaned documents.

        Raises:
            ValueError: If docs is None or lang is None.
            ValueError: If lang is invalid.
        """
        # Check if docs is not None
        if docs is None:
            raise ValueError("Documents must be specified")
        
        # Check if lang is specified
        if lang is None:
            raise ValueError("Language must be specified")
        
        processed_docs = []
        logging.debug(f"Cleaning {len(docs)} documents")

        # Determine the language of the documents and clean accordingly
        if lang == 'bilingual':
            processed_docs = self.bilingual_docs(docs)
        elif lang == 'fr':
            processed_docs = self.fr_docs(docs)
        elif lang == 'en':
            processed_docs = self.en_docs(docs)
        else:
            raise ValueError("Invalid language")
        
        return processed_docs
    
    def bilingual_docs(self, docs):
        logging.info("Cleaning bilingual documents")
        logging.debug(f"Documents to clean: {len(docs)}")

        try:
            flattened_docs = [item for sublist in docs for item in sublist]
            logging.debug(f"Flattened documents: {len(flattened_docs)}")

            merged_docs = {'fr': [], 'en': []}

            # Merge list of dictionaries into a single dictionary
            for doc in flattened_docs:
                if doc is None:
                    raise ValueError("Document is None")
                for key, value in doc.items():
                    if key in merged_docs:
                        merged_docs[key].append(value)
                    else:
                        merged_docs[key] = [value]
            logging.debug(f"Merged documents: {merged_docs}")

            translation_table = str.maketrans('', '', string.punctuation)

            merged_docs['fr'] = [value.translate(translation_table) for value in merged_docs['fr']]
            merged_docs['en'] = [value.translate(translation_table) for value in merged_docs['en']]
            logging.debug(f"Translated punctuation: {merged_docs}")

            merged_docs = {
                'fr': [value for value in merged_docs['fr'] if value],
                'en': [value for value in merged_docs['en'] if value]
            }
            logging.debug(f"Filtered empty strings: {merged_docs}")

            stopwords_lang = self.stopwords_lang
            merged_docs['fr'] = [value for value in merged_docs['fr'] if value.lower() not in stopwords_lang['fr']]
            merged_docs['en'] = [value for value in merged_docs['en'] if value.lower() not in stopwords_lang['en']]
            logging.debug(f"Filtered stopwords: {merged_docs}")

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
        """
        Cleans French documents by removing punctuation, stopwords, and digits.
        
        Args:
            docs (list): List of French documents to be cleaned.
        
        Returns:
            list: Cleaned French documents.
        """
        # Log start of document cleaning process
        logging.debug(f"Cleaning {len(docs)} documents")
        
        try:
            processed_docs = []  # List to store cleaned documents
            
            # Process each document in the list
            for doc in docs:
                if doc is None:
                    raise ValueError("Document is None")
                
                # Tokenize the document using French spaCy model
                tokens = self.nlp_fr(doc)
                
                # Filter tokens to remove punctuation, stopwords, and digits
                filtered_tokens = [token.text.lower() for token in tokens 
                                   if token.text.lower() not in self.stopwords_fr 
                                   and token.text not in string.punctuation 
                                   and not token.text.isdigit()]
                
                # Join the filtered tokens to form a processed document
                processed_doc = ' '.join(filtered_tokens)
                
                # Append cleaned document to the list if it is not empty
                if processed_doc and len(processed_doc) > 0:
                    processed_docs.append(processed_doc)
            
            # Return the list of cleaned documents
            return processed_docs
        
        # Log and return empty list if an error occurs during document cleaning
        except Exception as e:
            logging.error(f"Failed to clean documents. Error: {e}", exc_info=True)
            return []
        
    def en_docs(self, docs):
        """
        Cleans English documents by removing punctuation, stopwords, and digits.
        
        Args:
            docs (list): List of English documents to be cleaned.
        
        Returns:
            list: Cleaned English documents.
        """
        # Log start of document cleaning process
        logging.debug(f"Cleaning {len(docs)} documents")
        
        try:
            processed_docs = []  # List to store cleaned documents
            
            # Process each document in the list
            for doc in docs:
                if doc is None:
                    raise ValueError("Document is None")
                
                # Tokenize the document using English spaCy model
                tokens = self.nlp_en(doc)
                
                # Filter tokens to remove punctuation, stopwords, and digits
                filtered_tokens = [token.text.lower() for token in tokens 
                                   if token.text.lower() not in self.stopwords_en 
                                   and token.text not in string.punctuation 
                                   and not token.text.isdigit()]
                
                # Join the filtered tokens to form a processed document
                processed_doc = ' '.join(filtered_tokens)
                
                # Append cleaned document to the list if it is not empty
                if processed_doc and len(processed_doc) > 0:
                    processed_docs.append(processed_doc)
            
            # Return the list of cleaned documents
            return processed_docs
        
        # Log and return empty list if an error occurs during document cleaning
        except Exception as e:
            logging.error(f"Failed to clean documents. Error: {e}", exc_info=True)
            return []
