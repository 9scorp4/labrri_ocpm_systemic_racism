import logging
import string
from nltk.corpus import stopwords
from spacy.lang.fr import French
from spacy.lang.en import English

from scripts.topic_analysis.tools import Tools

class NoiseRemover:
    def __init__(self, lang):
        """
        Initializes the NoiseRemover object with the specified language.

        Args:
            lang (str): The language to use. Required.

        Raises:
            ValueError: If lang is None.
        """
        # Check if lang is specified
        if lang is None:
            raise ValueError("Language must be specified")
        
        # Initialize instance variables
        self.lang = lang  # The language used for text processing
        self.tools = Tools(lang)  # The Tools object for text processing
        self.nlp = self.tools.nlp  # The spaCy model for the specified language
        self.stopwords_lang = self.tools.stopwords_lang  # The stopwords for the specified language
    
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
        lang = lang or self.lang
        logging.info(f"Cleaning documents with lang={lang}")

        # Check inputs
        self._check_inputs(docs, lang)

        processed_docs = []
        logging.debug(f"Cleaning {len(docs)} documents")

        # Process each document in the list
        for i, doc in enumerate(docs):
            if doc is None:
                logging.debug(f"Skipping document {i} as it is None")
                continue

            # Convert the document to a string
            doc = self._convert_to_string(doc, i)

            # Tokenize the document based on the specified language
            tokens = self._tokenize_document(doc, lang, i)

            # Filter tokens to remove punctuation, stopwords, and digits
            filtered_tokens = self._filter_tokens(tokens, lang, i)

            # Join the filtered tokens to form a processed document
            processed_doc = self._join_filtered_tokens(filtered_tokens)

            # Append cleaned document to the list if it is not empty
            if processed_doc and len(processed_doc) > 0:
                processed_docs.append(processed_doc)

        logging.debug(f"Cleaned {len(processed_docs)} documents")
        return processed_docs

    def _check_inputs(self, docs, lang):
        """
        Check that the inputs are valid.
        """
        if docs is None:
            raise ValueError("Documents must be specified")
        if lang is None:
            raise ValueError("Language must be specified")

    def _convert_to_string(self, doc, index):
        """
        Convert the document to a string.
        """
        if isinstance(doc, list):
            doc = " ".join(doc)
        elif not isinstance(doc, str):
            doc = str(doc)
        logging.debug(f"Converted document {index} to string: {doc}")
        return doc

    def _tokenize_document(self, doc, lang, index):
        if lang not in self.nlp:
            raise ValueError(f"Invalid language: {lang}")
        tokens = self.nlp[lang](doc)
        logging.debug(f"Tokenized document {index} for language {lang}: {tokens}")
        return tokens

    def _filter_tokens(self, tokens, lang, index):
        filtered_tokens = [token.text.lower() for token in tokens
                           if token.text.lower() not in self.stopwords_lang[lang]
                           and token.text not in string.punctuation
                           and not token.is_digit]
        logging.debug(f"Filtered tokens for document {index} for language {lang}: {filtered_tokens}")
        return filtered_tokens

    def _join_filtered_tokens(self, filtered_tokens):
        """
        Join the filtered tokens to form a processed document.
        """
        return ' '.join(filtered_tokens)