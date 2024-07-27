import os
import logging
import sqlite3
from langdetect import detect_langs
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer

from scripts.database import Database
from scripts.topic_analysis.tools import Tools
from scripts.topic_analysis.noise_remover import NoiseRemover
from scripts.topic_analysis.documents import Documents, French, English, Bilingual
from scripts.topic_analysis.text_processing import Process
from exceptions import AnalysisError, DatabaseError

class Analysis:
    """
    Class for performing topic analysis on documents.

    Attributes:
        db (str): Path to the database file.
        lang (str): Language of the documents.
        noise_remover (NoiseRemover): Object for removing noise from documents.
        tools (Tools): Object for performing various tasks.
        vectorizer (TfidfVectorizer or None): Object for vectorizing documents.
    """
    def __init__(self, db=os.path.join('data', 'database.db'), lang=None):
        """
        Initialize the database connection and cursor.

        Args:
            db (str): Path to the database.
            lang (str): Language of the documents.
        """
        try:
            self.db = Database(db)
            self.lang = lang
            self.noise_remover = NoiseRemover(lang)
            self.tools = Tools(lang)
            self.vectorizer = None
        except Exception as e:
            logging.error(f"Error initializing Analysis: {str(e)}")
            raise AnalysisError("Error initializing Analysis", error=e)

    def analyze_docs(self, doc_ids=None):
        """
        Process a list of documents and perform topic analysis.

        Args:
            doc_ids (List[int] or None): List of document IDs to process. If None, process all documents.

        Returns:
            List[List[str]]: Topics with their top words.
        """
        logging.info(f"Processing documents with lang={self.lang}")
        
        try:
            if doc_ids:
                docs = self._fetch_documents(doc_ids)
            else:
                docs = self.db.fetch_all(self.lang)

            if not docs:
                logging.error("No documents found in the database.")
                return []

            cleaned_docs = self.noise_remover.clean_docs([doc[1] for doc in docs], self.lang)

            self._initialize_vectorizer(cleaned_docs)

            tfidf_matrix = self.vectorizer.fit_transform(cleaned_docs)

            topics = self._perform_topic_modeling(tfidf_matrix)

            return topics
        except DatabaseError as e:
            logging.error(f"Error processing documents: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Error processing documents: {str(e)}")
            raise AnalysisError("Error processing documents", error=e)
        
    def _fetch_documents(self, doc_ids):
        try:
            docs = [self.db.fetch_single(doc_id) for doc_id in doc_ids]
            return [doc for doc in docs if doc]
        except Exception as e:
            logging.error(f"Error fetching documents: {str(e)}")
            raise DatabaseError("Error fetching documents", error=e)
        
    def _initialize_vectorizer(self, docs):
        n_docs = len(docs)
        self.vectorizer = TfidfVectorizer(
            max_df=1.0,
            min_df=1,
            max_features=100,
            ngram_range=(1,2),
            stop_words=list(self.tools.stopwords_lang)
        )

    def _perform_topic_modeling(self, tfidf_matrix):
        try:
            num_topics = min(5, tfidf_matrix.shape[0])
            lda_model = LatentDirichletAllocation(
                n_components=num_topics,
                random_state=42,
                max_iter=100,
                learning_method='online'
            )
            lda_output = lda_model.fit_transform(tfidf_matrix)
            logging.debug(f"LDA output shape: {lda_output.shape}")

            feature_names = self.vectorizer.get_feature_names_out()

            topics = []
            for topic_idx, topic in enumerate(lda_model.components_):
                top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
                topics.append(top_words)

            return topics
        except Exception as e:
            logging.error(f"Failed to perform topic modeling. Error: {str(e)}")
            raise AnalysisError("Failed to perform topic modeling", error=e)
