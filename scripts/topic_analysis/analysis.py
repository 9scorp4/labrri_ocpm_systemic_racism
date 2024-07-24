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
        self.db = Database(db)
        self.lang = lang
        self.noise_remover = NoiseRemover(lang)
        self.tools = Tools(lang)
        self.vectorizer = None

    def process_documents(self, doc_ids=None):
        """
        Process a list of documents and perform topic analysis.

        Args:
            doc_ids (List[int] or None): List of document IDs to process. If None, process all documents.

        Returns:
            List[List[str]]: Topics with their top words.
        """
        if doc_ids:
            docs = [self.db.fetch_single(doc_id) for doc_id in doc_ids]
            docs = [doc for doc in docs if doc]
        else:
            docs = self.db.fetch_all(self.lang)

        if not docs:
            logging.error("No documents found in the database.")
            return []

        # Clean and preprocess the documents
        cleaned_docs = self.noise_remover.clean_docs([doc[1] for doc in docs], self.lang)

        # Initialize the vectorizer with parameters adjusted for the current document set
        n_docs = len(cleaned_docs)
        self.vectorizer = TfidfVectorizer(
            max_df=0.95 if n_docs > 1 else 1.0,
            min_df=1,
            max_features=100,
            ngram_range=(1, 2),
            stop_words=list(self.tools.stopwords(self.lang))
        )

        # Vectorize the documents
        tfidf_matrix = self.vectorizer.fit_transform(cleaned_docs)

        # Perform topic modeling
        num_topics = 5
        lda_model = LatentDirichletAllocation(
            n_components=num_topics,
            random_state=42,
            max_iter=100,
            learning_method='online'
        )
        lda_output = lda_model.fit_transform(tfidf_matrix)
        logging.debug(lda_output.shape)

        # Get the feature names
        feature_names = self.vectorizer.get_feature_names_out()

        # Extract topics
        topics = []
        for topic_idx, topic in enumerate(lda_model.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
            topics.append(top_words)

        return topics
