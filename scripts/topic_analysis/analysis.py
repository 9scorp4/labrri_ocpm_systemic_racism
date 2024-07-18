import os
import logging
import sqlite3
from langdetect import detect_langs
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
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
        self.db = db
        self.lang = lang
        self.noise_remover = NoiseRemover(lang)
        self.tools = Tools(lang)
        self.vectorizer = None

    def fetch_all(self):
        """
        Fetch all documents from the database.

        Returns:
            List[Tuple[int, str]]: Documents with their IDs and content.
        """
        try:
            with sqlite3.connect(self.db) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT d.id, c.content, d.language
                    FROM documents d
                    JOIN content c ON d.id = c.doc_id
                    WHERE d.language = ?
                """, (self.lang,))
                return cursor.fetchall()
        except Exception as e:
            logging.error(f"Error fetching documents from the database: {e}", exc_info=True)
            return []

    def fetch_single(self, doc_id):
        """
        Fetch a single document from the database by ID.

        Args:
            doc_id (int): ID of the document.

        Returns:
            Tuple[int, str] or None: Document with its ID and content, or None if not found.
        """
        try:
            with sqlite3.connect(self.db) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT doc_id, content FROM content WHERE doc_id=?", (doc_id,))
                return cursor.fetchone()
        except Exception as e:
            logging.error(f"Error fetching document {doc_id} from the database: {e}", exc_info=True)
            return None

    def process_documents(self, docs):
        """
        Process a list of documents and perform topic analysis.

        Args:
            docs (List[Tuple[int, str]]): Documents with their IDs and content.
            single_doc (bool): Whether to treat the documents as a single document or multiple documents.

        Returns:
            List[List[str]]: Topics with their top words.
        """
        if not docs:
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
