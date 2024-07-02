import logging
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer

class Documents:
    def __init__(self, topic_analysis, lang):
        """
        Initialize Documents class.

        Args:
            topic_analysis (TopicAnalysis): Instance of TopicAnalysis class.
            lang (str): Language of the documents.

        Raises:
            ValueError: If topic_analysis is None or lang is None.
        """
        # Check if topic_analysis is None
        if not topic_analysis:
            raise ValueError("topic_analysis cannot be None")
        # Check if lang is None
        if not lang:
            raise ValueError("lang cannot be None")

        # Assign topic_analysis to self.topic_analysis
        self.topic_analysis = topic_analysis
        # Assign lang to self.lang
        self.lang = lang
        # Initialize TfidfVectorizer with the following parameters
        self.vectorizer = None

    def vectorize(self, docs):
        """
        Vectorize the input documents using TfidfVectorizer.

        Args:
            docs (list): List of documents to be vectorized.

        Returns:
            tfidf_matrix (csr_matrix): Matrix of TF-IDF features.

        Raises:
            ValueError: If docs is None.
            Exception: If vectorization fails.
        """
        if not docs:
            raise ValueError("docs cannot be None or empty")
        
        try:
            logging.debug(f"Vectorizing {len(docs)} documents")

            # Extract text content from (doc_id, content) tuples
            doc_texts = []
            for doc in docs:
                if isinstance(doc[1], list):
                    content = ' '.join(map(str, doc[1]))
                else:
                    content = doc[1]

                if content and isinstance(content, str) and content.strip():
                    doc_texts.append(content)

            if not doc_texts:
                logging.warning("No documents were extracted; documents may be empty or contain only stopwords.")
                return None
            
            # Adjust vectorizer parameters based on the number of documents
            n_docs = len(doc_texts)
            max_df = min(0.9, 1.0)
            min_df = min(4, max(1, int(n_docs * 0.1)))

            self.vectorizer = TfidfVectorizer(
                max_df=max_df,
                min_df=min_df,
                max_features=100,
                ngram_range=(1,2)
            )
            
            tfidf_matrix = self.vectorizer.fit_transform(doc_texts)

            if tfidf_matrix.shape[1] == 0:
                logging.warning("No features were extracted; documents may be empty or contain only stopwords.")
                return None
            
            logging.debug(f"Vectorized {tfidf_matrix.shape[0]} documents with {tfidf_matrix.shape[1]} features")

            return tfidf_matrix
        except Exception as e:
            # Log the error that occurred during vectorization
            logging.error(f"Failed to vectorize. Error: {e}", exc_info=True)
            return None

    def topic_modeling(self, tfidf_matrix, num_topics=5):
        """
        Perform topic modeling on the input TF-IDF matrix.

        Args:
            tfidf_matrix (csr_matrix): Matrix of TF-IDF features.
            num_topics (int): Number of topics to model. Default is 5.

        Returns:
            lda_model (LatentDirichletAllocation): Trained Latent Dirichlet Allocation model.

        Raises:
            ValueError: If tfidf_matrix is None.
            Exception: If topic modeling fails.
        """
        # Check if tfidf_matrix is None
        if tfidf_matrix is None:
            raise ValueError("tfidf_matrix cannot be None")
        
        try:
            # Adjjust number of topics based on the number of documents
            num_topics = min(num_topics, tfidf_matrix.shape[0])

            # Instantiate a LatentDirichletAllocation model with the given number of topics
            lda_model = LatentDirichletAllocation(n_components=num_topics, max_iter=5, learning_method='online', random_state=42)
            
            # Fit the model on the tfidf_matrix
            lda_model.fit(tfidf_matrix)
            
            # Return the trained model
            return lda_model
        except Exception as e:
            # Log the error that occurred during topic modeling
            logging.error(f"Failed to topic modeling. Error: {e}", exc_info=True)

    def plot_topics(self, topics, tfidf_vectorizer, n_top_words=10):
        """
        Plot the top words for each topic.

        Args:
            topics (list): List of topic vectors.
            tfidf_vectorizer (TfidfVectorizer): TF-IDF vectorizer used for fitting the data.
            n_top_words (int): Number of top words to display for each topic. Default is 10.

        Raises:
            ValueError: If topics or tfidf_vectorizer is None.
            Exception: If plotting fails.
        """
        # Check if topics or tfidf_vectorizer is None
        if topics is None or tfidf_vectorizer is None:
            raise ValueError("topics and tfidf_vectorizer cannot be None")

        try:
            feature_names = tfidf_vectorizer.get_feature_names_out()

            for topic_id, topic in enumerate(topics):
                print(f"Topic {topic_id + 1}:")
                top_word_indices = topic.argsort()[:-n_top_words - 1:-1]
                top_words = [feature_names[i] for i in top_word_indices]
                print(', '.join(top_words))
        except Exception as e:
            logging.error(f"Failed to plot topics. Error: {e}", exc_info=True)

    def analyze_lang(self, docs, lang):
        if not docs:
            raise ValueError("docs cannot be None or empty")
        
        logging.debug(f"Starting analysis for {lang} with {len(docs)} documents")

        tfidf_matrix = self.vectorize(docs)

        if tfidf_matrix is None or tfidf_matrix.shape[0] == 0:
            logging.warning(f"No documents found for {lang}")
            return
        
        logging.debug(f"Starting LDA for {lang} with tfidf_matrix shape: {tfidf_matrix.shape}")

        lda_model = self.topic_modeling(tfidf_matrix)

        if lda_model is None:
            logging.warning(f"Failed to create LDA model for {lang}")
            return
        
        print(f"\n{lang} Topics:\n")
        self.plot_topics(lda_model.components_, self.vectorizer)

    def analyze(self, docs):
        if not docs:
            raise ValueError("docs cannot be None or empty")
        
        if self.lang == 'bilingual':
            fr_docs = [doc for doc in docs if doc[1].get('fr')]
            en_docs = [doc for doc in docs if doc[1].get('en')]

            if fr_docs:
                self.analyze_lang([(doc_id, content['fr']) for doc_id, content in fr_docs], 'fr')
            if en_docs:
                self.analyze_lang([(doc_id, content['en']) for doc_id, content in en_docs], 'en')
        else:
            self.analyze_lang(docs, self.lang)
        

class French(Documents):
    def __init__(self, topic_analysis, lang='fr'):
        super().__init__(topic_analysis, lang)

class English(Documents):
    def __init__(self, topic_analysis, lang='en'):
        super().__init__(topic_analysis, lang)

class Bilingual(Documents):
    def __init__(self, topic_analysis, lang='bilingual'):
        super().__init__(topic_analysis, lang)