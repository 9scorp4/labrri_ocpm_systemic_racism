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

            self.vectorizer = TfidfVectorizer(
                max_df=0.95,
                min_df=2,
                max_features=1000,
                ngram_range=(1,1)
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

    def topic_modeling(self, tfidf_matrix, num_topics=3, num_words=10):
        """
        Topic modeling using Latent Dirichlet Allocation.

        Args:
            tfidf_matrix (csr_matrix): Matrix of TF-IDF features.
            num_topics (int): Number of topics to be extracted. Default is 5.
            num_words (int): Number of words to display for each topic. Default is 10.

        Raises:
            ValueError: If tfidf_matrix is None.
            Exception: If topic modeling fails.

        Returns:
            lda_model (LatentDirichletAllocation): Trained LatentDirichletAllocation model.
            feature_names (list): List of feature names (words) from the vectorizer.
        """
        # Check if tfidf_matrix is None
        if tfidf_matrix is None:
            raise ValueError("tfidf_matrix cannot be None")
        
        try:
            # Log the start of topic modeling
            logging.debug("Starting topic modeling")

            # Instantiate a LatentDirichletAllocation model with the given number of topics
            lda_model = LatentDirichletAllocation(n_components=num_topics,
                                                  max_iter=10,
                                                  learning_method='online', 
                                                  random_state=42)

            # Fit the model and get the output
            lda_output = lda_model.fit_transform(tfidf_matrix)

            # Log the number of topics
            logging.debug(f"Topic modeling using {num_topics} topics")

            # Get feature names
            feature_names = self.vectorizer.get_feature_names_out()

            # Get top words for each topic
            topics = []
            for topic_idx, topic in enumerate(lda_model.components_):
                # Log the topic being processed
                logging.debug(f"Processing topic {topic_idx + 1}")
                top_features_ind = topic.argsort()[:-num_words - 1:-1]
                top_features = [feature_names[i] for i in top_features_ind]
                topics.append(top_features)

            # Log the number of topics found
            logging.debug(f"Found {len(topics)} topics")

            # Return the trained model, feature names and top words per topic
            return topics
        except Exception as e:
            # Log the error that occurred during topic modeling
            logging.error(f"Failed to topic modeling. Error: {e}", exc_info=True)
            return None, None, None

    def plot_topics(self, top_words_per_topic):
        for topic_idx, top_words in enumerate(top_words_per_topic):
            print(f"Topic {topic_idx + 1}:")
            print(", ".join(top_words))
            print()

    def analyze_lang(self, docs, lang):
        if not docs:
            raise ValueError("docs cannot be None or empty")
        
        logging.debug(f"Starting analysis for {lang} with {len(docs)} documents")

        tfidf_matrix = self.vectorize(docs)

        if tfidf_matrix is None or tfidf_matrix.shape[0] == 0:
            logging.warning(f"No documents found for {lang}")
            return
        
        logging.debug(f"Starting LDA for {lang} with tfidf_matrix shape: {tfidf_matrix.shape}")

        topics = self.topic_modeling(tfidf_matrix, num_topics=3, num_words=10)

        if topics is None:
            logging.warning(f"Failed to create LDA model for {lang}")
            return
        
        print(f"\n{lang.upper()} Topics:\n")
        for i, topic in enumerate(topics, 1):
            print(f"Topic {i}: {', '.join(word for word in topic if word)}")
            print()

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