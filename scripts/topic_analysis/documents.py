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
        self.vectorizer = TfidfVectorizer(
            max_df=0.09,  # Drop terms that are present in more than 90% of the documents
            min_df=4,  # Drop terms that are present in less than 4 documents
            max_features=100,  # Maximum number of features (terms) to consider
            ngram_range=(1, 2)  # Range of n-grams to consider
        )

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
        # Check if docs is None
        if docs is None:
            raise ValueError("docs cannot be None")
        
        try:
            # Log the number of documents being vectorized
            logging.debug(f"Vectorizing {len(docs)} documents")
            
            # Fit the vectorizer on the documents and transform them to TF-IDF features
            tfidf_matrix = self.vectorizer.fit_transform(docs)
            
            # Log the number of documents that were vectorized
            logging.debug(f"Vectorized {len(docs)} documents")
            
            return tfidf_matrix
        except Exception as e:
            # Log the error that occurred during vectorization
            logging.error(f"Failed to vectorize. Error: {e}", exc_info=True)

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
            # Instantiate a LatentDirichletAllocation model with the given number of topics
            lda_model = LatentDirichletAllocation(n_components=num_topics, max_iter=5, learning_method='online')
            
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
        try:
            # Check if topics or tfidf_vectorizer is None
            if topics is None or tfidf_vectorizer is None:
                raise ValueError("topics and tfidf_vectorizer cannot be None")

            # Get the feature names from the tfidf_vectorizer
            feature_names = tfidf_vectorizer.get_feature_names_out()

            # Iterate over each topic
            for topic_id, topic in enumerate(topics):
                # Print the topic number
                print(f"Topic {topic_id + 1}:")
                # Get the indices of the top words
                top_word_indices = topic.argsort()[:-n_top_words - 1:-1]
                # Get the top words from the feature names
                top_words = [feature_names[i] for i in top_word_indices]
                # Print the top words
                print(', '.join(top_words))
        except Exception as e:
            # Log the error that occurred during plotting
            logging.error(f"Failed to plot topics. Error: {e}", exc_info=True)

    def analyze_lang(self, sentences, lang):
        """
        Perform analysis for a given language.

        Args:
            sentences (list): List of sentences.
            lang (str): Language of the sentences.

        Returns:
            None

        Raises:
            ValueError: If sentences is None.
            Exception: If any error occurs during analysis.
        """
        # Check if sentences is None
        if sentences is None:
            raise ValueError("sentences cannot be None")

        # Log the start of the analysis for the given language and sentences
        logging.debug(f"Starting analysis for {lang} with sentences: {sentences}")

        # Vectorize the sentences using TF-IDF
        tfidf_matrix = self.vectorize(sentences)

        # If no sentences are found for the given language, log a warning and return None
        if tfidf_matrix.shape[0] == 0:
            logging.warning(f"No sentences found for {lang}")
            return

        # Log the start of LDA for the given language and TF-IDF matrix
        logging.debug(f"Starting LDA for {lang} with tfidf_matrix: {tfidf_matrix}")

        # Perform topic modeling on the TF-IDF matrix
        lda_model = self.topic_modeling(tfidf_matrix)

        # Print the topics for the given language
        print(f"\n{lang} topics:")

        # Plot the top words for each topic
        self.plot_topics(lda_model.components_, self.vectorizer)

    def analyze(self, sentences_by_lang):
        """
        Perform analysis for a given language or languages.

        Args:
            sentences_by_lang (dict): Dictionary of sentences by language.

        Raises:
            ValueError: If sentences_by_lang is None or if no sentences are found for the given language or languages.
        """
        # Check if sentences_by_lang is None
        if sentences_by_lang is None:
            raise ValueError("sentences_by_lang cannot be None")
        
        # Check if the analysis is for bilingual languages
        if self.lang == 'bilingual':
            # If French sentences are provided, analyze them
            if 'fr' in sentences_by_lang:
                lang_sentences = sentences_by_lang['fr']
                if lang_sentences is None:
                    raise ValueError("No sentences found for French")
                self.analyze_lang(lang_sentences, 'fr')
            # If English sentences are provided, analyze them
            if 'en' in sentences_by_lang:
                lang_sentences = sentences_by_lang['en']
                if lang_sentences is None:
                    raise ValueError("No sentences found for English")
                self.analyze_lang(lang_sentences, 'en')
        
        # If the analysis is for a single language
        else:
            # Check if sentences are provided for the given language
            if self.lang not in sentences_by_lang:
                raise ValueError(f"No sentences found for {self.lang}")
            lang_sentences = sentences_by_lang[self.lang]
            if lang_sentences is None:
                raise ValueError(f"No sentences found for {self.lang}")
        
        # Log the start of analysis for the given language and sentences
        logging.debug(f"Starting analysis for {self.lang} with sentences: {lang_sentences}")
        # Analyze the sentences for the given language
        self.analyze_lang(lang_sentences, self.lang)

class French(Documents):
    def __init__(self, topic_analysis, lang='fr'):
        super().__init__(topic_analysis, lang)
    
    def analyze(self, french_sentences):
        super().analyze(french_sentences)

class English(Documents):
    def __init__(self, topic_analysis, lang='en'):
        super().__init__(topic_analysis, lang)
    
    def analyze(self, english_sentences):
        super().analyze(english_sentences)

class Bilingual(Documents):
    def __init__(self, topic_analysis, lang='bilingual'):
        super().__init__(topic_analysis, lang)
    
    def analyze(self, french_sentences, english_sentences):
        print(f"\nBilingual Topics:\n")
        print("English Topics:")
        super().analyze(english_sentences)
        print("French Topics:")
        super().analyze(french_sentences)