import logging
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer

class Documents:
    def __init__(self, topic_analysis, lang):
        self.topic_analysis = topic_analysis
        self.lang = lang
        self.vectorizer = TfidfVectorizer(max_df=0.09, min_df=4, max_features=100, ngram_range=(1, 2))

    def vectorize(self, docs):
        try:
            logging.debug(f"Vectorizing {len(docs)} documents")
            tfidf_matrix = self.vectorizer.fit_transform(docs)
            logging.debug(f"Vectorized {len(docs)} documents")
            return tfidf_matrix
        except Exception as e:
            logging.error(f"Failed to vectorize. Error: {e}", exc_info=True)

    def topic_modeling(self, tfidf_matrix, num_topics=5):
        try:
            lda_model = LatentDirichletAllocation(n_components=num_topics, max_iter=5, learning_method='online')
            lda_model.fit(tfidf_matrix)
            return lda_model
        except Exception as e:
            logging.error(f"Failed to topic modeling. Error: {e}", exc_info=True)

    def plot_topics(self, topics, tfidf_vectorizer, n_top_words=10):
        try:
            feature_names = tfidf_vectorizer.get_feature_names_out()
            for topic_id, topic in enumerate(topics):
                print(f"Topic {topic_id + 1}:")
                top_word_indices = topic.argsort()[:-n_top_words - 1:-1]
                top_words = [feature_names[i] for i in top_word_indices]
                print(', '.join(top_words))
        except Exception as e:
            logging.error(f"Failed to plot topics. Error: {e}", exc_info=True)

    def analyze_lang(self, sentences, lang):
        logging.debug(f"Starting analysis for {lang} with sentences: {sentences}")
        tfidf_matrix = self.vectorize(sentences)
        if tfidf_matrix.shape[0] > 0:
            logging.debug(f"Starting LDA for {lang} with tfidf_matrix: {tfidf_matrix}")
        else:
            logging.warning(f"No sentences found for {lang}")
            return
        lda_model = self.topic_modeling(tfidf_matrix)
        print(f"\n{lang} topics:")
        self.plot_topics(lda_model.components_, self.vectorizer)

    def analyze(self, sentences_by_lang):
        logging.debug(f"Starting analysis for {self.lang} with sentences_by_lang: {sentences_by_lang}")
        if sentences_by_lang is None:
            raise ValueError("sentences_by_lang cannot be None")
        
        if self.lang == 'bilingual':
            if 'fr' in sentences_by_lang:
                logging.debug(f"Starting analysis for French with sentences_by_lang['fr']: {sentences_by_lang['fr']}")
                self.analyze_lang(sentences_by_lang['fr'], 'fr')
            if 'en' in sentences_by_lang:
                logging.debug(f"Starting analysis for English with sentences_by_lang['en']: {sentences_by_lang['en']}")
                self.analyze_lang(sentences_by_lang['en'], 'en')
        
        else:
            logging.debug(f"Retrieving sentences for {self.lang}")
            lang_sentences = sentences_by_lang[self.lang]
            if lang_sentences is None:
                raise ValueError(f"No sentences found for {self.lang}")
            logging.debug(f"Starting analysis for {self.lang} with sentences: {lang_sentences}")
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