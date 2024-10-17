from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import csv
from datetime import datetime
from pathlib import Path
from loguru import logger
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim import corpora, models
from gensim.corpora.dictionary import Dictionary
from gensim.models import KeyedVectors, CoherenceModel
from gensim.downloader import load
from scipy.sparse import csr_matrix
from tqdm import tqdm
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
import threading
from contextlib import contextmanager

from scripts.database import Database
from scripts.topic_analysis.tools import Tools
from scripts.topic_analysis.noise_remover import NoiseRemover
from scripts.topic_analysis.topic_labeler import TopicLabeler
from exceptions import AnalysisError, DatabaseError, VectorizationError

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
            logger.info("Analysis initialization completed successfully.")
        except Exception as e:
            logger.error(f"Error initializing Analysis: {str(e)}")
            raise AnalysisError("Error initializing Analysis", error=e)
        
        self.word2vec_model = self._load_word2vec_model()
        self.domain_terms = self._load_domain_terms()
        self.topic_labeler = TopicLabeler(self.vectorizer, self.domain_terms, self.lang)

    @staticmethod
    @contextmanager
    def timeout(seconds):
        timer = threading.Timer(seconds, lambda: (_ for _ in ()).throw(TimeoutError(f"Timed out after {seconds} seconds")))
        timer.start()
        try:
            yield
        finally:
            timer.cancel()

    def analyze_docs(self, docs, method='lda', num_topics=40, coherence_threshold=-5.0, min_topics=5, max_topics=20):
        try:
            logger.info(f"Starting document analysis with {method} method and {num_topics} initial topics")
            
            docs = list(docs)
            
            logger.info("Vectorizing documents...")
            tfidf_matrix = self.vectorize(docs)
            
            logger.info(f"Performing topic modeling with {method}...")
            topics = self._perform_topic_modeling(tfidf_matrix, method, num_topics)
            
            logger.info("Calculating topic coherence...")
            coherence_scores = self.calculate_topic_coherences(topics, docs)
            
            logger.info("Filtering topics by coherence...")
            filtered_topics = self.filter_topics_by_coherence(topics, coherence_scores, threshold=coherence_threshold, min_topics=min_topics, max_topics=max_topics)
            
            logger.info("Labeling topics...")
            labeled_topics = self._label_topics(filtered_topics, docs)
            
            result = [(label, words, score) for (label, words), score in zip(labeled_topics, coherence_scores)]
            
            logger.info("Analysis completed successfully")
            return result
        except Exception as e:
            logger.error(f"Error analyzing documents: {str(e)}", exc_info=True)
            raise AnalysisError("Error analyzing documents", error=e)

    def save_topics_to_csv(self, topics, filename=None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results/topic_analysis/topics_{timestamp}.csv"
        
        with open(filename, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Topic Number', 'Label', 'Words', 'Coherence Score'])
            
            for i, (label, words, score) in enumerate(topics, 1):
                writer.writerow([i, label, ', '.join(words), score])
        
        logger.info(f"Topics saved to {filename}")
        return filename

    def vectorize(self, docs):
        logger.debug(f"Vectorizing {len(docs)} documents...")

        if not isinstance(docs, list):
            docs = list(docs)

        # Check document content
        non_empty_docs = [doc for doc in docs if doc and isinstance(doc, str) and doc.strip()]
        if not non_empty_docs:
            logger.error("All documents are empty or non-string")
            return None
        
        # Print some stats about the documents
        doc_lengths = [len(doc.split()) for doc in non_empty_docs]
        logger.debug(f"Document length stats: min={min(doc_lengths)}, max={max(doc_lengths)}, avg={np.mean(doc_lengths):.2f}, median={np.median(doc_lengths):.2f}")

        if not self.vectorizer:
            self._initialize_vectorizer(non_empty_docs)

        try:
            tfidf_matrix = self.vectorizer.fit_transform(non_empty_docs)
            logger.debug(f"Vectorized {tfidf_matrix.shape[0]} documents with {tfidf_matrix.shape[1]} features")

            # Print top 10 most common words
            feature_names = self.vectorizer.get_feature_names_out()
            if len(feature_names) > 0:
                word_freq = np.asarray(tfidf_matrix.sum(axis=0)).ravel()
                top_words = [feature_names[i] for i in word_freq.argsort()[-10:][::-1]]
                logger.debug(f"Top 10 most common words: {', '.join(top_words)}")
            else:
                logger.warning("No features extracted during vectorization")

            return tfidf_matrix
        except ValueError as e:
            logger.error(f"Error vectorizing documents: {str(e)}")
            logger.debug(f"Vocabulary sixe: {len(self.vectorizer.vocabulary_)}")
            logger.error(f"Stopswords: {self.vectorizer.stop_words_}")
            raise VectorizationError("Error vectorizing documents", error=e)

    def _label_topics(self, topics, docs):
        logger.info(f"Labeling {len(topics)} topics...")

        if not hasattr(self, 'topic_labeler'):
            self.topic_labeler = TopicLabeler(self.vectorizer, self.domain_terms, self.lang)

        labeled_topics = []
        for i, topic in enumerate(topics):
            try:
                top_words = topic[:5]
                label = self.topic_labeler.generate_label(top_words, docs)
                labeled_topics.append((label, topic))
                logger.debug(f"Label for topic {i+1}: {label}")
            except Exception as e:
                logger.error(f"Error generating label for topic {i+1}: {str(e)}")
                fallback_label = f"Topic {i+1}: {' '.join(topic [:3])}"
                labeled_topics.append((fallback_label, topic))

        return labeled_topics
       
    def _perform_topic_modeling(self, tfidf_matrix, method, num_topics):
        if method == 'lda':
            model = LatentDirichletAllocation(
                n_components=num_topics,
                max_iter=50,
                learning_method='online',
                random_state=42,
                batch_size=128,
                doc_topic_prior=0.1,
                topic_word_prior=0.01,
                n_jobs=-1
            )
        elif method == 'nmf':
            model = NMF(n_components=num_topics, random_state=42)
        elif method == 'lsa':
            model = TruncatedSVD(n_components=num_topics, random_state=42)
        else:
            raise ValueError(f"Invalid topic modeling method: {method}")
        
        logger.debug(f"Fitting the model...")
        model.fit(tfidf_matrix)
        logger.debug("Model fitting completed")
        
        feature_names = self.vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(model.components_):
            top_features_ind = topic.argsort()[:-21:-1]
            top_features = [feature_names[i] for i in top_features_ind]
            topics.append(top_features)
            logger.debug(f"Topic {topic_idx}: {top_features[:5]}...")

        logger.info(f"Generated {len(topics)} topics")
        return topics
        
    def _calculate_word_doc_freq(self, docs):
        vectorizer = CountVectorizer(lowercase=True, token_pattern=r'\b\w+\b')
        X = vectorizer.fit_transform(docs)
        word_doc_freq = defaultdict(int)
        feature_names = vectorizer.get_feature_names_out()

        # Calculate document frequency
        doc_freq = np.bincount(X.indices, minlength=X.shape[1])

        for i, word in enumerate(feature_names):
            word_doc_freq[word] = doc_freq[i]

        logger.info(f"Number of unique words in corpus: {len(word_doc_freq)}")
        return word_doc_freq, X

    def _load_word2vec_model(self):
        model_path = Path(__file__).parent.parent.parent / 'data' / 'word2vec-google-news-300.model'

        if not model_path.exists():
            logger.info(f"Downloading Word2Vec model. This may take a while...")
            model = load('word2vec-google-news-300')
            model.save(str(model_path))
            logger.info(f"Word2Vec model saved to {model_path}")
        else:
            logger.info("Loading Word2Vec model...")
            model = KeyedVectors.load(str(model_path))

        logger.info("Word2Vec model loaded successfully.")
        return model

    def _load_domain_terms(self):
        filepath = Path(__file__).parent.parent / 'topic_analysis' / 'domain_terms.txt'

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                terms = [line.strip().lower() for line in f if line.strip()]
            logger.info(f"Successfully loaded {len(terms)} domain terms from {filepath}")
            return terms
        except FileNotFoundError:
            logger.error(f"Domain terms file not found at {filepath}. Using an empty list.")
            return []
        except Exception as e:
            logger.error(f"Error loading domain terms: {str(e)}")
            return []
        
    def _fetch_documents(self, doc_ids):
        try:
            docs = []
            for doc_id in doc_ids:
                doc = self.db.fetch_single(doc_id)
                if doc:
                    docs.append(doc)
                else:
                    logger.warning(f"Document with id {doc_id} not found")
            logger.debug(f"Fetched {len(docs)} documents")
            return docs
        except Exception as e:
            logger.error(f"Error fetching documents: {str(e)}")
            raise DatabaseError("Error fetching documents", error=e)
        
    def _initialize_vectorizer(self, docs):
        n_docs = len(docs)
        max_df = max(2, int(0.95 * n_docs))
        min_df = min(2, max(1, int(0.05 * n_docs)))

        if self.lang == 'bilingual':
            stop_words = list(set(self.tools.stopwords_lang['fr']).union(self.tools.stopwords_lang['en']))
        else:
            stop_words = list(self.tools.stopwords_lang.get(self.lang, []))

        self.vectorizer = TfidfVectorizer(
            max_df=max_df,
            min_df=min_df,
            max_features=1000,
            ngram_range=(1,2),
            stop_words=stop_words
        )
        logger.info(f"Initialized vectorizer with max_df={max_df}, min_df={min_df}")

    def _perform_lda(self, tfidf_matrix, num_topics):
        lda_model = LatentDirichletAllocation(
            n_components=num_topics,
            random_state=42,
            max_iter=10,
            learning_method='online',
        )
        lda_output = lda_model.fit_transform(tfidf_matrix)
        return self._extract_topics(lda_model)

    def _perform_nmf(self, tfidf_matrix, num_topics):
        nmf_model = NMF(
            n_components=num_topics,
            random_state=42,
            max_iter=1000,
        )
        nmf_output = nmf_model.fit_transform(tfidf_matrix)
        return self._extract_topics(nmf_model)

    def _perform_lsa(self, tfidf_matrix, num_topics):
        lsa_model = TruncatedSVD(
            n_components=num_topics,
            random_state=42
        )
        lsa_output = lsa_model.fit_transform(tfidf_matrix)
        return self._extract_topics(lsa_model)
    
    def _perform_dynamic_topic_modeling(self, docs, num_topics, time_slices=None):
        if time_slices is None:
            time_slices = [len(docs)]
        
        dictionary = corpora.Dictionary(docs)
        corpus = [dictionary.doc2bow(doc) for doc in docs]

        ldaseq = models.LdaSeqModel(
            corpus=corpus,
            id2word=dictionary,
            time_slice=time_slices,
            num_topics=num_topics,
            chunksize=1
        )
        topics = []
        for t in range(len(time_slices)):
            topic_terms = ldaseq.print_topics(time=t, top_terms=10)
            topics.append(topic_terms)
        return topics
    
    def _extract_topics(self, model):
        feature_names = self.vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(model.components_):
            top_features_ind = topic.argsort()[:-11:-1]
            top_features = [feature_names[i] for i in top_features_ind]
            topics.append(top_features)
        return topics
    
    def _find_similar_words(self, key_word, word_list, n=10):
        try:
            valid_words = [word for word in word_list if word in self.word2vec_model.key_to_index]

            if not valid_words:
                return word_list
            
            if key_word not in self.word2vec_model.key_to_index:
                key_word = valid_words[0]

            similar_words = self.word2vec_model.most_similar(key_word, topn=n)
            combined_words = list(dict.fromkeys(valid_words + [word for word, _ in similar_words]))

            return combined_words[:n]
        except KeyError:
            logger.warning(f"Word {key_word} not found in Word2Vec model")
            return word_list
        except Exception as e:
            logger.error(f"Error finding similar words: {str(e)}")
            return word_list
    
    def _identify_key_theme(self, words):
        domain_words = [word for word in words if any(term in word.lower() for term in self.domain_terms)]

        if domain_words:
            word_counts = Counter(domain_words)
            return word_counts.most_common(1)[0][0]
        else:
            word_counts = Counter(words)
            return word_counts.most_common(1)[0][0] if word_counts else None

    @staticmethod
    def calculate_coherence(topic, texts, dictionary, corpus):
        cm = CoherenceModel(model=topic, texts=texts, dictionary=dictionary, corpus=corpus, coherence='c_v')
        return cm.get_coherence()

    def calculate_topic_coherences(self, topics, docs):
        logger.info('Calculating topic coherence...')

        # Convert docs to a list if it's a generator
        docs = list(docs)
        
        # Tokenize documents and create vocabulary
        tokenized_docs = [doc.lower().split() for doc in docs]
        vocab = list(set(word for doc in tokenized_docs for word in doc))
        word_to_id = {word: i for i, word in enumerate(vocab)}
        
        # Create document-term matrix
        rows, cols, data = [], [], []
        for doc_id, doc in enumerate(tokenized_docs):
            for word in doc:
                if word in word_to_id:
                    rows.append(doc_id)
                    cols.append(word_to_id[word])
                    data.append(1)

        doc_term_matrix = csr_matrix((data, (rows, cols)), shape=(len(tokenized_docs), len(vocab)))

        doc_freqs = np.array((doc_term_matrix > 0).sum(0)).flatten()

        coherence_scores = []

        for topic in tqdm(topics, desc="Calculating topic coherence"):
            topic_word_ids = [word_to_id[word] for word in topic if word in word_to_id]
            if len(topic_word_ids) < 2:
                coherence_scores.append(0)
                continue

            topic_coherence = 0
            pairs_count = 0
            for i, word_id1 in enumerate(topic_word_ids[:-1]):
                for word_id2 in topic_word_ids[i + 1:]:
                    # Calculate co-document frequency
                    co_doc_freq = (doc_term_matrix[:, word_id1].multiply(doc_term_matrix[:, word_id2])).sum()
                    coherence = np.log((co_doc_freq + 1) / (doc_freqs[word_id1] + 1))
                    topic_coherence += coherence
                    pairs_count += 1
            
            if pairs_count > 0:
                topic_coherence /= pairs_count
            else:
                topic_coherence = 0
            coherence_scores.append(topic_coherence)

        logger.info(f"Calculated coherence scores: {coherence_scores}")
        return coherence_scores

    def filter_topics_by_coherence(self, topics, coherence_scores, threshold=-5.0, min_topics=10, max_topics=20):
        if len(topics) != len(coherence_scores):
            raise ValueError("The number of topics and coherence scores must match.")

        # Higher (closer to 0) scores are better
        filtered_topics = [(topic, score) for topic, score in zip(topics, coherence_scores) if score >= threshold]
        filtered_topics.sort(key=lambda x: x[1], reverse=True)

        if len(filtered_topics) < min_topics:
            logger.warning(f"Less than {min_topics} topics left after filtering. Keeping top {min_topics} topics.")
            filtered_topics = sorted(zip(topics, coherence_scores), key=lambda x: x[1], reverse=True)[:min_topics]
        elif len(filtered_topics) > max_topics:
            logger.info(f"More than {max_topics} topics left after filtering. Keeping top {max_topics} topics.")
            filtered_topics = filtered_topics[:max_topics]

        logger.info(f"Filtered topics from {len(topics)} to {len(filtered_topics)} based on coherence scores.")
        return [topic for topic, _ in filtered_topics]