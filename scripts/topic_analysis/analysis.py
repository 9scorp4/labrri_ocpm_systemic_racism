import os
from pathlib import Path
from loguru import logger
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim import corpora, models
from gensim.models import KeyedVectors, CoherenceModel
from gensim.downloader import load
import nltk
from nltk.util import ngrams
import numpy as np
from collections import Counter
from itertools import chain

from scripts.database import Database
from scripts.topic_analysis.tools import Tools
from scripts.topic_analysis.noise_remover import NoiseRemover
from scripts.topic_analysis.topic_labeler import TopicLabeler
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
            logger.info("Analysis initialization completed successfully.")
        except Exception as e:
            logger.error(f"Error initializing Analysis: {str(e)}")
            raise AnalysisError("Error initializing Analysis", error=e)
        
        self.domain_terms = self._load_domain_terms()
        self.word2vec_model = CountVectorizer(ngram_range=(1, 3))
        self.topic_labeler = None

    def analyze_docs(self, doc_ids=None, method='lda', num_topics=3, coherence_threshold=0.3):
        """
        Process a list of documents and perform topic analysis.

        Args:
            doc_ids (List[int] or None): List of document IDs to process. If None, process all documents.

        Returns:
            List[List[str]]: Topics with their top words.
        """
        logger.info(f"Processing documents with lang={self.lang}, method={method}")
        
        try:
            if doc_ids:
                docs = self._fetch_documents(doc_ids)
            else:
                docs = self.db.fetch_all(self.lang)

            if not docs:
                logger.error("No documents found in the database.")
                return []

            logger.info(f'Cleaning {len(docs)} documents...')
            cleaned_docs = self.noise_remover.clean_docs([doc[1] for doc in docs], self.lang)
            logger.info('Document cleaning completed.')

            if not cleaned_docs:
                logger.error("All documents were empty after cleaning.")
                return []
            
            logger.info('Initializing vectorizer...')
            self._initialize_vectorizer(cleaned_docs)

            logger.info('Vectorizing documents...')
            try:
                tfidf_matrix = self.vectorizer.fit_transform(cleaned_docs)
            except ValueError as ve:
                logger.error(f"Error vectorizing documents: {str(ve)}")
                logger.info('Attempting to vectorize with minimun settings...')
                self.vectorizer.set_params(max_df=1, min_df=1, max_features=None)
                tfidf_matrix = self.vectorizer.fit_transform(cleaned_docs)
            
            logger.info(f'Performing topic modeling using {method}...')
            if method == 'lda':
                topics = self._perform_lda(tfidf_matrix, num_topics)
            elif method == 'nmf':
                topics = self._perform_nmf(tfidf_matrix, num_topics)
            elif method == 'lsa':
                topics = self._perform_lsa(tfidf_matrix, num_topics)
            elif method == 'dynamic':
                topics = self._perform_dynamic_topic_modeling(cleaned_docs, num_topics)
            else:
                raise ValueError(f"Invalid method: {method}")
            
            logger.info('Filtering topics by coherence...')
            texts = [doc.split() for doc in cleaned_docs]
            filtered_topics = []
            coherence_scores = []
            for topic in topics:
                score = self.calculate_coherence_score([topic], texts)
                if score >= coherence_threshold:
                    filtered_topics.append(topic)
                    coherence_scores.append(score)
            
            logger.info('Labeling topics...')
            try:
                self.topic_labeler = TopicLabeler(self.vectorizer, self.domain_terms, self.lang)
                labeled_topics = self.topic_labeler.label_topics(filtered_topics, cleaned_docs)
            except Exception as e:
                logger.error(f"Error labeling topics: {str(e)}")
                labeled_topics = [(f"Topic {i+1}: {topic}", topic) for i, topic in enumerate(filtered_topics)]
            
            labeled_topics_with_scores = [
                (label, words, score) for (label, words), score in zip(labeled_topics, coherence_scores)
            ]

            logger.info('Topic analysis completed successfully.')
            logger.info(f"Topics with scores: {labeled_topics_with_scores}")
            return labeled_topics_with_scores
        except DatabaseError as e:
            logger.error(f"Error processing documents: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            raise AnalysisError("Error processing documents", error=e)

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
            return docs
        except Exception as e:
            logger.error(f"Error fetching documents: {str(e)}")
            raise DatabaseError("Error fetching documents", error=e)
        
    def _initialize_vectorizer(self, docs):
        n_docs = len(docs)
        max_df = max(2, int(0.95 * n_docs))
        min_df = min(2, max(1, int(0.05 * n_docs)))
        self.vectorizer = TfidfVectorizer(
            max_df=max_df,
            min_df=min_df,
            max_features=100,
            ngram_range=(1,2),
            stop_words=list(self.tools.stopwords_lang.get(self.lang, []))
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
    
    def calculate_coherence_score(self, topics, texts):
        try:
            id2word = corpora.Dictionary(texts)
            corpus = [id2word.doc2bow(text) for text in texts]
            
            coherence_model = CoherenceModel(topics=topics,
                                            texts=texts,
                                            dictionary=id2word,
                                            coherence='c_v')
            coherence_score = coherence_model.get_coherence()
            
            logger.info(f"Coherence score: {coherence_score}")
            return coherence_score
        except Exception as e:
            logger.error(f"Error calculating coherence score: {str(e)}")
            return None
        
    def filter_topics_by_coherence(self, topics, texts, threshold=0.3):
        try:
            coherence_scores = []
            for topic in topics:
                score = self.calculate_coherence_score([topic], texts)
                if score is not None:
                    coherence_scores.append(score)

            if not coherence_scores:
                logger.warning("No coherence scores found. Skipping filtering.")
                return topics

            filtered_topics = [topic for topic, score in zip(topics, coherence_scores) if score >= threshold]

            logger.info(f"Filtered topics: {len(filtered_topics)} out of {len(topics)} topics kept.")
            return filtered_topics
        except Exception as e:
            logger.error(f"Error filtering topics by coherence: {str(e)}")
            return topics
                

    def topic_diversity(self, topics):
        # To be implemented
        pass