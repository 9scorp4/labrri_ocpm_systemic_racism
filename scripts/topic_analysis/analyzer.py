"""Enhanced topic analyzer with improved language handling and customization."""
import numpy as np
from loguru import logger
from datetime import datetime
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Optional, Dict, Tuple, Any, Union, Callable, Set
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import asyncio
import pandas as pd
from pathlib import Path
import aiofiles
import spacy

from scripts.language_detector import LanguageDetector
from scripts.topic_analysis.word_vectors import WordVectorManager
from scripts.topic_analysis.matrix_utils import MatrixUtils
from scripts.topic_analysis.topic_labeler import TopicLabeler
from scripts.topic_analysis.ta_text_processing import Process
from scripts.topic_analysis.error_handlers import ErrorHandler, AnalysisErrorType

class TopicAnalyzer:
    """Enhanced topic analyzer with improved language handling."""
    
    MODEL_CONFIGS = {
        'lda': {
            'class': LatentDirichletAllocation,
            'params': {
                'random_state': 42,
                'n_jobs': -1,
                'max_iter': 30,
                'learning_method': 'online',
                'batch_size': 256
            }
        },
        'nmf': {
            'class': NMF,
            'params': {
                'random_state': 42,
                'init': 'nndsvdar',
                'max_iter': 300
            }
        },
        'lsa': {
            'class': TruncatedSVD,
            'params': {
                'random_state': 42,
                'algorithm': 'randomized'
            }
        }
    }

    def __init__(self, db_path: Optional[str], lang: str = 'bilingual'):
        """Initialize analyzer with language support."""
        self.db_path = db_path
        self.lang = lang

        # Initialize thread pool executor
        self._executor = ThreadPoolExecutor(max_workers=multiprocessing.cpu_count())
        self._initialization_lock = asyncio.Lock()
        self._initialized = False
        self._stopwords_path = Path('scripts/topic_analysis/stopwords.txt')
        
        # Initialize components
        self.text_processor = None
        self.word_vectors = None
        self.matrix_utils = None
        self.error_handler = None
        self.topic_labeler = None
        self.language_detector = None
        self.nlp = {}
        
        # Performance tracking
        self.performance_metrics = {
            'processing_time': [],
            'memory_usage': [],
            'document_count': [],
            'coherence_scores': []
        }
        
        logger.info(f"Topic analyzer initialized for {lang}")

    async def _load_stopwords(self) -> Set[str]:
        """Load stopwords from file asynchronously."""
        try:
            if not self._stopwords_path.exists():
                logger.warning(f"Stopwords file not found at {self._stopwords_path}")
                return set()

            async with aiofiles.open(self._stopwords_path, mode='r', encoding='utf-8') as f:
                content = await f.read()
                stopwords = {
                    word.strip().lower() 
                    for word in content.splitlines() 
                    if word.strip() and not word.startswith('#')
                }
                logger.info(f"Loaded {len(stopwords)} stopwords from {self._stopwords_path}")
                return stopwords

        except Exception as e:
            logger.error(f"Error loading stopwords: {e}")
            return set()

    async def initialize(self):
        """Initialize all components asynchronously."""
        if self._initialized:
            return
            
        async with self._initialization_lock:
            if self._initialized:
                return
                
            try:
                # Initialize text processor
                self.text_processor = Process(self.lang)
                await self.text_processor.initialize()

                # Load stopwords
                self._stopwords = await self._load_stopwords()
                
                # Initialize error handler
                self.error_handler = ErrorHandler()
                await self.error_handler.initialize()
                
                async with self.error_handler.error_context(
                    AnalysisErrorType.INITIALIZATION,
                    component="analyzer"
                ):
                    # Initialize language models
                    models_to_load = []
                    if self.lang in ['bilingual', 'fr']:
                        models_to_load.append(('fr', 'fr_core_news_md'))
                    if self.lang in ['bilingual', 'en']:
                        models_to_load.append(('en', 'en_core_web_md'))

                    # Load models in parallel
                    tasks = []
                    for lang, model_name in models_to_load:
                        task = asyncio.create_task(self._load_spacy_model(lang, model_name))
                        tasks.append(task)

                    models = await asyncio.gather(*tasks)

                    for lang, model in models:
                        self.nlp[lang] = model
                    
                    # Initialize language detector with loaded models
                    self.language_detector = LanguageDetector(
                        self.nlp.get('fr'),
                        self.nlp.get('en')
                    )
                    
                    # Initialize text processor with stopwords
                    self.processor = Process(self.lang)
                    await self.processor.initialize()

                    # Initialize word vectors manager
                    self.word_vectors = WordVectorManager(use_mini=True)
                    await self.word_vectors.initialize()

                    # Initialize matrix utils
                    self.matrix_utils = MatrixUtils()

                    # Initialize topic labeler with word vectors
                    domain_terms_dir = Path('scripts/topic_analysis/domain_terms')
                    self.topic_labeler = TopicLabeler(
                        vectorizer=None,
                        domain_terms_dir=domain_terms_dir,
                        lang=self.lang
                    )
                    await self.topic_labeler.initialize()

                    self._initialized = True
                    logger.info("Analysis components initialized")
                    
            except Exception as e:
                logger.error(f"Error initializing components: {e}")
                await self.cleanup()
                raise

    async def _load_spacy_model(self, lang: str, model_name: str) -> Tuple[str, Any]:
        """Load spaCy model asynchronously."""
        try:
            loop = asyncio.get_event_loop()
            model = await loop.run_in_executor(
                self._executor,
                spacy.load,
                model_name
            )
            logger.info(f"Loaded spaCy model {model_name} for {lang}")
            return lang, model
            
        except Exception as e:
            logger.error(f"Error loading spaCy model {model_name}: {e}")
            raise

    async def analyze_topics(
        self,
        texts: List[str],
        method: str = 'lda',
        num_topics: int = 20,
        coherence_threshold: float = 0.3,
        min_topic_size: int = 10,
        max_words: int = 20,
        model_params: Optional[Dict] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        language: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Run topic analysis with enhanced language handling.
        
        Args:
            texts: List of document texts
            method: Analysis method ('lda', 'nmf', or 'lsa')
            num_topics: Number of topics to extract
            coherence_threshold: Minimum coherence score
            min_topic_size: Minimum words per topic
            max_words: Maximum words per topic
            model_params: Additional model parameters
            progress_callback: Optional progress callback
            language: Optional language override
            
        Returns:
            List of topic dictionaries
        """
        if not self._initialized:
            await self.initialize()

        async with self._analysis_lock:
            try:
                start_time = datetime.now()
                
                async with self.error_handler.error_context(
                    AnalysisErrorType.ANALYSIS,
                    component="topic_analysis"
                ):
                    # Process documents
                    if progress_callback:
                        await progress_callback(0.1, "Processing documents...")
                        
                    processed_docs = []
                    total = len(texts)
                    batch_size = 50
                    
                    # Process in batches
                    for i in range(0, total, batch_size):
                        batch = texts[i:i + batch_size]
                        batch_processed = []
                        
                        for text in batch:
                            if not isinstance(text, str) or not text.strip():
                                continue
                                
                            # Detect language if not overridden
                            doc_language = language
                            if not doc_language:
                                lang_result = await self.language_detector.detect_language(text)
                                if not lang_result:
                                    continue
                                doc_language, meta = lang_result
                                
                            # Process text using text processor
                            result = await self.text_processor.process_single_doc(
                                text,
                                lang=doc_language
                            )
                            
                            if result.text:
                                # Prepare metadata dictionary
                                metadata = result.metadata or {}
                                if 'meta' in locals():
                                    metadata.update(meta)

                                # Create document dictionary
                                doc_dict = {
                                    'text': result.text,
                                    'language': doc_language,
                                    'tokens': result.tokens,
                                    'metadata': metadata
                                }
                                batch_processed.append(doc_dict)
                                
                        processed_docs.extend(batch_processed)
                        
                        if progress_callback:
                            progress = min(1.0, (i + len(batch)) / total)
                            await progress_callback(
                                0.1 + progress * 0.3,
                                f"Processed {len(processed_docs)}/{total} documents"
                            )

                    if not processed_docs:
                        raise ValueError("No valid documents after processing")

                    logger.info(
                        f"Processed {len(processed_docs)} documents. "
                        f"(Analysis stopwords: {len(self.text_processor._base_stopwords.get(self.lang, set()))})"
                    )

                    if progress_callback:
                        await progress_callback(0.4, "Vectorizing documents...")

                    # Configure vectorizer
                    vectorizer = TfidfVectorizer(
                        max_df=0.95,
                        min_df=2,
                        max_features=10000,
                        ngram_range=(1, 2)
                    )

                    # Vectorize documents
                    doc_term_matrix = await self.vectorize_documents(
                        [doc['text'] for doc in processed_docs],
                        vectorizer
                    )

                    if progress_callback:
                        await progress_callback(0.5, f"Training {method.upper()} model...")

                    # Configure model
                    model_config = self.MODEL_CONFIGS[method].copy()
                    if model_params:
                        model_config['params'].update(model_params)
                    model_config['params']['n_components'] = num_topics

                    # Train model and extract topics
                    topics = await self._train_model(
                        doc_term_matrix,
                        vectorizer,
                        model_config,
                        lambda p, m: progress_callback(0.5 + p * 0.3, m)
                        if progress_callback else None
                    )

                    if progress_callback:
                        await progress_callback(0.8, "Calculating coherence...")

                    # Calculate coherence scores
                    coherence_scores = await self.calculate_coherence(
                        [topic['words'] for topic in topics],
                        [doc['text'] for doc in processed_docs]
                    )

                    # Filter and enhance topics
                    valid_topics = []
                    for topic, score in zip(topics, coherence_scores):
                        if (score >= coherence_threshold and 
                            min_topic_size <= len(topic['words']) <= max_words):
                            topic['coherence_score'] = score
                            # Generate topic label
                            topic['label'] = await self.topic_labeler.generate_label(
                                topic['words']
                            )
                            valid_topics.append(topic)

                    if not valid_topics:
                        logger.warning(
                            f"No topics met criteria (threshold={coherence_threshold}, "
                            f"min_size={min_topic_size}, max_words={max_words})"
                        )
                        # Take top topics by coherence
                        topics_scores = list(zip(topics, coherence_scores))
                        topics_scores.sort(key=lambda x: x[1], reverse=True)
                        valid_topics = [
                            {**topic, 'coherence_score': score}
                            for topic, score in topics_scores[:num_topics]
                        ]
                        for topic in valid_topics:
                            topic['label'] = await self.topic_labeler.generate_label(
                                topic['words']
                            )

                    # Calculate topic similarities
                    similarity_matrix = await self.data_handler.calculate_topic_similarity_matrix(
                        valid_topics
                    )

                    # Add similarity information
                    for i, topic in enumerate(valid_topics):
                        topic['similar_topics'] = [
                            {
                                'id': j,
                                'label': valid_topics[j]['label'],
                                'similarity': float(similarity_matrix[i, j])
                            }
                            for j in range(len(valid_topics))
                            if i != j and similarity_matrix[i, j] > 0.3
                        ]

                    # Update performance metrics
                    processing_time = (datetime.now() - start_time).total_seconds()
                    self.performance_metrics['processing_time'].append(processing_time)
                    self.performance_metrics['document_count'].append(len(texts))
                    self.performance_metrics['coherence_scores'].extend(coherence_scores)

                    if progress_callback:
                        await progress_callback(1.0, "Analysis completed")

                    # Add metadata to topics
                    for topic in valid_topics:
                        topic['metadata'] = {
                            'processing_time': processing_time,
                            'document_count': len(texts),
                            'language_distribution': {
                                lang: sum(1 for doc in processed_docs if doc['language'] == lang)
                                for lang in set(doc['language'] for doc in processed_docs)
                            }
                        }

                    return valid_topics

            except Exception as e:
                logger.error(f"Error in topic analysis: {str(e)}")
                raise RuntimeError(f"Topic analysis failed: {str(e)}") from e

    async def _process_documents(
        self,
        texts: List[str],
        progress_callback: Optional[Callable] = None,
        analysis_specific_stopwords: Optional[List[str]] = None,
        language: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Process documents with enhanced language detection and cleaning."""
        try:
            processed_docs = []
            total = len(texts)

            for i, text in enumerate(texts):
                if not isinstance(text, str) or not text.strip():
                    continue

                # Detect language
                doc_language = language
                meta = {}
                if not doc_language:
                    lang_result = await self.language_detector.detect_language(text)
                    if not lang_result:
                        continue
                    doc_language, meta = lang_result

                # Process text sith detected language and stopwords
                result = await self.processor.process_single_doc(
                    text,
                    lang=doc_language,
                    stopwords=analysis_specific_stopwords
                )

                if result.text:
                    processed_docs.append({
                        'text': result.text,
                        'language': doc_language,
                        'tokens': result.tokens,
                        'metadata': {
                            **meta,
                            **result.metadata,
                            'analysis_stopwords_used': len(analysis_specific_stopwords or set())
                        }
                    })

                if progress_callback:
                    await progress_callback(
                        (i + 1) / total,
                        f"Processing {i + 1}/{total} documents. Language: {doc_language}"
                    )
            
            logger.info(
                f"Processed {len(processed_docs)} documents. "
                f"(Analysis stopwords: {len(analysis_specific_stopwords or set())})"
            )
            return processed_docs

        except Exception as e:
            logger.error(f"Error processing documents: {e}")
            raise
    
    async def _run_in_executor(self, func: Callable, *args, **kwargs) -> Any:
        """Run a function in the thread pool executor."""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self._executor,
                lambda: func(*args, **kwargs)
            )
        except Exception as e:
            logger.error(f"Error running in executor: {e}")
            raise

    async def _vectorize_documents(
        self,
        texts: List[str],
        vectorizer: Any
    ) -> Any:
        """Vectorize documents and update topic labeler."""
        try:
            # Vectorize documents
            doc_term_matrix = await self._run_in_executor(
                vectorizer.fit_transform,
                texts
            )
            
            # Update topic labeler with fitted vectorizer
            if self.topic_labeler:
                self.topic_labeler.vectorizer = vectorizer
                
            return doc_term_matrix
            
        except Exception as e:
            logger.error(f"Error vectorizing documents: {e}")
            raise

    async def _train_model(
        self,
        doc_term_matrix: Any,
        vectorizer: TfidfVectorizer,
        model_config: Dict[str, Any],
        progress_callback: Optional[Callable] = None
    ) -> List[Dict[str, Any]]:
        """Train topic model asynchronously."""
        try:
            # Initialize model
            model_class = model_config['class']
            model = model_class(**model_config['params'])
            
            # Train model
            loop = asyncio.get_event_loop()
            doc_topics = await loop.run_in_executor(
                self._executor,
                model.fit_transform,
                doc_term_matrix
            )
            
            if progress_callback:
                await progress_callback(0.5, "Extracting topics...")
            
            # Extract topics
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            
            for topic_idx, topic_vec in enumerate(model.components_):
                # Get top words
                top_indices = topic_vec.argsort()[-model_config['params']['n_components']:][::-1]
                words = [feature_names[i] for i in top_indices]
                
                topics.append({
                    'id': topic_idx,
                    'words': words,
                    'word_scores': topic_vec[top_indices].tolist()
                })
            
            return topics
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise

    async def _calculate_vector_coherence(
        self,
        words: List[str],
        word_vectors: Any
    ) -> float:
        """Calculate coherence using word vectors."""
        try:
            similarities = []
            for i, word1 in enumerate(words[:-1]):
                for word2 in words[i+1:]:
                    sim = await self.word_vectors.calculate_similarity(
                        word1,
                        word2,
                        self.lang
                    )
                    similarities.append(sim)
            
            return float(np.mean(similarities)) if similarities else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating vector coherence: {e}")
            return 0.0

    async def _calculate_similarities(
        self,
        topics: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Calculate topic similarity matrix using MatrixUtils."""
        if not self.matrix_utils:
            raise RuntimeError("Matrix utils not initialized")
            
        try:
            # Extract topic words for comparison
            topic_words = [topic['words'] for topic in topics]
            feature_names = list(set(word for words in topic_words for word in words))
            
            # Calculate similarity matrix
            similarity_matrix = self.matrix_utils.calculate_similarity_matrix(
                topic_words,
                feature_names
            )
            
            return similarity_matrix
            
        except Exception as e:
            logger.error(f"Error calculating similarities: {e}")
            raise

    async def _calculate_vector_similarity(
        self,
        words1: List[str],
        words2: List[str]
    ) -> float:
        """Calculate similarity between word sets using word vectors."""
        try:
            # Get word vectors
            vectors1 = []
            vectors2 = []
            
            for word in words1:
                vector = await self.word_vectors.get_vector(word, self.lang)
                if vector is not None:
                    vectors1.append(vector)
            
            for word in words2:
                vector = await self.word_vectors.get_vector(word, self.lang)
                if vector is not None:
                    vectors2.append(vector)
            
            if not vectors1 or not vectors2:
                return 0.0
            
            # Calculate centroid vectors
            centroid1 = np.mean(vectors1, axis=0)
            centroid2 = np.mean(vectors2, axis=0)
            
            # Calculate cosine similarity
            similarity = np.dot(centroid1, centroid2) / (
                np.linalg.norm(centroid1) * np.linalg.norm(centroid2)
            )
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating vector similarity: {e}")
            return 0.0

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get analyzer performance metrics."""
        try:
            metrics = {
                'average_processing_time': np.mean(self.performance_metrics['processing_time']),
                'total_documents_processed': sum(self.performance_metrics['doc_count']),
                'average_topics_per_analysis': np.mean(self.performance_metrics['topic_count']),
                'average_coherence': np.mean(self.performance_metrics['coherence_scores']),
                'coherence_std': np.std(self.performance_metrics['coherence_scores'])
            }
            
            # Calculate processing speed
            if self.performance_metrics['processing_time']:
                total_docs = sum(self.performance_metrics['doc_count'])
                total_time = sum(self.performance_metrics['processing_time'])
                metrics['docs_per_second'] = total_docs / total_time
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}

    async def reset_performance_metrics(self):
        """Reset performance tracking."""
        self.performance_metrics.clear()

    async def cleanup(self):
        """Clean up all resources."""
        try:
            # Clean up executor
            if hasattr(self, '_executor'):
                self._executor.shutdown(wait=False)
            
            # Clean up components
            cleanup_tasks = []
            
            if self.text_processor:
                cleanup_tasks.append(self.text_processor.cleanup())
                
            if self.word_vectors:
                cleanup_tasks.append(self.word_vectors.cleanup())
                
            if self.topic_labeler:
                cleanup_tasks.append(self.topic_labeler.cleanup())
                
            if self.error_handler:
                cleanup_tasks.append(self.error_handler.cleanup())
            
            # Clean up matrix utils
            if self.matrix_utils:
                self.matrix_utils.cleanup()
            
            if cleanup_tasks:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            
            # Clear stopwords
            self._stopwords.clear()
            
            self._initialized = False
            logger.info("Topic analyzer cleaned up")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            raise

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()