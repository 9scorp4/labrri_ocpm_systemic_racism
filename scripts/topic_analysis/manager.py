"""Enhanced topic analysis manager with improved language detection and async support."""
from datetime import datetime
import pandas as pd
import asyncio
from typing import List, Optional, Dict, Any, Callable, Tuple
from loguru import logger
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import spacy
from pathlib import Path
import numpy as np

from scripts.language_detector import LanguageDetector
from scripts.topic_analysis.analyzer import TopicAnalyzer
from scripts.topic_analysis.topic_handler import TopicHandler
from scripts.topic_analysis.data_handler import TopicAnalysisDataHandler
from scripts.topic_analysis.async_db_helper import AsyncDatabaseHelper
from scripts.topic_analysis.error_handlers import ErrorHandler, AnalysisErrorType

class TopicAnalysisManager:
    """Enhanced manager with improved language handling and async coordination."""
    
    def __init__(self, db_path: str, lang: str = 'bilingual'):
        """Initialize manager with specified language mode."""
        if not db_path:
            raise ValueError("Database path must be specified")
            
        self.db_path = db_path
        self.lang = lang
        
        # Initialize state management
        self._initialization_lock = asyncio.Lock()
        self._analysis_lock = asyncio.Lock()
        self._initialized = False
        self._executor = ThreadPoolExecutor(max_workers=multiprocessing.cpu_count())
        
        # Initialize component placeholders
        self.db = None
        self.nlp = {}
        self.analyzer = None
        self.topic_handler = None
        self.data_handler = None
        self.error_handler = None
        self.language_detector = None
        
        # Track analysis statistics
        self.analysis_stats = {
            'processed_docs': 0,
            'language_distribution': {'en': 0, 'fr': 0, 'unknown': 0},
            'coherence_scores': [],
            'processing_times': []
        }
        
        logger.info(f"Topic analysis manager initialized for {lang}")

    async def initialize(self):
        """Initialize all components asynchronously."""
        if self._initialized:
            return
            
        async with self._initialization_lock:
            if self._initialized:
                return
                
            try:
                # Initialize error handler first
                self.error_handler = ErrorHandler()
                await self.error_handler.initialize()
                
                async with self.error_handler.error_context(
                    AnalysisErrorType.INITIALIZATION,
                    component="manager"
                ):
                    # Initialize database helper
                    self.db = AsyncDatabaseHelper(self.db_path)
                    
                    # Initialize language models concurrently
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
                    
                    # Initialize language detector
                    self.language_detector = LanguageDetector(
                        self.nlp.get('fr'),
                        self.nlp.get('en')
                    )
                    
                    # Initialize analysis components
                    self.analyzer = TopicAnalyzer(self.db_path, lang=self.lang)
                    await self.analyzer.initialize()
                    
                    # Initialize handlers
                    self.topic_handler = TopicHandler(self.db)
                    self.data_handler = TopicAnalysisDataHandler()
                    
                    self._initialized = True
                    logger.info("Analysis pipeline initialized successfully")
                    
            except Exception as e:
                logger.error(f"Error initializing components: {e}")
                await self.cleanup()
                raise RuntimeError(f"Pipeline initialization failed: {str(e)}")

    async def analyze_topics(
        self,
        texts: List[str],
        method: str = 'lda',
        num_topics: int = 20,
        coherence_threshold: float = 0.3,
        min_topic_size: int = 10,
        max_words: int = 20,
        model_params: Optional[Dict] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
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
                    if progress_callback:
                        await progress_callback(0.1, "Detecting languages...")
                    
                    # Process documents by language
                    processed_docs = await self._process_documents_by_language(
                        texts,
                        lambda p, m: progress_callback(0.1 + p * 0.3, m)
                        if progress_callback else None
                    )
                    
                    if not processed_docs:
                        raise ValueError("No valid documents for analysis")
                        
                    # Update language statistics
                    self.analysis_stats['processed_docs'] += len(processed_docs)
                    for doc in processed_docs:
                        self.analysis_stats['language_distribution'][doc['language']] += 1
                    
                    if progress_callback:
                        await progress_callback(0.4, "Analyzing topics...")
                    
                    # Separate documents by language
                    docs_by_lang = {
                        'en': [d['text'] for d in processed_docs if d['language'] == 'en'],
                        'fr': [d['text'] for d in processed_docs if d['language'] == 'fr']
                    }
                    
                    # Analyze each language separately
                    all_topics = []
                    for lang, docs in docs_by_lang.items():
                        if not docs:
                            continue
                            
                        # Run analysis
                        topics = await self.analyzer.analyze_topics(
                            docs,
                            method=method,
                            num_topics=num_topics,
                            coherence_threshold=coherence_threshold,
                            min_topic_size=min_topic_size,
                            max_words=max_words,
                            model_params=model_params,
                            language=lang
                        )
                        
                        # Add language tag and enhance topic information
                        enhanced_topics = await self._enhance_topics(
                            topics,
                            lang,
                            docs
                        )
                        
                        all_topics.extend(enhanced_topics)
                    
                    if not all_topics:
                        raise ValueError("No topics found in analysis")
                    
                    # Sort by coherence score
                    all_topics.sort(key=lambda x: x['coherence_score'], reverse=True)
                    
                    # Update analysis statistics
                    processing_time = (datetime.now() - start_time).total_seconds()
                    self.analysis_stats['processing_times'].append(processing_time)
                    self.analysis_stats['coherence_scores'].extend(
                        [t['coherence_score'] for t in all_topics]
                    )
                    
                    if progress_callback:
                        await progress_callback(1.0, "Analysis completed")
                    
                    return all_topics

            except Exception as e:
                logger.error(f"Error in topic analysis: {str(e)}")
                raise

    async def _process_documents_by_language(
        self,
        texts: List[str],
        progress_callback: Optional[Callable] = None
    ) -> List[Dict]:
        """Process documents with language detection."""
        try:
            processed_docs = []
            total = len(texts)
            
            for i, text in enumerate(texts):
                if not isinstance(text, str) or not text.strip():
                    continue
                    
                # Detect language
                lang_result = await self.language_detector.detect_language(text)
                if not lang_result:
                    continue
                    
                lang, meta = lang_result
                
                # Process text
                processed = await self.analyzer.process_text(text, lang)
                if not processed.get('text'):
                    continue
                
                processed_docs.append({
                    'text': processed['text'],
                    'language': lang,
                    'tokens': processed.get('tokens', []),
                    'metadata': {**meta, **processed.get('metadata', {})}
                })
                
                if progress_callback:
                    await progress_callback((i + 1) / total, f"Processed {i + 1}/{total} documents")
            
            return processed_docs
            
        except Exception as e:
            logger.error(f"Error processing documents: {e}")
            return []

    async def _enhance_topics(
        self,
        topics: List[Dict],
        language: str,
        documents: List[str]
    ) -> List[Dict]:
        """Enhance topics with additional metrics and information."""
        try:
            enhanced_topics = []
            
            # Calculate additional metrics
            coherence_scores, coherence_meta = await self.data_handler.calculate_topic_coherence(
                [t['words'] for t in topics],
                documents,
                method='c_npmi'
            )
            
            similarity_matrix = await self.data_handler.calculate_topic_similarity_matrix(topics)
            
            for i, topic in enumerate(topics):
                # Calculate topic metrics
                enhanced_topic = {
                    **topic,
                    'language': language,
                    'coherence_score': coherence_scores[i],
                    'coherence_metrics': {
                        k: v[i] for k, v in coherence_meta.items()
                        if isinstance(v, list)
                    },
                    'similar_topics': self._get_similar_topics(
                        i,
                        similarity_matrix,
                        topics,
                        threshold=0.3
                    )
                }
                
                enhanced_topics.append(enhanced_topic)
            
            return enhanced_topics
            
        except Exception as e:
            logger.error(f"Error enhancing topics: {e}")
            return topics

    def _get_similar_topics(
        self,
        topic_idx: int,
        similarity_matrix: np.ndarray,
        topics: List[Dict],
        threshold: float = 0.3
    ) -> List[Dict]:
        """Get similar topics based on similarity matrix."""
        try:
            similarities = similarity_matrix[topic_idx]
            similar_topics = []
            
            for idx, sim in enumerate(similarities):
                if idx != topic_idx and sim >= threshold:
                    similar_topics.append({
                        'id': idx,
                        'label': topics[idx]['label'],
                        'similarity': float(sim)
                    })
            
            # Sort by similarity
            similar_topics.sort(key=lambda x: x['similarity'], reverse=True)
            return similar_topics
            
        except Exception as e:
            logger.error(f"Error getting similar topics: {e}")
            return []

    async def _load_spacy_model(
        self,
        lang: str,
        model_name: str
    ) -> Tuple[str, Any]:
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

    async def cleanup(self):
        """Clean up all resources."""
        try:
            # Clean up executor
            if hasattr(self, '_executor'):
                self._executor.shutdown(wait=False)
            
            # Clean up components
            cleanup_tasks = []
            
            if self.analyzer:
                cleanup_tasks.append(self.analyzer.cleanup())
                
            if self.db:
                cleanup_tasks.append(self.db.cleanup())
                
            if self.error_handler:
                cleanup_tasks.append(self.error_handler.cleanup())
            
            if cleanup_tasks:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            
            self._initialized = False
            logger.info("Analysis manager cleaned up")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            raise

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()