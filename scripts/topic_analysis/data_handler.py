from loguru import logger
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import scipy.sparse as sp
from gensim.matutils import Sparse2Corpus
from gensim.models import CoherenceModel
import warnings

class TopicAnalysisDataHandler:
    """Enhanced data handler with improved coherence calculations and caching."""
    
    def __init__(self, cache_ttl: int = 900):  # 15 minutes default TTL
        """Initialize data handler with cache.
        
        Args:
            cache_ttl: Cache time-to-live in seconds
        """
        self._cache = {}
        self._cache_lock = asyncio.Lock()
        self._cache_ttl = timedelta(seconds=cache_ttl)
        self._last_cleanup = datetime.now()
        self._vectorizer = TfidfVectorizer(
            max_df=0.95,
            min_df=2,
            max_features=10000,
            ngram_range=(1, 2)
        )
        
        # Suppress UserWarnings from gensim
        warnings.filterwarnings("ignore", category=UserWarning)

    async def calculate_topic_coherence(
        self,
        topic_words: List[List[str]],
        documents: List[str],
        method: str = 'c_npmi',
        cache_key: Optional[str] = None
    ) -> Tuple[List[float], Dict[str, Any]]:
        """Calculate enhanced topic coherence with multiple metrics.
        
        Args:
            topic_words: List of word lists for each topic
            documents: Original documents for reference
            method: Coherence calculation method ('c_v', 'c_npmi', 'u_mass')
            cache_key: Optional cache key for results
            
        Returns:
            Tuple of (coherence scores, metadata)
        """
        try:
            # Check cache if key provided
            if cache_key:
                cached_result = await self._get_cached('coherence', cache_key)
                if cached_result is not None:
                    return cached_result

            # Create document term matrix
            dtm = self._vectorizer.fit_transform(documents)
            vocab = self._vectorizer.get_feature_names_out()
            word_to_id = {word: idx for idx, word in enumerate(vocab)}

            # Calculate multiple coherence metrics
            scores = []
            metadata = defaultdict(list)

            for topic_terms in topic_words:
                # Filter terms to those in vocabulary
                valid_terms = [term for term in topic_terms if term in word_to_id]
                if not valid_terms:
                    scores.append(0.0)
                    continue

                # Calculate NPMI coherence
                npmi_score = await self._calculate_npmi_coherence(
                    valid_terms,
                    dtm,
                    word_to_id
                )

                # Calculate PMI coherence
                pmi_score = await self._calculate_pmi_coherence(
                    valid_terms,
                    dtm,
                    word_to_id
                )

                # Calculate semantic coherence using word co-occurrence
                semantic_score = await self._calculate_semantic_coherence(
                    valid_terms,
                    dtm,
                    word_to_id
                )

                # Combine scores with weights
                combined_score = (
                    0.4 * npmi_score +
                    0.3 * pmi_score +
                    0.3 * semantic_score
                )

                scores.append(combined_score)
                
                # Store individual metrics in metadata
                metadata['npmi_scores'].append(npmi_score)
                metadata['pmi_scores'].append(pmi_score)
                metadata['semantic_scores'].append(semantic_score)
                metadata['term_coverage'].append(len(valid_terms) / len(topic_terms))

            # Add overall statistics to metadata
            metadata.update({
                'mean_coherence': np.mean(scores),
                'std_coherence': np.std(scores),
                'min_coherence': np.min(scores),
                'max_coherence': np.max(scores)
            })

            # Cache results if key provided
            if cache_key:
                await self._cache_result('coherence', cache_key, (scores, metadata))

            return scores, metadata

        except Exception as e:
            logger.error(f"Error calculating coherence: {e}")
            return [0.0] * len(topic_words), {'error': str(e)}

    async def calculate_topic_similarity_matrix(
        self,
        topics: List[Dict],
        use_cache: bool = True,
        min_similarity: float = 0.1
    ) -> np.ndarray:
        """Calculate similarity matrix between topics with improved metrics.
        
        Args:
            topics: List of topic dictionaries
            use_cache: Whether to use cache
            min_similarity: Minimum similarity threshold
            
        Returns:
            Similarity matrix
        """
        try:
            # Create cache key from topic hashes
            if use_cache:
                cache_key = '_'.join(str(hash(str(t))) for t in topics)
                cached_result = await self._get_cached('similarity', cache_key)
                if cached_result is not None:
                    return cached_result

            # Create word sets and term-frequency vectors
            topic_words = []
            topic_vectors = []
            
            for topic in topics:
                # Handle different topic formats
                if isinstance(topic, dict):
                    words = topic.get('words', [])
                    if isinstance(words, str):
                        words = words.split(',')
                else:
                    words = topic[1] if isinstance(topic, (tuple, list)) else []
                
                # Create word set
                word_set = set(words)
                topic_words.append(word_set)
                
                # Create term frequency vector
                vector = [words.count(word) for word in word_set]
                topic_vectors.append(vector)

            # Calculate similarity matrix
            n_topics = len(topics)
            similarity_matrix = np.zeros((n_topics, n_topics))
            
            for i in range(n_topics):
                for j in range(i, n_topics):
                    if i == j:
                        similarity_matrix[i, j] = 1.0
                        continue
                        
                    # Calculate Jaccard similarity
                    intersection = len(topic_words[i] & topic_words[j])
                    union = len(topic_words[i] | topic_words[j])
                    jaccard_sim = intersection / union if union > 0 else 0.0
                    
                    # Calculate cosine similarity of term frequencies
                    vec_i = topic_vectors[i]
                    vec_j = topic_vectors[j]
                    if vec_i and vec_j:  # Only if both have words
                        cosine_sim = np.dot(vec_i, vec_j) / (
                            np.linalg.norm(vec_i) * np.linalg.norm(vec_j)
                        )
                    else:
                        cosine_sim = 0.0
                    
                    # Combine similarities with weights
                    similarity = 0.7 * jaccard_sim + 0.3 * cosine_sim
                    
                    # Apply threshold
                    if similarity < min_similarity:
                        similarity = 0.0
                    
                    similarity_matrix[i, j] = similarity_matrix[j, i] = similarity

            # Cache result
            if use_cache:
                await self._cache_result('similarity', cache_key, similarity_matrix)

            return similarity_matrix

        except Exception as e:
            logger.error(f"Error calculating similarity matrix: {e}")
            return np.eye(len(topics))

    async def _calculate_npmi_coherence(
        self,
        terms: List[str],
        dtm: sp.spmatrix,
        word_to_id: Dict[str, int],
        window_size: int = 10
    ) -> float:
        """Calculate NPMI coherence score."""
        try:
            term_ids = [word_to_id[term] for term in terms]
            n_docs = dtm.shape[0]
            
            # Calculate document frequencies
            doc_freqs = np.array(dtm.sum(axis=0)).flatten()
            
            # Calculate pairwise NPMI
            npmi_scores = []
            for i, term1_id in enumerate(term_ids[:-1]):
                for term2_id in term_ids[i+1:]:
                    # Get document frequencies
                    docs_with_term1 = dtm.getcol(term1_id).toarray().flatten()
                    docs_with_term2 = dtm.getcol(term2_id).toarray().flatten()
                    
                    # Calculate co-occurrence
                    co_occur = np.sum(docs_with_term1 & docs_with_term2)
                    if co_occur == 0:
                        continue
                        
                    # Calculate probabilities
                    p1 = doc_freqs[term1_id] / n_docs
                    p2 = doc_freqs[term2_id] / n_docs
                    p12 = co_occur / n_docs
                    
                    # Calculate NPMI
                    pmi = np.log(p12 / (p1 * p2))
                    npmi = pmi / (-np.log(p12))
                    
                    npmi_scores.append(npmi)
            
            return float(np.mean(npmi_scores)) if npmi_scores else 0.0

        except Exception as e:
            logger.error(f"Error calculating NPMI coherence: {e}")
            return 0.0

    async def _calculate_pmi_coherence(
        self,
        terms: List[str],
        dtm: sp.spmatrix,
        word_to_id: Dict[str, int]
    ) -> float:
        """Calculate PMI coherence score."""
        try:
            term_ids = [word_to_id[term] for term in terms]
            n_docs = dtm.shape[0]
            
            # Calculate document frequencies
            doc_freqs = np.array(dtm.sum(axis=0)).flatten()
            
            # Calculate pairwise PMI
            pmi_scores = []
            for i, term1_id in enumerate(term_ids[:-1]):
                for term2_id in term_ids[i+1:]:
                    # Get document frequencies
                    docs_with_term1 = dtm.getcol(term1_id).toarray().flatten()
                    docs_with_term2 = dtm.getcol(term2_id).toarray().flatten()
                    
                    # Calculate co-occurrence
                    co_occur = np.sum(docs_with_term1 & docs_with_term2)
                    if co_occur == 0:
                        continue
                        
                    # Calculate probabilities
                    p1 = doc_freqs[term1_id] / n_docs
                    p2 = doc_freqs[term2_id] / n_docs
                    p12 = co_occur / n_docs
                    
                    # Calculate PMI
                    pmi = np.log(p12 / (p1 * p2))
                    pmi_scores.append(pmi)
            
            return float(np.mean(pmi_scores)) if pmi_scores else 0.0

        except Exception as e:
            logger.error(f"Error calculating PMI coherence: {e}")
            return 0.0

    async def _calculate_semantic_coherence(
        self,
        terms: List[str],
        dtm: sp.spmatrix,
        word_to_id: Dict[str, int]
    ) -> float:
        """Calculate semantic coherence using word co-occurrence patterns."""
        try:
            term_ids = [word_to_id[term] for term in terms]
            
            # Create co-occurrence matrix
            cooc_matrix = (dtm.T @ dtm).toarray()
            
            # Calculate pairwise semantic similarities
            similarities = []
            for i, term1_id in enumerate(term_ids[:-1]):
                for term2_id in term_ids[i+1:]:
                    vec1 = cooc_matrix[term1_id]
                    vec2 = cooc_matrix[term2_id]
                    
                    # Calculate cosine similarity
                    similarity = np.dot(vec1, vec2) / (
                        np.linalg.norm(vec1) * np.linalg.norm(vec2)
                    )
                    similarities.append(similarity)
            
            return float(np.mean(similarities)) if similarities else 0.0

        except Exception as e:
            logger.error(f"Error calculating semantic coherence: {e}")
            return 0.0

    async def _get_cached(self, category: str, key: str) -> Optional[Any]:
        """Get cached result if not expired."""
        async with self._cache_lock:
            cache_key = f"{category}_{key}"
            if cache_key in self._cache:
                timestamp, value = self._cache[cache_key]
                if datetime.now() - timestamp <= self._cache_ttl:
                    return value
                del self._cache[cache_key]
        return None

    async def _cache_result(self, category: str, key: str, value: Any):
        """Cache a result with timestamp."""
        async with self._cache_lock:
            self._cache[f"{category}_{key}"] = (datetime.now(), value)

    async def update_cache(self):
        """Clean expired cache entries."""
        try:
            current_time = datetime.now()
            
            # Only clean cache every minute
            if (current_time - self._last_cleanup) < timedelta(minutes=1):
                return
                
            async with self._cache_lock:
                # Remove expired entries
                expired_keys = [
                    key for key, (timestamp, _) in self._cache.items()
                    if (current_time - timestamp) > self._cache_ttl
                ]
                
                for key in expired_keys:
                    del self._cache[key]
                    
                self._last_cleanup = current_time
                
                if expired_keys:
                    logger.debug(f"Cleared {len(expired_keys)} expired cache entries")
                    
        except Exception as e:
            logger.error(f"Error updating cache: {e}")

    async def cleanup(self):
        """Clean up resources."""
        try:
            async with self._cache_lock:
                self._cache.clear()
            logger.info("Data handler cache cleared")
        except Exception as e:
            logger.error(f"Error cleaning up data handler: {e}")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()