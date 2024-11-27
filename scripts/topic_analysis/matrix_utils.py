# scripts/topic_analysis/matrix_utils.py

"""Optimized matrix operations for topic analysis."""
import numpy as np
from collections import Counter
from typing import List, Dict, Optional, Tuple
from loguru import logger
from scipy.sparse import csr_matrix, vstack
import gc
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

class MatrixUtils:
    """Handles matrix operations with improved memory efficiency."""
    
    def __init__(self):
        """Initialize with thread pool for parallel operations."""
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, multiprocessing.cpu_count() - 1)
        )

    def _calculate_pairwise_score(
        self,
        word1_id: int,
        word2_id: int,
        doc_term_matrix: csr_matrix,
        doc_freqs: np.ndarray,
        epsilon: float = 1e-12,
        use_npmi: bool = True
    ) -> Optional[float]:
        """Calculate pairwise coherence score using NPMI."""
        try:
            # Verify indices are within bounds
            matrix_width = doc_term_matrix.shape[1]
            if (word1_id >= matrix_width or word2_id >= matrix_width or
                word1_id >= len(doc_freqs) or word2_id >= len(doc_freqs)):
                return 0.0

            # Get document frequencies efficiently using sparse operations
            docs_with_word1 = doc_term_matrix.getcol(word1_id).toarray().flatten() > 0
            docs_with_word2 = doc_term_matrix.getcol(word2_id).toarray().flatten() > 0
            
            # Calculate co-occurrence and individual frequencies
            n_docs = float(doc_term_matrix.shape[0])
            doc_with_both = float(np.sum(docs_with_word1 & docs_with_word2))
            freq_1 = doc_freqs[word1_id]
            freq_2 = doc_freqs[word2_id]
            
            # Skip if no co-occurrences
            if doc_with_both == 0:
                return 0.0
                
            # Calculate probabilities with smoothing
            p_both = (doc_with_both + epsilon) / n_docs
            p_1 = (freq_1 + epsilon) / n_docs
            p_2 = (freq_2 + epsilon) / n_docs
            
            if use_npmi:
                # Normalized Pointwise Mutual Information
                pmi = np.log(p_both / (p_1 * p_2))
                npmi = pmi / -np.log(p_both + epsilon)
                return max(0.0, min(1.0, npmi))
            else:
                # Regular PMI
                return max(0.0, min(1.0, np.log(p_both / (p_1 * p_2))))
            
        except Exception as e:
            logger.error(f"Error calculating pairwise score: {e}")
            return 0.0

    def calculate_topic_coherence(
        self,
        topic_words: List[str],
        word_to_id: Dict[str, int],
        doc_term_matrix: csr_matrix,
        doc_freqs: np.ndarray,
        epsilon: float = 1e-12,
        batch_size: int = 50,
        use_npmi: bool = True
    ) -> float:
        """Calculate coherence score with improved batching."""
        try:
            # Filter out words not in vocabulary
            word_ids = []
            for word in topic_words:
                word_id = word_to_id.get(word)
                if word_id is not None and word_id < doc_term_matrix.shape[1]:
                    word_ids.append(word_id)
                    
            if not word_ids:
                return 0.0
            
            # Create word pairs for scoring
            pairs = [
                (word_ids[i], word_ids[j])
                for i in range(len(word_ids))
                for j in range(i + 1, len(word_ids))
            ]
            
            if not pairs:
                return 0.0
                
            # Process pairs in batches
            scores = []
            for i in range(0, len(pairs), batch_size):
                batch = pairs[i:i + batch_size]
                batch_scores = []
                
                # Calculate scores for batch in parallel
                futures = []
                for w1, w2 in batch:
                    future = self._executor.submit(
                        self._calculate_pairwise_score,
                        w1, w2, doc_term_matrix, doc_freqs,
                        epsilon, use_npmi
                    )
                    futures.append(future)
                
                # Collect results
                for future in futures:
                    try:
                        score = future.result()
                        if score is not None:
                            batch_scores.append(score)
                    except Exception as e:
                        logger.error(f"Error in coherence calculation: {e}")
                
                scores.extend(batch_scores)
                
                # Force garbage collection between batches
                gc.collect()
                    
            # Calculate final coherence score
            if not scores:
                return 0.0
                
            # Use geometric mean for better scaling
            scores = np.array(scores)
            scores = scores[scores > 0]  # Remove zeros before geometric mean
            if len(scores) == 0:
                return 0.0
                
            return float(np.exp(np.mean(np.log(scores + epsilon))))
            
        except Exception as e:
            logger.error(f"Error calculating topic coherence: {e}")
            return 0.0

    def create_doc_term_matrix(
        self,
        docs: List[List[str]],
        word_to_id: Dict[str, int]
    ) -> csr_matrix:
        """Create document-term matrix with batching."""
        try:
            if not word_to_id:
                return csr_matrix((0, 0))
            
            vocab_size = max(word_to_id.values()) + 1
            batch_size = 1000  # Process 1000 docs at a time
            
            # Process documents in batches
            matrices = []
            for i in range(0, len(docs), batch_size):
                batch_docs = docs[i:i + batch_size]
                rows = []
                cols = []
                data = []
                
                # Process each document in batch
                for doc_id, doc in enumerate(batch_docs):
                    # Count word frequencies
                    word_counts = Counter(
                        word_to_id[word] 
                        for word in doc 
                        if word in word_to_id
                    )
                    
                    # Add to sparse matrix data
                    for word_id, count in word_counts.items():
                        if word_id < vocab_size:
                            rows.append(doc_id)
                            cols.append(word_id)
                            data.append(count)
                
                # Create batch matrix
                batch_matrix = csr_matrix(
                    (data, (rows, cols)),
                    shape=(len(batch_docs), vocab_size),
                    dtype=np.float32
                )
                matrices.append(batch_matrix)
                
                # Clean up
                del rows, cols, data
                gc.collect()
            
            # Combine all batches
            if not matrices:
                return csr_matrix((0, vocab_size), dtype=np.float32)
                
            return vstack(matrices, format='csr')
            
        except Exception as e:
            logger.error(f"Error creating doc-term matrix: {e}")
            return csr_matrix((0, 0))

    def calculate_similarity_matrix(
        self,
        topics: List[List[str]],
        feature_names: List[str],
        batch_size: int = 100
    ) -> np.ndarray:
        """Calculate similarity matrix between topics with batching."""
        try:
            num_topics = len(topics)
            sim_matrix = np.zeros((num_topics, num_topics), dtype=np.float32)
            
            # Create word sets for each topic
            topic_sets = [set(topic) for topic in topics]
            
            # Process topic pairs in batches
            for i in range(0, num_topics, batch_size):
                for j in range(i, num_topics, batch_size):
                    # Process batch of pairs
                    batch_i = range(i, min(i + batch_size, num_topics))
                    batch_j = range(j, min(j + batch_size, num_topics))
                    
                    for ii in batch_i:
                        for jj in batch_j:
                            if ii == jj:
                                sim_matrix[ii, jj] = 1.0
                            else:
                                # Calculate Jaccard similarity
                                intersection = len(topic_sets[ii] & topic_sets[jj])
                                union = len(topic_sets[ii] | topic_sets[jj])
                                similarity = intersection / max(union, 1)
                                sim_matrix[ii, jj] = similarity
                                sim_matrix[jj, ii] = similarity
                    
                    # Clean up after each batch
                    gc.collect()
                    
            return sim_matrix
            
        except Exception as e:
            logger.error(f"Error calculating similarity matrix: {e}")
            return np.eye(len(topics), dtype=np.float32)

    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)