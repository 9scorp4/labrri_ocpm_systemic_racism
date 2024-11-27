"""Topic labeling functionality with async support."""
import os
from loguru import logger
from typing import List, Dict, Optional, Set, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import spacy
from collections import Counter
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
import aiofiles

class TopicLabeler:
    """Handles topic labeling with domain awareness and async support."""
    
    def __init__(self, vectorizer: Any, domain_terms_dir: List[str], lang: str):
        """Initialize topic labeler.
        
        Args:
            vectorizer: Fitted TF-IDF vectorizer
            domain_terms: List of domain-specific terms
            lang: Language ('en', 'fr', or 'bilingual')
        """
        self.vectorizer = vectorizer
        self.lang = lang
        self.domain_terms_dir = domain_terms_dir or Path('scripts/topic_analysis/domain_terms')
        self._executor = ThreadPoolExecutor()
        self._initialization_lock = asyncio.Lock()
        self._initialized = False
        self._cache = {}
        self._cache_lock = asyncio.Lock()
        
        # Initialize domain terms storage
        self.domain_terms: Dict[str, Set[str]] = {
            'en': set(),
            'fr': set()
        }
        self.nlp = None

    async def _load_domain_terms(self) -> Dict[str, Set[str]]:
        """Load language-specific domain terms.
        
        Returns:
            Dictionary mapping language codes to sets of domain terms
        """
        try:
            terms = {'en': set(), 'fr': set()}
            
            # Ensure directory exists
            if not self.domain_terms_dir.exists():
                logger.warning(f"Domain terms directory not found at {self.domain_terms_dir}")
                return terms

            # Load terms for each language
            for lang in ['en', 'fr']:
                if self.lang != 'bilingual' and self.lang != lang:
                    continue
                    
                file_path = self.domain_terms_dir / f"domain_terms_{lang}.txt"
                if not file_path.exists():
                    logger.warning(f"Domain terms file not found: {file_path}")
                    continue

                try:
                    async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                        content = await f.read()
                        # Process lines, skip comments and empty lines
                        terms[lang] = {
                            line.strip().lower()
                            for line in content.splitlines()
                            if line.strip() and not line.startswith('#')
                        }
                        logger.info(f"Loaded {len(terms[lang])} {lang} domain terms from {file_path}")
                except Exception as e:
                    logger.error(f"Error loading domain terms for {lang}: {e}")

            # Validate loaded terms
            total_terms = sum(len(terms[lang]) for lang in terms)
            if total_terms == 0:
                logger.warning("No domain terms loaded")
            else:
                logger.info(
                    f"Loaded domain terms - "
                    f"EN: {len(terms['en'])}, FR: {len(terms['fr'])}"
                )

            return terms

        except Exception as e:
            logger.error(f"Error loading domain terms: {e}")
            return {'en': set(), 'fr': set()}

    async def initialize(self):
        """Initialize components asynchronously."""
        if self._initialized:
            return
            
        async with self._initialization_lock:
            if self._initialized:
                return
                
            try:
                # Load domain terms
                self.domain_terms = await self._load_domain_terms()
                
                # Initialize NLP components
                await self._initialize_nlp()
                
                self._initialized = True
                logger.info(f"Topic labeler initialized with domain terms")
                
            except Exception as e:
                logger.error(f"Error initializing topic labeler: {e}")
                raise

    async def _initialize_nlp(self):
        """Initialize NLP components asynchronously."""
        try:
            if self.lang == 'bilingual':
                self.nlp = {
                    'fr': await self._load_spacy_model('fr_core_news_md'),
                    'en': await self._load_spacy_model('en_core_web_md')
                }
            else:
                model = 'fr_core_news_md' if self.lang == 'fr' else 'en_core_web_md'
                self.nlp = await self._load_spacy_model(model)
                
            logger.info(f"Initialized NLP models for {self.lang}")
            
        except Exception as e:
            logger.error(f"Error initializing NLP components: {e}")
            raise

    async def _load_spacy_model(self, model_name: str) -> Any:
        """Load spaCy model asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, spacy.load, model_name)

    async def generate_label(self, topic_words: List[str]) -> str:
        """Generate descriptive label for topic using domain terms."""
        if not self._initialized:
            await self.initialize()

        try:
            # Detect language
            detected_lang = await self._detect_language(' '.join(topic_words))
            
            # Get domain-specific terms for detected language
            domain_terms = self.domain_terms.get(detected_lang, set())
            
            # Find domain terms in topic words
            topic_domain_terms = [
                word for word in topic_words 
                if word.lower() in domain_terms
            ]
            
            if topic_domain_terms:
                # Use domain terms in label if found
                primary_terms = topic_domain_terms[:2]
                other_terms = [w for w in topic_words[:3] if w not in primary_terms]
                
                label_parts = []
                if primary_terms:
                    label_parts.append(' '.join(primary_terms).title())
                if other_terms:
                    label_parts.append(f"({', '.join(other_terms)})")
                
                # Add language tag for bilingual
                if self.lang == 'bilingual':
                    label_parts.append(f"[{detected_lang.upper()}]")
                    
                return ' '.join(label_parts)
            else:
                # Fallback labeling without domain terms
                return await self._create_fallback_label(topic_words, detected_lang)
                
        except Exception as e:
            logger.error(f"Error generating label: {e}")
            return self._create_generic_label(topic_words)

    def _create_generic_label(self, words: List[str]) -> str:
        """Create a generic label when all else fails."""
        return f"Topic: {', '.join(words[:3])}"

    async def _extract_key_terms(self, words: List[str], detected_lang: str) -> List[str]:
        """Extract key terms from topic words asynchronously."""
        try:
            # Process with appropriate NLP model
            if isinstance(self.nlp, dict):
                nlp = self.nlp.get(detected_lang, self.nlp['en'])
            else:
                nlp = self.nlp
                
            # Process text using executor
            text = ' '.join(words)
            doc = await asyncio.get_event_loop().run_in_executor(
                self._executor,
                nlp,
                text
            )
            
            # Collect terms
            terms = []
            seen = set()
            
            # Add noun chunks
            for chunk in doc.noun_chunks:
                if chunk.text.lower() not in seen:
                    terms.append(chunk.text.lower())
                    seen.add(chunk.text.lower())
            
            # Add named entities
            for ent in doc.ents:
                if ent.text.lower() not in seen:
                    terms.append(ent.text.lower())
                    seen.add(ent.text.lower())
            
            # Add remaining important words
            for token in doc:
                if (token.pos_ in {'NOUN', 'PROPN'} and 
                    token.text.lower() not in seen):
                    terms.append(token.text.lower())
                    seen.add(token.text.lower())
            
            return terms

        except Exception as e:
            logger.error(f"Error extracting key terms: {e}")
            return words[:3]

    async def _get_domain_terms(self, terms: List[str], lang: str) -> List[str]:
        """Identify domain-specific terms asynchronously."""
        try:
            domain_terms = self.domain_terms.get(lang, set())
            return [term for term in terms 
                   if any(domain_term in term 
                         for domain_term in domain_terms)]
        except Exception as e:
            logger.error(f"Error getting domain terms: {e}")
            return []

    async def _detect_language(self, text: str) -> str:
        """Detect language of text asynchronously."""
        try:
            if not isinstance(self.nlp, dict):
                return self.lang
                
            # Process with both models using executor
            loop = asyncio.get_event_loop()
            doc_fr = await loop.run_in_executor(
                self._executor,
                self.nlp['fr'],
                text
            )
            doc_en = await loop.run_in_executor(
                self._executor,
                self.nlp['en'],
                text
            )
            
            # Count tokens recognized by each model
            fr_count = sum(1 for token in doc_fr if token.is_alpha)
            en_count = sum(1 for token in doc_en if token.is_alpha)
            
            return 'fr' if fr_count >= en_count else 'en'
            
        except Exception as e:
            logger.error(f"Error detecting language: {e}")
            return self.lang

    async def _create_fallback_label(self, words: List[str], lang: str) -> str:
        """Create fallback label when normal labeling fails."""
        try:
            prefix = "Sujet:" if lang == 'fr' else "Topic:"
            return f"{prefix} {', '.join(words[:3])}"
        except Exception as e:
            logger.error(f"Error creating fallback label: {e}")
            return "Unlabeled Topic"

    async def _cache_label(self, key: str, label: str):
        """Cache generated label."""
        async with self._cache_lock:
            self._cache[key] = label

    async def calculate_label_quality(
        self,
        label: str,
        topic_words: List[str]
    ) -> float:
        """Calculate quality score for generated label asynchronously."""
        try:
            quality_scores = []
            
            # Detect language
            detected_lang = await self._detect_language(label)
            
            # Check domain term coverage
            label_terms = set(label.lower().split())
            domain_terms = self.domain_terms.get(detected_lang, set())
            domain_coverage = len(label_terms & domain_terms) / max(1, len(domain_terms))
            quality_scores.append(domain_coverage)
            
            # Check topic word representation
            topic_word_set = set(word.lower() for word in topic_words)
            topic_coverage = len(label_terms & topic_word_set) / len(topic_word_set)
            quality_scores.append(topic_coverage)
            
            # Calculate coherence
            if isinstance(self.nlp, dict):
                doc = await asyncio.get_event_loop().run_in_executor(
                    self._executor,
                    self.nlp[detected_lang],
                    label
                )
            else:
                doc = await asyncio.get_event_loop().run_in_executor(
                    self._executor,
                    self.nlp,
                    label
                )
                
            coherence = len(list(doc.noun_chunks)) / max(1, len(label.split()))
            quality_scores.append(coherence)
            
            # Return weighted average
            weights = [0.4, 0.4, 0.2]  # Prioritize domain and topic coverage
            return sum(score * weight for score, weight in zip(quality_scores, weights))
            
        except Exception as e:
            logger.error(f"Error calculating label quality: {e}")
            return 0.0

    async def cleanup(self):
        """Clean up resources."""
        try:
            self._executor.shutdown(wait=False)
            self._cache.clear()
            self._initialized = False
            logger.info("Topic labeler resources cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()