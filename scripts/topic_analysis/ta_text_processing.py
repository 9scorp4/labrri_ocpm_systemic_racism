"""Text processing for topic analysis with async support and proper stopwords handling."""
from loguru import logger
import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from typing import List, Optional, Dict, Set, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import aiofiles
from dataclasses import dataclass, field
from collections import defaultdict
import functools
import os

@dataclass
class ProcessingResult:
    """Container for text processing results."""
    text: str
    tokens: List[str] = field(default_factory=list)
    language: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

class Process:
    """Handles text processing for topic analysis with async support."""
    
    def __init__(self, lang: str):
        """Initialize text processor.
        
        Args:
            lang: Language for processing ('en', 'fr', or 'bilingual')
        """
        logger.info('Initializing text processor')
        self.lang = lang
        self._executor = ThreadPoolExecutor()
        self._initialization_lock = asyncio.Lock()
        self._initialized = False
        self._cache = {}
        self._cache_lock = asyncio.Lock()
        
        # Initialize stopwords storage
        self._base_stopwords: Dict[str, Set[str]] = {
            'en': set(),
            'fr': set()
        }
        
        # Initialize components
        self.nlp = None

    async def initialize(self):
        """Initialize components asynchronously."""
        if self._initialized:
            return
            
        async with self._initialization_lock:
            if self._initialized:
                return
                
            try:
                # Initialize NLP components
                await self._initialize_nlp()
                
                # Load stopwords
                await self._initialize_nltk_stopwords()
                
                # Load stopwords from file and merge with NLTK stopwords
                await self._load_file_stopwords()
                
                self._initialized = True
                logger.info("Text processor initialized successfully")
                
            except Exception as e:
                logger.error(f"Error initializing text processor: {e}")
                raise

    async def _initialize_nlp(self):
        """Initialize NLP components asynchronously."""
        try:
            # Load language-specific models
            if self.lang == 'bilingual':
                self.nlp = {
                    'fr': await self._load_spacy_model('fr_core_news_md'),
                    'en': await self._load_spacy_model('en_core_web_md')
                }
            else:
                model = 'fr_core_news_md' if self.lang == 'fr' else 'en_core_web_md'
                self.nlp = await self._load_spacy_model(model)
                
            logger.info(f"NLP models loaded for {self.lang}")
            
        except Exception as e:
            logger.error(f"Error initializing NLP components: {e}")
            raise

    async def _load_spacy_model(self, model_name: str) -> Any:
        """Load spaCy model asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, spacy.load, model_name)

    async def _initialize_nltk_stopwords(self):
        """Initialize stopwords from NLTK asynchronously."""
        try:
            loop = asyncio.get_event_loop()

            # Create wrapped download function
            async def download_nltk_data(resource):
                return await loop.run_in_executor(
                    self._executor,
                    functools.partial(nltk.download, resource)
                )
            
            # Download required NLTK data
            await download_nltk_data('stopwords')
            await download_nltk_data('punkt')
            
            # Load language-specific stopwords
            if self.lang == 'bilingual':
                self._base_stopwords['fr'] = set(await loop.run_in_executor(
                    None, stopwords.words, 'french'
                ))
                self._base_stopwords['en'] = set(await loop.run_in_executor(
                    None, stopwords.words, 'english'
                ))
            else:
                lang_name = 'french' if self.lang == 'fr' else 'english'
                self._base_stopwords[self.lang] = set(await loop.run_in_executor(
                    None, stopwords.words, lang_name
                ))
            
            logger.info("Stopwords loaded successfully")
            
        except Exception as e:
            logger.error(f"Error initializing stopwords: {e}")
            raise

    async def _load_file_stopwords(self):
        """Load additional stopwords from file and merge with base stopwords."""
        try:
            stopwords_path = Path('scripts/topic_analysis/stopwords.txt')
            
            if not os.path.exists(stopwords_path):
                logger.warning(f"Stopwords file not found: {stopwords_path}")
                return
            
            async with aiofiles.open(stopwords_path, mode='r', encoding='utf-8') as f:
                content = await f.read()
                file_stopwords = {
                    line.strip().lower()
                    for line in content.splitlines()
                    if line.strip() and not line.startswith('#')
                }

                # Add file stopwords to all language sets
                for lang in self._base_stopwords:
                    self._base_stopwords[lang].update(file_stopwords)

                logger.info(f"Loaded {len(file_stopwords)} additional stopwords from file")

        except Exception as e:
            logger.error(f"Error loading custom stopwords: {e}")

    async def process_documents(
        self,
        docs: List[str],
        lang: Optional[str] = None,
        batch_size: int = 50
    ) -> List[ProcessingResult]:
        """Process multiple documents asynchronously."""
        try:
            await self.initialize()
            lang = lang or self.lang
            
            # Process documents in batches
            results = []
            for i in range(0, len(docs), batch_size):
                batch = docs[i:i + batch_size]
                batch_tasks = [
                    self.process_single_doc(doc, lang)
                    for doc in batch
                ]
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for result in batch_results:
                    if isinstance(result, Exception):
                        logger.error(f"Error processing document: {result}")
                        results.append(ProcessingResult(
                            text="",
                            error=str(result)
                        ))
                    else:
                        results.append(result)
                        
            return results
            
        except Exception as e:
            logger.error(f"Error processing documents: {e}")
            return []

    async def process_single_doc(
        self,
        text: str,
        lang: Optional[str] = None,
        stopwords: Optional[Set[str]] = None
    ) -> ProcessingResult:
        """Process a single document asynchronously."""
        try:
            await self.initialize()
            lang = lang or self.lang
            
            # Check cache
            cache_key = f"{lang}:{text[:100]}"
            if stopwords:
                cache_key += f":{hash(frozenset(stopwords))}"

            async with self._cache_lock:
                if cache_key in self._cache:
                    return self._cache[cache_key]
            
            # Get base stopwords for language
            base_stopwords = set()
            if isinstance(self._base_stopwords, dict):
                if lang in self._base_stopwords:
                    base_stopwords.update(self._base_stopwords[lang])
            else:
                base_stopwords.update(*self._base_stopwords.values())
            
            # Combine with additional stopwords if provided
            if stopwords:
                base_stopwords.update(stopwords)

            # Tokenize text
            tokens = await self._tokenize(text, lang)
            if not tokens:
                return ProcessingResult(text="", error="Tokenization failed")

            # Remove stopwords
            tokens = [
                token for token in tokens
                if token.lower() not in base_stopwords
            ]
            
            # Create result
            result = ProcessingResult(
                text=' '.join(tokens),
                tokens=tokens,
                language=lang,
                metadata={
                    'original_length': len(text),
                    'stopwords_removed': len(base_stopwords)
                }
            )
            
            # Cache result
            async with self._cache_lock:
                self._cache[cache_key] = result
            
            return result

        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return ProcessingResult(text="", error=str(e))

    async def _tokenize(self, text: str, lang: str) -> List[str]:
        """Tokenize text asynchronously."""
        try:
            if isinstance(self.nlp, dict):
                # Process with both models for bilingual
                fr_doc = await asyncio.get_event_loop().run_in_executor(
                    self._executor,
                    self.nlp['fr'],
                    text
                )
                en_doc = await asyncio.get_event_loop().run_in_executor(
                    self._executor,
                    self.nlp['en'],
                    text
                )
                return [token.text.lower() for doc in (fr_doc, en_doc) 
                        for token in doc if token.is_alpha]
            else:
                doc = await asyncio.get_event_loop().run_in_executor(
                    self._executor,
                    self.nlp,
                    text
                )
                return [token.text.lower() for token in doc if token.is_alpha]
                
        except Exception as e:
            logger.error(f"Error tokenizing text: {e}")
            return []

    async def update_stopwords(self, new_stopwords: Set[str], lang: Optional[str] = None):
        """Update stopwords for specified language."""
        try:
            if lang:
                if lang in self._stopwords:
                    self._stopwords[lang].update(new_stopwords)
            else:
                # Update all language stopwords
                for lang_stopwords in self._stopwords.values():
                    lang_stopwords.update(new_stopwords)
                    
            # Clear cache since stopwords changed
            async with self._cache_lock:
                self._cache.clear()
                
        except Exception as e:
            logger.error(f"Error updating stopwords: {e}")

    async def cleanup(self):
        """Clean up resources."""
        try:
            if hasattr(self, '_executor'):
                self._executor.shutdown(wait=False)
            
            if hasattr(self, '_cache'):
                async with self._cache_lock:
                    self._cache.clear()
            
            self._initialized = False
            logger.info("Text processor resources cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()