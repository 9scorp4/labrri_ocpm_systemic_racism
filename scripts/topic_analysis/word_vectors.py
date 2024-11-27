"""Word vector downloading with initialization and improved robustness."""
from pathlib import Path
import httpx
import asyncio
from typing import Optional, Dict, Any
from loguru import logger
import aiofiles
from tqdm.asyncio import tqdm
import backoff
from gensim.models import KeyedVectors
import numpy as np

class WordVectorManager:
    """Manages word vector models with improved download handling."""
    
    CHUNK_SIZE = 1024 * 1024  # 1MB chunks
    MAX_RETRIES = 10
    TIMEOUT = 300.0  # 5 minutes timeout
    
    VECTOR_CONFIGS = {
        'en': {
            'name': 'word2vec-google-news-300',
            'url': None,  # Uses gensim downloader
            'dim': 300,
            'size': None,
            'fallback_url': 'https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec',
            'md5': None
        },
        'fr': {
            'name': 'wiki.fr.vec',
            'url': 'https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.fr.vec',
            'dim': 300,
            'size': 8_649_004_489,
            'fallback_url': None,
            'md5': None
        }
    }

    def __init__(self, cache_dir: Optional[Path] = None, use_mini: bool = False):
        """Initialize word vector manager."""
        self.cache_dir = cache_dir or Path('data/word_vectors')
        self._use_mini = use_mini
        self._vectors: Dict[str, Any] = {}
        self._dummy_vectors: Dict[str, Any] = {}
        self._download_progress: Dict[str, float] = {}
        self._download_lock = asyncio.Lock()
        self._cache_lock = asyncio.Lock()
        self._initialization_lock = asyncio.Lock()
        self._initialized = False

        # Mini vector sizes for faster loading
        self._mini_dims = {
            'en': 100,
            'fr': 100
        }

    async def initialize(self):
        """Initialize the word vector manager."""
        if self._initialized:
            return

        async with self._initialization_lock:
            if self._initialized:
                return

            try:
                # Create necessary directories
                for subdir in ['downloads', 'temp', 'cache']:
                    (self.cache_dir / subdir).mkdir(parents=True, exist_ok=True)

                # Initialize vector storage
                self._vectors = {}
                self._dummy_vectors = {}
                self._download_progress = {}

                # Load any cached vectors
                await self._load_cached_vectors()

                self._initialized = True
                logger.info("Word vector manager initialized successfully")

            except Exception as e:
                logger.error(f"Error initializing word vector manager: {e}")
                raise

    async def _load_cached_vectors(self):
        """Load any previously cached vectors."""
        try:
            cache_dir = self.cache_dir / 'cache'
            for lang in self.VECTOR_CONFIGS:
                cache_file = cache_dir / f"{lang}_vectors.bin"
                if cache_file.exists():
                    try:
                        self._vectors[lang] = KeyedVectors.load(str(cache_file))
                        logger.info(f"Loaded cached vectors for {lang}")
                    except Exception as e:
                        logger.warning(f"Could not load cached vectors for {lang}: {e}")
        except Exception as e:
            logger.error(f"Error loading cached vectors: {e}")

    async def get_vectors(self, lang: str) -> Optional[Any]:
        """Get word vectors with mini vector fallback.
        
        Args:
            lang: Language code
            
        Returns:
            Word vectors or None if unavailable
        """
        try:
            # Use mini vectors if enabled
            if self._use_mini:
                return self._create_dummy_vectors(self._mini_dims.get(lang, 100))
                
            # Try getting real vectors
            if lang in self._vectors:
                return self._vectors[lang]
                
            # Load vectors if needed
            if lang in ['en', 'fr']:
                path = await self.download_vectors(lang)
                if path:
                    vectors = self._load_vectors(path)
                    if vectors:
                        self._vectors[lang] = vectors
                        return vectors
                        
            # Fall back to dummy vectors
            return self._create_dummy_vectors(300)  # Standard 300d vectors
            
        except Exception as e:
            logger.error(f"Error getting vectors for {lang}: {e}")
            return self._create_dummy_vectors(300)

    async def _download_vectors(self, config: Dict[str, Any]) -> Optional[Path]:
        """Download vectors with proper progress tracking."""
        local_path = self.cache_dir / config['local_path']
        
        async with self._download_lock:
            try:
                # Create parent directory
                local_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Download with progress bar
                async with httpx.AsyncClient() as client:
                    async with client.stream('GET', config['url']) as response:
                        total = int(response.headers['content-length'])
                        with tqdm(total=total, unit='iB', unit_scale=True) as pbar:
                            async with aiofiles.open(local_path, 'wb') as f:
                                async for chunk in response.aiter_bytes():
                                    await f.write(chunk)
                                    pbar.update(len(chunk))

                return local_path
                
            except Exception as e:
                logger.error(f"Error downloading vectors: {e}")
                return None

    def _load_vectors(self, path: Path) -> Any:
        """Load vectors from file."""
        try:
            return KeyedVectors.load(str(path))
        except Exception as e:
            logger.error(f"Error loading vectors: {e}")
            return None

    def _create_dummy_vectors(self, dim: int) -> Any:
        """Create minimal dummy vectors for testing."""
        vectors = KeyedVectors(vector_size=dim)
        vocab_size = 1000
        dummy_vectors = np.random.rand(vocab_size, dim).astype(np.float32)
        dummy_words = [f"dummy_{i}" for i in range(vocab_size)]
        vectors.add_vectors(dummy_words, dummy_vectors)
        return vectors

    async def _get_bilingual_vectors(
        self,
        fallback_to_dummy: bool = True
    ) -> Dict[str, Any]:
        """Load vectors for both languages."""
        vectors = {}
        errors = []

        for lang in ['en', 'fr']:
            try:
                lang_vectors = await self.get_vectors(
                    lang,
                    fallback_to_dummy=fallback_to_dummy
                )
                if lang_vectors is not None:
                    vectors[lang] = lang_vectors
                else:
                    errors.append(f"Failed to load {lang} vectors")
            except Exception as e:
                errors.append(f"Error loading {lang} vectors: {e}")

        if not vectors and errors:
            logger.error("Failed to load any vectors: " + "; ".join(errors))
            if fallback_to_dummy:
                return {
                    'en': self._get_dummy_vectors('en'),
                    'fr': self._get_dummy_vectors('fr')
                }
            return {}

        return vectors
        
    async def _verify_download(self, file_path: Path, expected_size: int) -> bool:
        """Verify downloaded file integrity."""
        try:
            if not file_path.exists():
                return False
                
            actual_size = file_path.stat().size
            if actual_size != expected_size:
                logger.warning(
                    f"Size mismatch for {file_path.name}: "
                    f"expected {expected_size}, got {actual_size}"
                )
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error verifying download: {e}")
            return False

    @backoff.on_exception(
        backoff.expo,
        (httpx.HTTPError, IOError),
        max_tries=MAX_RETRIES,
        max_time=3600  # Max 1 hour total
    )
    async def _download_with_resume(
        self,
        url: str,
        dest_path: Path,
        expected_size: Optional[int] = None,
        callback: Optional[callable] = None
    ) -> bool:
        """Download with resume capability and robust error handling."""
        temp_path = self.cache_dir / 'temp' / f"{dest_path.name}.partial"
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get current size of partial download
        current_size = temp_path.stat().st_size if temp_path.exists() else 0
        
        try:
            async with httpx.AsyncClient(timeout=self.TIMEOUT) as client:
                # First make a HEAD request to get the total size
                response = await client.head(url)
                response.raise_for_status()
                total_size = expected_size or int(response.headers['content-length'])
                
                if current_size >= total_size:
                    logger.info(f"File {dest_path.name} already completely downloaded")
                    if await self._verify_download(temp_path, total_size):
                        temp_path.rename(dest_path)
                        return True
                    current_size = 0  # Reset if verification failed
                
                # Prepare headers for resumed download
                headers = {'Range': f'bytes={current_size}-'} if current_size > 0 else {}
                
                async with client.stream('GET', url, headers=headers) as response:
                    response.raise_for_status()
                    
                    with tqdm(
                        total=total_size,
                        initial=current_size,
                        unit='iB',
                        unit_scale=True,
                        desc=f"Downloading {dest_path.name}"
                    ) as pbar:
                        async with aiofiles.open(temp_path, mode='ab' if current_size > 0 else 'wb') as f:
                            async for chunk in response.aiter_bytes(self.CHUNK_SIZE):
                                if chunk:
                                    await f.write(chunk)
                                    current_size += len(chunk)
                                    pbar.update(len(chunk))
                                    if callback:
                                        callback(current_size / total_size)
                
                # Verify download
                if await self._verify_download(temp_path, total_size):
                    temp_path.rename(dest_path)
                    return True
                    
                raise ValueError("Download verification failed")
                
        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            # Don't delete partial file to allow resume
            return False

    async def download_vectors(self, lang: str) -> Optional[Path]:
        """Download word vectors with proper error handling and resume capability."""
        if lang not in self.VECTOR_CONFIGS:
            logger.error(f"Unsupported language: {lang}")
            return None
            
        config = self.VECTOR_CONFIGS[lang]
        if not config['url']:
            logger.info(f"No direct download URL for {lang}, using fallback")
            return None
            
        dest_path = self.cache_dir / 'downloads' / config['name']
        
        try:
            # Check if file already exists and is valid
            if dest_path.exists() and await self._verify_download(dest_path, config['size']):
                logger.info(f"Using existing download for {lang}")
                return dest_path
                
            # Attempt download
            async with self._download_lock:  # Prevent concurrent downloads
                success = await self._download_with_resume(
                    config['url'],
                    dest_path,
                    config['size'],
                    lambda p: self._update_progress(lang, p)
                )
                
                if success:
                    logger.info(f"Successfully downloaded vectors for {lang}")
                    return dest_path
                    
                # Try fallback URL if available
                if config['fallback_url']:
                    logger.info(f"Attempting fallback download for {lang}")
                    success = await self._download_with_resume(
                        config['fallback_url'],
                        dest_path,
                        None,  # Size might be different for fallback
                        lambda p: self._update_progress(lang, p)
                    )
                    
                    if success:
                        return dest_path
                        
                logger.error(f"Failed to download vectors for {lang}")
                return None
                
        except Exception as e:
            logger.error(f"Error downloading vectors for {lang}: {e}")
            return None

    def _update_progress(self, lang: str, progress: float):
        """Update download progress."""
        self._download_progress[lang] = progress
        
    def get_download_progress(self, lang: str) -> float:
        """Get current download progress."""
        return self._download_progress.get(lang, 0.0)

    def get_word_similarity(
        self,
        word1: str,
        word2: str,
        lang: Optional[str] = None
    ) -> float:
        """Get similarity between two words.
        
        Args:
            word1: First word
            word2: Second word
            lang: Language code (for bilingual mode)
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            vectors = self._vectors.get(lang)
            if not vectors:
                return 0.0
                
            if isinstance(vectors, dict):  # Bilingual case
                similarities = []
                for lang_vectors in vectors.values():
                    try:
                        if word1 in lang_vectors and word2 in lang_vectors:
                            sim = np.dot(lang_vectors[word1], lang_vectors[word2]) / (
                                np.linalg.norm(lang_vectors[word1]) * 
                                np.linalg.norm(lang_vectors[word2])
                            )
                            similarities.append(sim)
                    except KeyError:
                        continue
                
                return max(similarities) if similarities else 0.0
                
            else:  # Single language case
                try:
                    if word1 in vectors and word2 in vectors:
                        return vectors.similarity(word1, word2)
                except KeyError:
                    pass
                    
            return 0.0
            
        except Exception as e:
            logger.error(f"Error getting word similarity: {e}")
            return 0.0

    async def calculate_similarity(
        self,
        word1: str,
        word2: str,
        lang: Optional[str] = None
    ) -> float:
        """Calculate similarity between words asynchronously."""
        try:
            # If vectors aren't loaded yet, load them
            if lang not in self._vectors and lang is not None:
                await self.get_vectors(lang)
            
            return self.get_word_similarity(word1, word2, lang)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0

    async def cleanup(self):
        """Clean up resources."""
        try:
            async with self._cache_lock:
                self._vectors.clear()
                self._dummy_vectors.clear()
                self._download_progress.clear()

            # Clean up temp files
            temp_dir = self.cache_dir / 'temp'
            if temp_dir.exists():
                for file in temp_dir.glob('*.partial'):
                    try:
                        file.unlink()
                    except OSError:
                        pass

            self._initialized = False
            logger.info("Word vector manager cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")