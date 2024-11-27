"""Enhanced language detection with proper profile initialization and thread safety."""
import re
from typing import Tuple, Dict, Any, Optional
from loguru import logger
from langdetect import detect, DetectorFactory, LangDetectException
import spacy
from spacy.tokens import Doc
from spacy.language import Language
import threading
import asyncio

class LanguageDetector:
    """Enhanced language detection with French and English specific rules."""
    
    # French-specific patterns
    FRENCH_PATTERNS = [
        r'\b(le|la|les|un|une|des|du|au|aux)\b',  # Articles
        r'\b(je|tu|il|elle|nous|vous|ils|elles)\b',  # Personal pronouns
        r'\b(être|avoir|faire|dire|voir|aller|venir|pouvoir|vouloir)\b',  # Common verbs
        r'\b(dans|sur|avec|sans|pour|par|chez|avant|après)\b',  # Prepositions
        r'[àâçèéêëîïôùûü]',  # French accents
        r'\b(monsieur|madame|mesdames|messieurs)\b',  # Honorifics
        r'\b(ce|cet|cette|ces|celui|celle|ceux|celles)\b',  # Demonstratives
        r'\b(notre|votre|leur|nos|vos|leurs)\b',  # Possessives
        r'\b(pourquoi|comment|quand|où|qui|que|quoi)\b',  # Question words
        r'\b(très|beaucoup|peu|trop|assez|plus|moins)\b'  # Adverbs
    ]
    
    # English-specific patterns
    ENGLISH_PATTERNS = [
        r'\b(the|a|an|this|that|these|those)\b',  # Articles and demonstratives
        r'\b(I|you|he|she|it|we|they|my|your|his|her|its|our|their)\b',  # Pronouns
        r'\b(is|are|was|were|has|have|had|do|does|did)\b',  # Auxiliary verbs
        r'\b(in|on|at|by|for|with|from|to|of)\b',  # Prepositions
        r'\b(Mr\.|Mrs\.|Ms\.|Dr\.)\b',  # Honorifics
        r'\b(will|would|shall|should|can|could|may|might)\b',  # Modal verbs
        r"\b(isn't|aren't|wasn't|weren't|haven't|hasn't|don't|doesn't|didn't)\b",  # Contractions
        r'\b(very|much|many|more|most|some|any)\b',  # Quantifiers
        r'\b(why|how|when|where|who|what|which)\b',  # Question words
        r'\b(first|second|third|fourth|fifth)\b'  # Ordinal numbers
    ]
    
    def __init__(self, nlp_fr: Language, nlp_en: Language):
        """Initialize with spaCy models."""
        self.nlp_fr = nlp_fr
        self.nlp_en = nlp_en
        
        # Initialize langdetect with seed for reproducibility
        DetectorFactory.seed = 0
        # Load profiles
        try:
            detect("test")  # Force profile loading
        except LangDetectException:
            pass  # Profile loading error is expected here
            
        # Thread safety for language detection
        self._lock = threading.Lock()
        
        logger.info("Language detector initialized with French and English models")
        
    async def detect_language(self, text: str) -> Tuple[Optional[str], Dict[str, Any]]:
        """Detect language with enhanced validation and bilingual rules."""
        metadata = {
            'original_text': text[:100] + '...' if len(text) > 100 else text,
            'text_length': len(text),
            'word_count': len(text.split()),
            'validation_passed': False,
            'detection_method': None,
            'pattern_matches': {
                'french': 0,
                'english': 0
            },
            'confidence_scores': {}
        }
        
        if not self._validate_input(text):
            return None, metadata
            
        metadata['validation_passed'] = True
        
        try:
            with self._lock:  # Thread-safe detection
                # Check for language-specific patterns
                french_patterns_found = sum(
                    1 for pattern in self.FRENCH_PATTERNS 
                    if re.search(pattern, text, re.IGNORECASE)
                )
                english_patterns_found = sum(
                    1 for pattern in self.ENGLISH_PATTERNS 
                    if re.search(pattern, text, re.IGNORECASE)
                )
                
                metadata['pattern_matches'] = {
                    'french': french_patterns_found,
                    'english': english_patterns_found
                }
                
                # Initial detection with langdetect
                try:
                    initial_lang = detect(text)
                    metadata['detection_method'] = 'langdetect'
                except LangDetectException:
                    # Fallback to pattern-based detection
                    initial_lang = 'fr' if french_patterns_found > english_patterns_found else 'en'
                    metadata['detection_method'] = 'pattern_fallback'
                
                # Verify with spaCy models
                loop = asyncio.get_event_loop()
                doc_fr = await loop.run_in_executor(None, self.nlp_fr, text[:1000])
                doc_en = await loop.run_in_executor(None, self.nlp_en, text[:1000])
                
                # Calculate confidence scores
                total_words = len(text.split())
                if total_words > 0:
                    fr_words = sum(1 for token in doc_fr if token.is_alpha)
                    en_words = sum(1 for token in doc_en if token.is_alpha)
                    
                    fr_confidence = fr_words / total_words
                    en_confidence = en_words / total_words
                    
                    # Adjust confidence scores based on pattern matches
                    pattern_weight = 0.3
                    fr_confidence = (fr_confidence * (1 - pattern_weight) + 
                                   (french_patterns_found / max(1, french_patterns_found + english_patterns_found)) * pattern_weight)
                    en_confidence = (en_confidence * (1 - pattern_weight) + 
                                   (english_patterns_found / max(1, french_patterns_found + english_patterns_found)) * pattern_weight)
                else:
                    fr_confidence = en_confidence = 0
                
                metadata['confidence_scores'] = {
                    'fr': fr_confidence,
                    'en': en_confidence
                }
                
                # Enhanced decision logic with strong indicators
                if french_patterns_found > english_patterns_found * 1.5:
                    final_lang = 'fr'
                    metadata['detection_method'] = 'french_pattern_dominance'
                elif english_patterns_found > french_patterns_found * 1.5:
                    final_lang = 'en'
                    metadata['detection_method'] = 'english_pattern_dominance'
                else:
                    if fr_confidence > en_confidence * 1.2:
                        final_lang = 'fr'
                    elif en_confidence > fr_confidence * 1.2:
                        final_lang = 'en'
                    else:
                        final_lang = initial_lang
                    metadata['detection_method'] = 'confidence_weights'
                
                # Additional context confirmation
                if self._has_strong_context_indicators(text, final_lang):
                    metadata['detection_method'] = f'context_confirmed_{final_lang}'
                
                logger.debug(
                    f"Language detection result: {final_lang} "
                    f"(method: {metadata['detection_method']}, "
                    f"fr_conf: {fr_confidence:.2f}, "
                    f"en_conf: {en_confidence:.2f})"
                )
                
                return final_lang, metadata
                
        except Exception as e:
            logger.error(f"Language detection failed: {str(e)}")
            metadata['error'] = str(e)
            return None, metadata
    
    def _validate_input(self, text: str) -> bool:
        """Validate input text."""
        if not isinstance(text, str):
            logger.warning(f"Invalid input type: {type(text)}. Expected str.")
            return False
            
        if not text.strip():
            logger.warning("Empty or whitespace-only input text.")
            return False
            
        # Check for minimum meaningful content
        if len(text.split()) < 3:
            logger.warning(f"Text too short for reliable detection: '{text}'")
            return False
            
        return True
    
    def _has_strong_context_indicators(self, text: str, predicted_lang: str) -> bool:
        """Check for strong contextual language indicators."""
        # French context indicators
        if predicted_lang == 'fr':
            strong_fr_indicators = [
                r'\b(québec|montréal|laval|gatineau)\b',
                r'\b(gouvernement du québec|province de québec)\b',
                r'\b(université|école|collège)\b',
                r'\b(ministère|département)\b'
            ]
            return any(re.search(pattern, text, re.IGNORECASE) for pattern in strong_fr_indicators)
            
        # English context indicators
        elif predicted_lang == 'en':
            strong_en_indicators = [
                r'\b(canadian|canada|ontario|british columbia)\b',
                r'\b(government of canada|province of)\b',
                r'\b(university|college|school)\b',
                r'\b(department|ministry)\b'
            ]
            return any(re.search(pattern, text, re.IGNORECASE) for pattern in strong_en_indicators)
            
        return False