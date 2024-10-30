import pandas as pd
from loguru import logger
from typing import Optional, Dict, Any, Tuple
from langdetect import detect, DetectorFactory, LangDetectException
import spacy
from spacy.tokens import Doc
from spacy.language import Language

class LanguageDetector:
    """Enhanced language detection with validation and error handling."""
    
    def __init__(self, nlp_fr: Language, nlp_en: Language):
        """
        Initialize the language detector with spaCy models.
        
        Args:
            nlp_fr: French spaCy model
            nlp_en: English spaCy model
        """
        self.nlp_fr = nlp_fr
        self.nlp_en = nlp_en
        # Set seed for reproducibility
        DetectorFactory.seed = 0
        
    def validate_input(self, text: str) -> bool:
        """
        Validate input text before processing.
        
        Args:
            text: Input text to validate
            
        Returns:
            bool: Whether the text is valid for processing
        """
        if not isinstance(text, str):
            logger.warning(f"Invalid input type: {type(text)}. Expected str.")
            return False
            
        if not text.strip():
            logger.warning("Empty or whitespace-only input text.")
            return False
            
        # Check for minimum meaningful content (e.g., at least 3 words)
        if len(text.split()) < 3:
            logger.warning(f"Text too short for reliable detection: '{text}'")
            return False
            
        return True

    def detect_language(self, text: str) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Detect the language of a text with enhanced validation and confidence scoring.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Tuple containing:
                - Detected language code or None if detection fails
                - Dictionary with metadata about the detection process
        """
        metadata = {
            'original_text': text[:100] + '...' if len(text) > 100 else text,
            'text_length': len(text),
            'word_count': len(text.split()),
            'validation_passed': False,
            'detection_method': None,
            'confidence_scores': {}
        }
        
        if not self.validate_input(text):
            return None, metadata
            
        metadata['validation_passed'] = True
        
        try:
            # Primary detection using langdetect
            lang = detect(text)
            metadata['detection_method'] = 'langdetect'
            
            # Verify with spaCy models
            doc_fr = self.nlp_fr(text[:1000])  # Limit text length for performance
            doc_en = self.nlp_en(text[:1000])
            
            # Calculate confidence scores
            fr_words = sum(1 for token in doc_fr if token.lang_ == 'fr')
            en_words = sum(1 for token in doc_en if token.lang_ == 'en')
            
            total_words = len(text.split())
            fr_confidence = fr_words / total_words if total_words > 0 else 0
            en_confidence = en_words / total_words if total_words > 0 else 0
            
            metadata['confidence_scores'] = {
                'fr': fr_confidence,
                'en': en_confidence
            }
            
            # Verify initial detection with spaCy confidence scores
            if lang == 'fr' and fr_confidence < 0.3:
                if en_confidence > fr_confidence:
                    lang = 'en'
                    metadata['detection_method'] = 'spacy_override'
            elif lang == 'en' and en_confidence < 0.3:
                if fr_confidence > en_confidence:
                    lang = 'fr'
                    metadata['detection_method'] = 'spacy_override'
            
            return lang, metadata
            
        except LangDetectException as e:
            logger.error(f"Language detection failed: {str(e)}")
            metadata['error'] = str(e)
            return None, metadata