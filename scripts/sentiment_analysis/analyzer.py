from loguru import logger
import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
from sqlalchemy import text
from pathlib import Path
import json
from datetime import datetime

from scripts.database import Database

class SentimentAnalyzer:
    def __init__(self, db_path):
        """Initialize the sentiment analyzer with required models and components."""
        if not db_path:
            raise ValueError("Database path cannot be None")
            
        self.db = Database(db_path)
        
        # Initialize VADER sentiment analyzer
        self.vader = SentimentIntensityAnalyzer()
        
        # Initialize HuggingFace pipeline for multilingual sentiment
        model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.sentiment_pipeline = pipeline("sentiment-analysis", 
                                        model=self.model, 
                                        tokenizer=self.tokenizer)
        
        # Initialize spaCy models
        self.nlp = {
            'fr': spacy.load('fr_core_news_md'),
            'en': spacy.load('en_core_web_md')
        }
        
        # Define aspects related to systemic racism and discrimination
        self.aspects = [
            "discrimination", "racism", "prejudice", "bias",
            "equality", "equity", "diversity", "inclusion",
            "race", "ethnicity", "systemic", "institutional"
        ]
        
        logger.info("SentimentAnalyzer initialized successfully")

    def get_basic_sentiment(self, category=None):
        """Get basic sentiment analysis for documents."""
        try:
            query = """
                SELECT d.id, d.category, d.language, c.content
                FROM documents d
                JOIN content c ON d.id = c.doc_id
                WHERE c.content IS NOT NULL
            """
            params = {}
            
            if category and category != "All categories":
                query += " AND d.category = :category"
                params['category'] = category

            df = self.db.df_from_query(query, params)
            
            if df.empty:
                logger.warning("No documents found for analysis")
                return None

            results = []
            for _, row in df.iterrows():
                sentiment_scores = self._analyze_text_sentiment(row['content'])
                results.append({
                    'doc_id': row['id'],
                    'category': row['category'],
                    'language': row['language'],
                    'sentiment_score': sentiment_scores['compound'],
                    'positive_score': sentiment_scores['pos'],
                    'negative_score': sentiment_scores['neg'],
                    'neutral_score': sentiment_scores['neu']
                })

            return pd.DataFrame(results)

        except Exception as e:
            logger.error(f"Error in basic sentiment analysis: {str(e)}")
            return None

    def get_aspect_based_sentiment(self, category=None):
        """Analyze sentiment for specific aspects in documents."""
        try:
            query = """
                SELECT d.id, d.category, d.language, c.content
                FROM documents d
                JOIN content c ON d.id = c.doc_id
                WHERE c.content IS NOT NULL
            """
            params = {}
            
            if category and category != "All categories":
                query += " AND d.category = :category"
                params['category'] = category

            df = self.db.df_from_query(query, params)
            
            if df.empty:
                logger.warning("No documents found for analysis")
                return None

            results = []
            for _, row in df.iterrows():
                aspect_sentiments = self._analyze_aspect_sentiment(row['content'])
                aspect_sentiments.update({
                    'doc_id': row['id'],
                    'category': row['category'],
                    'language': row['language']
                })
                results.append(aspect_sentiments)

            return pd.DataFrame(results)

        except Exception as e:
            logger.error(f"Error in aspect-based sentiment analysis: {str(e)}")
            return None

    def get_comparative_sentiment(self, categories=None):
        """Compare sentiment across different categories."""
        try:
            if not categories:
                return None

            results = []
            for category in categories:
                df = self.get_basic_sentiment(category)
                if df is not None and not df.empty:
                    avg_sentiment = df['sentiment_score'].mean()
                    pos_ratio = (df['sentiment_score'] > 0).mean()
                    neg_ratio = (df['sentiment_score'] < 0).mean()
                    
                    results.append({
                        'category': category,
                        'avg_sentiment': avg_sentiment,
                        'positive_ratio': pos_ratio,
                        'negative_ratio': neg_ratio,
                        'document_count': len(df)
                    })

            return pd.DataFrame(results) if results else None

        except Exception as e:
            logger.error(f"Error in comparative sentiment analysis: {str(e)}")
            return None

    def _analyze_text_sentiment(self, text):
        """Analyze sentiment of a text using both VADER and HuggingFace."""
        # Get VADER sentiment
        vader_scores = self.vader.polarity_scores(text)
        
        # Get HuggingFace sentiment
        try:
            # Truncate text to fit model's maximum length
            hf_result = self.sentiment_pipeline(text[:512])[0]
            hf_score = (int(hf_result['label'][0]) - 3) / 2  # Normalize to [-1, 1]
            
            # Combine scores (weighted average)
            compound = (vader_scores['compound'] + hf_score) / 2
            
            return {
                'compound': compound,
                'pos': vader_scores['pos'],
                'neg': vader_scores['neg'],
                'neu': vader_scores['neu']
            }
        except Exception as e:
            logger.warning(f"Error in HuggingFace analysis: {str(e)}, using VADER scores only")
            return vader_scores

    def _analyze_aspect_sentiment(self, text):
        """Analyze sentiment for specific aspects in the text."""
        doc_en = self.nlp['en'](text)
        doc_fr = self.nlp['fr'](text)
        
        aspect_sentiments = {aspect: [] for aspect in self.aspects}
        
        # Analyze sentences from both language models
        for doc in [doc_en, doc_fr]:
            for sent in doc.sents:
                sent_text = sent.text.lower()
                for aspect in self.aspects:
                    if aspect in sent_text:
                        sentiment = self._analyze_text_sentiment(sent_text)['compound']
                        aspect_sentiments[aspect].append(sentiment)

        # Calculate average sentiment for each aspect
        results = {}
        for aspect, sentiments in aspect_sentiments.items():
            if sentiments:
                results[f'{aspect}_sentiment'] = np.mean(sentiments)
                results[f'{aspect}_mentions'] = len(sentiments)
            else:
                results[f'{aspect}_sentiment'] = 0
                results[f'{aspect}_mentions'] = 0

        return results