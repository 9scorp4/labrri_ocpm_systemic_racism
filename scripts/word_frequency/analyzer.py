# word_frequency.py - Analysis Logic
import os
from loguru import logger
import pandas as pd
from sqlalchemy import text
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from scripts.database import Database

class WordFrequencyAnalyzer:
    """Handles the core word frequency analysis logic."""
    
    def __init__(self, db_path):
        if db_path is None:
            raise ValueError('db_path cannot be None')

        self.db_path = db_path
        self.db = Database(self.db_path)
        self.valid_languages = {'fr', 'en'}

        # Initialize stopwords
        self.en_stopwords = set(stopwords.words('english'))
        self.fr_stopwords = set(stopwords.words('french'))
        
        # Load custom stopwords
        self._load_custom_stopwords()

    def _load_custom_stopwords(self):
        """Load custom stopwords from file."""
        try:
            with open('scripts/topic_analysis/stopwords.txt', 'r', encoding='utf-8') as f:
                self.custom_stopwords = set(line.strip().lower() for line in f if line.strip())
            self.all_stopwords = self.en_stopwords.union(self.fr_stopwords, self.custom_stopwords)
            logger.info(f"Loaded {len(self.custom_stopwords)} custom stopwords")
        except Exception as e:
            logger.error(f"Error loading custom stopwords: {e}")
            self.custom_stopwords = set()
            self.all_stopwords = self.en_stopwords.union(self.fr_stopwords)

    def get_word_frequencies(self, category=None, n=20, ngram=1, lang=None):
        """Get word frequencies with proper parameter validation."""
        # Validate language parameter
        if lang and lang.lower() not in self.valid_languages:
            logger.warning(f"Invalid language specified: {lang}")
            return None

        # Normalize parameters
        category = category if category and category != 'all' else None
        lang = lang.lower() if lang else None
        
        logger.info(f'Analyzing frequencies for category: {category}, language: {lang}')
        
        try:
            # Build and execute query with proper parameter handling
            query = """
                SELECT c.content 
                FROM content c 
                JOIN documents d ON c.doc_id = d.id 
                WHERE 1=1
            """
            params = {}
            
            if category:
                query += " AND d.category = :category"
                params['category'] = category
            if lang:
                query += " AND d.language = :language"
                params['language'] = lang

            logger.debug(f"Executing query: {query} with params: {params}")
            
            # Execute query
            with self.db.engine.connect() as connection:
                df = pd.read_sql_query(sql=text(query), con=connection, params=params)
            
            if df.empty:
                logger.warning(f"No data found for category: {category}, language: {lang}")
                return None

            # Process content
            all_ngrams = []
            for content in df['content'].dropna():
                tokens = self.tokenize_and_clean(str(content), lang)
                if tokens:
                    all_ngrams.extend(self._get_ngrams(tokens, ngram))

            if not all_ngrams:
                logger.warning("No valid n-grams generated")
                return None

            # Calculate frequencies
            ngram_freq = Counter(all_ngrams)
            most_common = ngram_freq.most_common(n)

            # Create DataFrame
            result_df = pd.DataFrame(most_common, columns=['Word', 'Frequency'])
            if ngram > 1:
                result_df['Word'] = result_df['Word'].apply(lambda x: ' '.join(x))
            
            logger.info(f"Generated frequency analysis with {len(result_df)} words")
            return result_df

        except Exception as e:
            logger.error(f"Error in word frequency analysis: {str(e)}", exc_info=True)
            return None

    def _build_query(self, category=None, lang=None):
        """Build PostgreSQL query with proper parameter handling."""
        base_query = """
            SELECT c.content 
            FROM content c 
            JOIN documents d ON c.doc_id = d.id 
            WHERE 1=1
        """
        
        if category and category != 'all':
            base_query += f" AND d.category = '{category}'"
        if lang:
            base_query += f" AND d.language = '{lang}'"
            
        return base_query

    def tokenize_and_clean(self, text, lang=None):
        """Tokenize and clean text with proper stopword handling."""
        try:
            tokens = word_tokenize(text.lower())
            stopwords = self._get_stopwords(lang)
            return [word for word in tokens 
                   if word.isalpha() and word not in stopwords]
        except Exception as e:
            logger.error(f"Error in tokenization: {str(e)}")
            return []

    def _get_stopwords(self, lang=None):
        """Get appropriate stopwords based on language."""
        if lang == 'fr':
            return self.fr_stopwords
        elif lang == 'en':
            return self.en_stopwords
        return self.all_stopwords

    def _get_ngrams(self, tokens, n):
        """Generate n-grams from tokens."""
        if n == 1:
            return tokens
        return list(ngrams(tokens, n))

    def compare_categories(self, categories, n=20, ngram=1):
        """Compare word frequencies across categories."""
        if not categories or len(categories) < 2:
            logger.warning("At least two categories are required for comparison")
            return None
            
        results = {}
        for category in categories:
            try:
                df_result = self.get_word_frequencies(category, n=n, ngram=ngram)
                if df_result is not None and not df_result.empty:
                    # Ensure consistent column names
                    df_result.columns = ['Word', 'Frequency']
                    results[category] = {
                        'words': df_result['Word'].tolist(),
                        'frequencies': df_result['Frequency'].tolist()
                    }
                else:
                    logger.warning(f"No data found for category: {category}")
            except Exception as e:
                logger.error(f"Error processing category {category}: {str(e)}")
                
        if not results:
            logger.warning("No valid results found for any category")
            return None
            
        return results

    def compare_languages(self, n=20, ngram=1):
        """Compare word frequencies across languages."""
        results = {}
        for lang in ['fr', 'en']:
            df_result = self.get_word_frequencies(
                category='all', 
                n=n, 
                ngram=ngram, 
                lang=lang
            )
            if df_result is not None:
                results[lang] = {
                    'words': df_result['Word'].tolist(),
                    'frequencies': df_result['Frequency'].tolist()
                }
        return results

    def calculate_tfidf(self, category=None, lang=None):
        """Calculate TF-IDF scores with improved structure and error handling."""
        try:
            # Build and execute query with proper filtering
            query = """
                SELECT c.content 
                FROM content c 
                JOIN documents d ON c.doc_id = d.id 
                WHERE 1=1
            """
            params = {}
            
            if category:
                query += " AND d.category = :category"
                params['category'] = category
            if lang:
                query += " AND d.language = :language"
                params['language'] = lang

            with self.db.engine.connect() as connection:
                df = pd.read_sql_query(sql=text(query), con=connection, params=params)

            if df.empty:
                logger.warning(f"No documents found for category: {category}, language: {lang}")
                return None

            # Configure TF-IDF vectorizer with improved parameters
            vectorizer = TfidfVectorizer(
                stop_words=list(self._get_stopwords(lang)),
                max_features=1000,
                ngram_range=(1, 1),  # Unigrams only for TF-IDF
                min_df=2,  # Minimum document frequency
                max_df=0.95  # Maximum document frequency
            )
            
            # Calculate TF-IDF
            tfidf_matrix = vectorizer.fit_transform(df['content'].fillna(''))
            feature_names = vectorizer.get_feature_names_out()
            
            # Calculate mean TF-IDF scores across all documents
            mean_tfidf = pd.DataFrame(
                tfidf_matrix.mean(axis=0).A1,
                index=feature_names,
                columns=['Frequency']  # Using 'Frequency' to match visualizer expectations
            )
            
            # Sort and prepare the results
            top_terms = mean_tfidf.nlargest(20, 'Frequency')
            result_df = top_terms.reset_index().rename(columns={'index': 'Word'})
            
            # Scale the frequencies to be more visually appealing
            result_df['Frequency'] = result_df['Frequency'] * 100
            
            logger.info(f"Generated TF-IDF analysis with {len(result_df)} terms")
            return result_df
            
        except Exception as e:
            logger.error(f"Error calculating TF-IDF: {str(e)}", exc_info=True)
            return None