import os
from loguru import logger
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from scripts.database import Database
import seaborn as sns

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

class WordFrequencyChart:
    def __init__(self, db_path):
        if db_path is None:
            raise ValueError('db_path cannot be None')

        self.db_path = db_path
        self.db = Database(self.db_path)

        self.en_stopwords = set(stopwords.words('english'))
        self.fr_stopwords = set(stopwords.words('french'))

        with open(r'scripts\topic_analysis\stopwords.txt', 'r', encoding='utf-8') as f:
            self.custom_stopwords = set(line.strip().lower() for line in f if line.strip())

        self.all_stopwords = self.en_stopwords.union(self.fr_stopwords, self.custom_stopwords)

    def tokenize_and_clean(self, text, lang=None):
        tokens = word_tokenize(text.lower())
        if lang == 'fr':
            stopwords = self.fr_stopwords
        elif lang == 'en':
            stopwords = self.en_stopwords
        else:
            stopwords = self.all_stopwords
        return [word for word in tokens if word.isalpha() and word not in stopwords and word not in self.custom_stopwords]

    def get_ngrams(self, tokens, n):
        return list(ngrams(tokens, n))

    def top_n_words(self, where, n=20, ngram=1, lang=None):
        logger.info(f'Analyzing top {n} {ngram}-grams for {"category: " + where if lang is None else "language: " + lang}')
        
        if lang is None:
            query = f"SELECT c.content FROM content c JOIN documents d ON c.doc_id = d.id WHERE category = '{where}'"
        else:
            query = f"SELECT c.content FROM content c JOIN documents d ON c.doc_id = d.id WHERE d.language = '{lang}'"
        
        df = self.db.df_from_query(query)
        
        if df is None or df.empty:
            logger.warning(f"No data found for {'category: ' + where if lang is None else 'language: ' + lang}")
            return None

        all_ngrams = []
        for content in df['content']:
            tokens = self.tokenize_and_clean(content, lang)
            all_ngrams.extend(self.get_ngrams(tokens, ngram))

        ngram_freq = Counter(all_ngrams)
        most_common = ngram_freq.most_common(n)

        words, frequencies = zip(*most_common)
        words = [' '.join(word) for word in words]

        df_result = pd.DataFrame(most_common, columns=[f'{ngram}-gram', 'Frequency'])
        df_result[f'{ngram}-gram'] = df_result[f'{ngram}-gram'].apply(lambda x: ' '.join(x))

        # Create a valid filename
        filename = f"top_{n}_{ngram}grams_{'category_' + where if lang is None else 'lang_' + lang}.png"
        save_path = os.path.join('results', 'word_frequency', filename)

        # Use the plot_word_frequency function
        self.plot_word_frequency(df_result.set_index(f'{ngram}-gram'), 
                                 f"Top {n} {ngram}-grams in {'category: ' + where if lang is None else 'language: ' + lang}",
                                 save_path=save_path)

        df_result.to_csv(save_path.replace('.png', '.csv'), index=False)

        logger.info(f'Analysis complete for {"category: " + where if lang is None else "language: " + lang}')
        return df_result

    @staticmethod
    def plot_word_frequency(data, title, save_path=None):
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Frequency', y=data.index, data=data)
        plt.title(title)
        plt.xlabel('Frequency')
        plt.ylabel('Words/N-grams')
        
        if save_path:
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            logger.info(f"Plot saved to: {save_path}")
        else:
            plt.show()

    def compare_languages(self, n=20, ngram=1):
        """Compare top n words across languages, return processed data."""
        logger.info(f'Comparing top {n} {ngram}-grams across languages')
        results = {}
        
        for lang in ['fr', 'en']:
            df_result = self.top_n_words(lang, n=n, ngram=ngram, lang=lang)
            if df_result is not None:
                # Always ensure we have the correct column names from top_n_words
                df_result = df_result.rename(columns={
                    f'{ngram}-gram': 'word',  # Standardize to 'word' column
                    'Frequency': 'frequency'  # Standardize to lowercase
                })
                
                results[lang] = {
                    'words': df_result['word'].tolist(),
                    'frequencies': df_result['frequency'].tolist()
                }
        
        logger.debug(f"Compare languages results: {results}")  # Debug log
        return results

    def compare_categories(self, categories, n=20, ngram=1):
        """Compare top n words across categories, return processed data."""
        logger.info(f'Comparing top {n} {ngram}-grams across categories: {categories}')
        results = {}
        
        for category in categories:
            df_result = self.top_n_words(category, n=n, ngram=ngram)
            if df_result is not None:
                # Always ensure we have the correct column names from top_n_words
                df_result = df_result.rename(columns={
                    f'{ngram}-gram': 'word',  # Standardize to 'word' column
                    'Frequency': 'frequency'  # Standardize to lowercase
                })
                
                results[category] = {
                    'words': df_result['word'].tolist(),
                    'frequencies': df_result['frequency'].tolist()
                }
        
        logger.debug(f"Compare categories results: {results}")  # Debug log
        return results

    def tfidf_analysis(self, where, lang=None):
        logger.info(f'Performing TF-IDF analysis for {"category: " + where if lang is None else "language: " + lang}')
        
        if lang is None:
            query = f"SELECT c.content FROM content c JOIN documents d ON c.doc_id = d.id WHERE category = '{where}'"
        else:
            query = f"SELECT c.content FROM content c JOIN documents d ON c.doc_id = d.id WHERE d.language = '{lang}'"
        
        df = self.db.df_from_query(query)
        
        if df is None or df.empty:
            logger.warning(f"No data found for {'category: ' + where if lang is None else 'language: ' + lang}")
            return None

        stopwords = self.fr_stopwords if lang == 'fr' else self.en_stopwords if lang == 'en' else self.all_stopwords
        vectorizer = TfidfVectorizer(stop_words=list(stopwords.union(self.custom_stopwords)))
        tfidf_matrix = vectorizer.fit_transform(df['content'])

        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.sum(axis=0).A1
        word_scores = list(zip(feature_names, tfidf_scores))
        word_scores.sort(key=lambda x: x[1], reverse=True)

        top_words = word_scores[:20]
        words, scores = zip(*top_words)

        plt.figure(figsize=(12, 6))
        plt.bar(words, scores)
        plt.title(f"Top 20 words by TF-IDF score in {'category: ' + where if lang is None else 'language: ' + lang}")
        plt.xlabel("Words")
        plt.ylabel("TF-IDF Score")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'results/word_frequency/tfidf_top20_{"category_" + where if lang is None else "lang_" + lang}.png')
        plt.close()

        df_result = pd.DataFrame(top_words, columns=['Word', 'TF-IDF Score'])
        df_result.to_csv(f'results/word_frequency/tfidf_top20_{"category_" + where if lang is None else "lang_" + lang}.csv', index=False)

        logger.info(f'TF-IDF analysis complete for {"category: " + where if lang is None else "language: " + lang}')
        return df_result

    def __del__(self):
        logger.info('WordFrequencyChart object is being destroyed')