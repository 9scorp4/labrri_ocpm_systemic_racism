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

        self.custom_stopwords = {
            'entre', 'ainsi', 'leurs', 'beaucoup', 'souvent', 'dire', 'plus', 'cette', 'fait', 'faire', 'donc',
            'aussi', 'parce', 'peut', 'avoir', 'autres', 'sténo', 'tout', 'alors', 'vraiment', 'bien', 'être',
            'quand', 'puis', 'très', 'faut', 'comme', 'ariane', 'émond', 'a', 'plus', 'comme', 'cette', 'ça', 'fait',
            'être', 'faire', 'mme', 'donc', 'aussi', 'autres', 'si', 'entre', 'bien', 'tout', 'g', 'peut', 'leurs',
            'o', 'gh', 'avoir', 'non', 'the', 'de', 'la', 'et', 'des', 'le', 'les', 'l', 'may', 'would', 'also', 'see',
            'one', 'http', 'à', 'du', 'like', 'coprésidente', 'well', 'non', 'think', 'see', 'xurukulasuriya', 'dexter',
            'plus', 'aussi', 'très', 'get', 'mme', 'novembre', 'séance', 'sténo', 'mmm', 'commissaire', 'coprésidente',
            'know', 'sarah', 'soirée', 'go', 'oui', 'holness', 'ça', 'émond', 'thierry', 'thuot', 'lindor', 'merci',
            'would', 'balarama', 'ariane', 'like', 'lot', 'donc', 'fait', 'si', 'comme', 'judy', 'ouellet', 'one',
            'years', 'parce', 'going', 'pinet', 'monsieur', 'avoir', 'dit'
        }

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

        plt.figure(figsize=(12, 6))
        plt.bar(words, frequencies)
        plt.title(f"Top {n} {ngram}-grams in {'category: ' + where if lang is None else 'language: ' + lang}")
        plt.xlabel("Words")
        plt.ylabel("Frequency")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'results/word_frequency/top_{n}_{ngram}grams_{"category_" + where if lang is None else "lang_" + lang}.png')
        plt.close()

        df_result = pd.DataFrame(most_common, columns=[f'{ngram}-gram', 'Frequency'])
        df_result[f'{ngram}-gram'] = df_result[f'{ngram}-gram'].apply(lambda x: ' '.join(x))
        df_result.to_csv(f'results/word_frequency/top_{n}_{ngram}grams_{"category_" + where if lang is None else "lang_" + lang}.csv', index=False)

        logger.info(f'Analysis complete for {"category: " + where if lang is None else "language: " + lang}')
        return df_result

    def compare_categories(self, categories, n=20, ngram=1):
        logger.info(f'Comparing top {n} {ngram}-grams across categories: {categories}')
        results = {}
        for category in categories:
            results[category] = self.top_n_words(category, n, ngram)

        plt.figure(figsize=(15, 8))
        x = list(range(n))
        width = 0.8 / len(categories)
        
        for i, (category, df) in enumerate(results.items()):
            if df is not None:
                plt.bar([xi + i * width for xi in x], df['Frequency'], width, label=category)

        plt.xlabel(f'{ngram}-grams')
        plt.ylabel('Frequency')
        plt.title(f'Top {n} {ngram}-grams Comparison Across Categories')
        plt.legend()
        plt.xticks([xi + width * (len(categories) - 1) / 2 for xi in x], 
                   results[categories[0]][f'{ngram}-gram'] if results[categories[0]] is not None else [], 
                   rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'results/word_frequency/category_comparison_{ngram}grams.png')
        plt.close()

        logger.info('Category comparison complete')

    def compare_languages(self, n=20, ngram=1):
        logger.info(f'Comparing top {n} {ngram}-grams across languages')
        results = {}
        for lang in ['fr', 'en']:
            results[lang] = self.top_n_words(lang, n, ngram, lang=lang)

        plt.figure(figsize=(15, 8))
        x = list(range(n))
        width = 0.35
        
        for i, (lang, df) in enumerate(results.items()):
            if df is not None:
                plt.bar([xi + i * width for xi in x], df['Frequency'], width, label=lang)

        plt.xlabel(f'{ngram}-grams')
        plt.ylabel('Frequency')
        plt.title(f'Top {n} {ngram}-grams Comparison Across Languages')
        plt.legend()
        plt.xticks([xi + width / 2 for xi in x], 
                   results['fr'][f'{ngram}-gram'] if results['fr'] is not None else [], 
                   rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'results/word_frequency/language_comparison_{ngram}grams.png')
        plt.close()

        logger.info('Language comparison complete')

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