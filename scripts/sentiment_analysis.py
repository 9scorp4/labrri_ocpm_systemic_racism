import logging
import sqlite3
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.auto import tqdm

from scripts.database import Database

nltk.download('vader_lexicon')

tqdm.pandas()

class SentimentAnalysis:
    def __init__(self, db):
        self.db = Database(db)
        self.sia = SentimentIntensityAnalyzer()

    def analyze_sentiment(self, text):
        return self.sia.polarity_scores(text)['compound']
    
    def analyze_docs_by_language(self, lang):
        query = """
            SELECT d.id, d.organization, d.document_type, d.category, d.clientele, d.knowledge_type, c.content
            FROM documents d
            JOIN content c ON d.id = c.doc_id
            WHERE d.language = ?
        """
        df = pd.read_sql_query(query, self.db.conn, params=(lang,))

        logging.info(f"Analyzing sentiment for {len(df)} {lang} documents")
        df['sentiment'] = df['content'].progress_apply(self.analyze_sentiment)

        df_without_content = df.drop(columns=['content'])

        df_without_content.to_csv(rf"results/sentiment_analysis/{lang}.csv", index=False)
        df_without_content.to_excel(rf"results/sentiment_analysis/{lang}.xlsx", index=False)

        return df
    
    def analyze_all_docs(self):
        query = """SELECT DISTINCT language FROM documents"""
        langs = pd.read_sql_query(query, self.db.conn)['language'].tolist()

        all_results = []
        for lang in langs:
            df = self.analyze_docs_by_language(lang)
            all_results.append(df)

        return pd.concat(all_results, ignore_index=False)
    
    def get_avg_sentiment_by_category(self, category_column):
        all_docs = self.analyze_all_docs()
        return all_docs.groupby(category_column)['sentiment'].mean().sort_values(ascending=False)