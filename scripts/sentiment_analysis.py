from loguru import logger
import pandas as pd
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from nltk.tokenize import sent_tokenize
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, classification_report
import spacy
from sqlalchemy import text

from scripts.database import Database

# Download necessary NLTK data
nltk.download('punkt', quiet=True)

class SentimentAnalysis:
    def __init__(self, db):
        self.db = Database(db)
        
        # Initialize VADER sentiment analyzer
        self.vader = SentimentIntensityAnalyzer()
        
        # Initialize HuggingFace pipeline for multilingual sentiment analysis
        self.tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
        self.model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
        self.sentiment_pipeline = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)
        
        # Initialize spaCy for named entity recognition
        self.nlp = spacy.load("en_core_web_sm")
        
        # Define aspects related to systemic racism and discrimination
        self.aspects = ["race", "ethnicity", "discrimination", "equality", "diversity", "inclusion", "bias", "prejudice"]
        
        logger.info("SentimentAnalysis initialized successfully")

    def analyze_sentiment(self, text):
        # Combine VADER and HuggingFace sentiment analysis
        vader_sentiment = self.vader.polarity_scores(text)['compound']
        hf_sentiment = self.sentiment_pipeline(text[:512])[0]  # Limit to 512 tokens for BERT
        
        # Normalize HuggingFace sentiment score to [-1, 1] range
        hf_score = (int(hf_sentiment['label'][0]) - 3) / 2
        
        # Combine scores (you can adjust the weights)
        combined_score = (vader_sentiment + hf_score) / 2
        
        return combined_score

    def aspect_based_sentiment(self, text):
        doc = self.nlp(text)
        aspect_sentiments = {}
        
        for aspect in self.aspects:
            relevant_sentences = []
            for sent in doc.sents:
                if aspect in sent.text.lower():
                    relevant_sentences.append(sent.text)
            
            if relevant_sentences:
                aspect_score = sum(self.analyze_sentiment(sent) for sent in relevant_sentences) / len(relevant_sentences)
                aspect_sentiments[aspect] = aspect_score
        
        return aspect_sentiments

    def analyze_docs_by_language(self, lang):
        query = text("""
            SELECT d.id, d.organization, d.document_type, d.category, d.clientele, d.knowledge_type, c.content
            FROM documents d
            JOIN content c ON d.id = c.doc_id
            WHERE d.language = :lang
        """)
        with self.db.session_scope() as session:
            result = session.execute(query, {'lang': lang})
            df = pd.DataFrame(result.fetchall(), columns=result.keys())

        logger.info(f"Analyzing sentiment for {len(df)} {lang} documents")
        
        results = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            sentiment_score = self.analyze_sentiment(row['content'])
            aspect_sentiments = self.aspect_based_sentiment(row['content'])
            
            result = {
                'id': row['id'],
                'organization': row['organization'],
                'document_type': row['document_type'],
                'category': row['category'],
                'clientele': row['clientele'],
                'knowledge_type': row['knowledge_type'],
                'overall_sentiment': sentiment_score,
                **{f"aspect_{k}": v for k, v in aspect_sentiments.items()}
            }
            results.append(result)

        results_df = pd.DataFrame(results)
        
        results_df.to_csv(f"results/sentiment_analysis/{lang}_sentiment.csv", index=False)
        results_df.to_excel(f"results/sentiment_analysis/{lang}_sentiment.xlsx", index=False)

        logger.info(f"Sentiment analysis for {lang} documents completed")
        return results_df
    
    def analyze_all_docs(self):
        query = text("""SELECT DISTINCT language FROM documents""")
        
        with self.db.session_scope() as session:
            result = session.execute(query)
            langs = [row[0] for row in result]

        all_results = []
        for lang in langs:
            df = self.analyze_docs_by_language(lang)
            all_results.append(df)

        combined_results = pd.concat(all_results, ignore_index=True)
        
        logger.info("Sentiment analysis for all documents completed")
        return combined_results
    
    def get_avg_sentiment_by_category(self, category_column):
        all_docs = self.analyze_all_docs()
        return all_docs.groupby(category_column)['overall_sentiment'].mean().sort_values(ascending=False)

    def calibrate_model(self, labeled_data):
        """
        Calibrate the sentiment analysis model using labeled data specific to systemic racism and discrimination context.
        
        :param labeled_data: DataFrame with 'text' and 'label' columns
        """
        logger.info("Calibrating sentiment analysis model")
        
        predictions = []
        for text in tqdm(labeled_data['text']):
            predictions.append(self.analyze_sentiment(text))
        
        # Convert continuous predictions to categorical for evaluation
        categorical_predictions = ['positive' if p > 0.05 else 'negative' if p < -0.05 else 'neutral' for p in predictions]
        
        accuracy = accuracy_score(labeled_data['label'], categorical_predictions)
        report = classification_report(labeled_data['label'], categorical_predictions)
        
        logger.info(f"Calibration results:\nAccuracy: {accuracy}\n{report}")
        
        # TODO: Implement fine-tuning of the model based on these results
        
        return accuracy, report

# Usage example:
if __name__ == "__main__":
    db_path = "path/to/your/database.db"
    sentiment_analyzer = SentimentAnalysis(db_path)
    
    # Analyze all documents
    results = sentiment_analyzer.analyze_all_docs()
    print(results.head())
    
    # Get average sentiment by category
    avg_sentiment = sentiment_analyzer.get_avg_sentiment_by_category('category')
    print(avg_sentiment)
    
    # Calibrate the model (you would need labeled data for this)
    # labeled_data = pd.read_csv("path/to/labeled_data.csv")
    # accuracy, report = sentiment_analyzer.calibrate_model(labeled_data)