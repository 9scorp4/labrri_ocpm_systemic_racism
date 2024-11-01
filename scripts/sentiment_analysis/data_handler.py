from loguru import logger
import pandas as pd
import numpy as np

class SentimentDataHandler:
    """Handles data processing and validation for sentiment analysis."""
    
    EXPECTED_COLUMNS = {
        'basic': ['doc_id', 'category', 'language', 'sentiment_score', 
                 'positive_score', 'negative_score', 'neutral_score'],
        'aspect': ['doc_id', 'category', 'language'],
        'temporal': ['doc_id', 'category', 'language', 'sentiment_score'],
        'comparative': ['category', 'avg_sentiment', 'positive_ratio', 
                       'negative_ratio', 'document_count']
    }

    @staticmethod
    def validate_and_format_basic_sentiment(data):
        """Validate and format basic sentiment analysis data."""
        try:
            if data is None or data.empty:
                logger.warning("Empty data provided for basic sentiment validation")
                return None

            # Create a copy to avoid modifying original data
            formatted_data = data.copy()
            
            # Validate required columns
            missing_cols = [col for col in SentimentDataHandler.EXPECTED_COLUMNS['basic'] 
                          if col not in formatted_data.columns]
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                return None

            # Ensure numeric columns are float
            numeric_cols = ['sentiment_score', 'positive_score', 
                          'negative_score', 'neutral_score']
            for col in numeric_cols:
                formatted_data[col] = pd.to_numeric(formatted_data[col], 
                                                  errors='coerce')

            # Remove rows with NaN values
            formatted_data = formatted_data.dropna(subset=numeric_cols)

            return formatted_data

        except Exception as e:
            logger.error(f"Error formatting basic sentiment data: {str(e)}")
            return None

    @staticmethod
    def validate_and_format_aspect_sentiment(data):
        """Validate and format aspect-based sentiment analysis data."""
        try:
            if data is None or data.empty:
                logger.warning("Empty data provided for aspect sentiment validation")
                return None

            # Create a copy to avoid modifying original data
            formatted_data = data.copy()
            
            # Validate required columns
            base_cols = SentimentDataHandler.EXPECTED_COLUMNS['aspect']
            missing_cols = [col for col in base_cols if col not in formatted_data.columns]
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                return None

            # Identify aspect columns
            aspect_cols = [col for col in formatted_data.columns 
                         if col.endswith(('_sentiment', '_mentions'))]
            
            # Ensure numeric columns are float
            numeric_cols = [col for col in aspect_cols if col.endswith('_sentiment')]
            for col in numeric_cols:
                formatted_data[col] = pd.to_numeric(formatted_data[col], 
                                                  errors='coerce')

            # Convert mention counts to int
            count_cols = [col for col in aspect_cols if col.endswith('_mentions')]
            for col in count_cols:
                formatted_data[col] = pd.to_numeric(formatted_data[col], 
                                                  errors='coerce').astype('Int64')

            return formatted_data

        except Exception as e:
            logger.error(f"Error formatting aspect sentiment data: {str(e)}")
            return None

    @staticmethod
    def validate_and_format_temporal_sentiment(data):
        """Validate and format temporal sentiment analysis data."""
        try:
            if data is None or data.empty:
                logger.warning("Empty data provided for temporal sentiment validation")
                return None

            # Create a copy to avoid modifying original data
            formatted_data = data.copy()
            
            # Validate required columns
            missing_cols = [col for col in SentimentDataHandler.EXPECTED_COLUMNS['temporal'] 
                          if col not in formatted_data.columns]
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                return None

            # Ensure sentiment_score is float
            formatted_data['sentiment_score'] = pd.to_numeric(
                formatted_data['sentiment_score'], 
                errors='coerce'
            )

            # Remove rows with NaN values
            formatted_data = formatted_data.dropna(subset=['sentiment_score'])

            # Sort by document ID
            formatted_data = formatted_data.sort_values('doc_id')

            return formatted_data

        except Exception as e:
            logger.error(f"Error formatting temporal sentiment data: {str(e)}")
            return None

    @staticmethod
    def validate_and_format_comparative_sentiment(data):
        """Validate and format comparative sentiment analysis data."""
        try:
            if data is None or data.empty:
                logger.warning("Empty data provided for comparative sentiment validation")
                return None

            # Create a copy to avoid modifying original data
            formatted_data = data.copy()
            
            # Validate required columns
            missing_cols = [col for col in SentimentDataHandler.EXPECTED_COLUMNS['comparative'] 
                          if col not in formatted_data.columns]
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                return None

            # Ensure numeric columns are float
            numeric_cols = ['avg_sentiment', 'positive_ratio', 
                          'negative_ratio', 'document_count']
            for col in numeric_cols:
                formatted_data[col] = pd.to_numeric(formatted_data[col], 
                                                  errors='coerce')

            # Remove rows with NaN values
            formatted_data = formatted_data.dropna(subset=numeric_cols)

            return formatted_data

        except Exception as e:
            logger.error(f"Error formatting comparative sentiment data: {str(e)}")
            return None

    @staticmethod
    def calculate_summary_statistics(data):
        """Calculate summary statistics for sentiment analysis results."""
        try:
            if data is None or data.empty:
                return None

            stats = {
                'total_documents': len(data),
                'average_sentiment': data['sentiment_score'].mean(),
                'sentiment_std': data['sentiment_score'].std(),
                'positive_docs': (data['sentiment_score'] > 0).sum(),
                'negative_docs': (data['sentiment_score'] < 0).sum(),
                'neutral_docs': (data['sentiment_score'] == 0).sum(),
                'by_category': data.groupby('category')['sentiment_score'].agg([
                    'mean', 'std', 'count'
                ]).to_dict('index'),
                'by_language': data.groupby('language')['sentiment_score'].agg([
                    'mean', 'std', 'count'
                ]).to_dict('index')
            }

            return stats

        except Exception as e:
            logger.error(f"Error calculating summary statistics: {str(e)}")
            return None

    @staticmethod
    def calculate_aspect_summary(data):
        """Calculate summary statistics for aspect-based sentiment analysis."""
        try:
            if data is None or data.empty:
                return None

            # Get all aspect columns
            aspect_cols = [col for col in data.columns if col.endswith('_sentiment')]
            mention_cols = [col for col in data.columns if col.endswith('_mentions')]

            # Calculate statistics for each aspect
            aspect_stats = {}
            for aspect_col, mention_col in zip(aspect_cols, mention_cols):
                aspect_name = aspect_col.replace('_sentiment', '')
                aspect_stats[aspect_name] = {
                    'average_sentiment': data[aspect_col].mean(),
                    'sentiment_std': data[aspect_col].std(),
                    'total_mentions': data[mention_col].sum(),
                    'documents_with_mentions': (data[mention_col] > 0).sum(),
                    'by_category': data.groupby('category')[aspect_col].agg([
                        'mean', 'std', 'count'
                    ]).to_dict('index')
                }

            return aspect_stats

        except Exception as e:
            logger.error(f"Error calculating aspect summary: {str(e)}")
            return None
        
    @staticmethod
    def calculate_basic_summary(data):
        """Calculate summary statistics for basic sentiment analysis."""
        try:
            if data is None or data.empty:
                return None

            summary = {
                'total_documents': len(data),
                'average_sentiment': data['sentiment_score'].mean(),
                'sentiment_std': data['sentiment_score'].std(),
                'positive_docs': (data['sentiment_score'] > 0).sum(),
                'negative_docs': (data['sentiment_score'] < 0).sum(),
                'neutral_docs': (data['sentiment_score'] == 0).sum()
            }

            # Calculate percentages
            total = summary['total_documents']
            if total > 0:
                summary['positive_percentage'] = (summary['positive_docs'] / total) * 100
                summary['negative_percentage'] = (summary['negative_docs'] / total) * 100
                summary['neutral_percentage'] = (summary['neutral_docs'] / total) * 100

            # Add category-wise statistics if category column exists
            if 'category' in data.columns:
                summary['by_category'] = data.groupby('category')['sentiment_score'].agg([
                    'mean', 'std', 'count'
                ]).to_dict('index')

            return summary

        except Exception as e:
            logger.error(f"Error calculating basic summary statistics: {str(e)}")
            return None