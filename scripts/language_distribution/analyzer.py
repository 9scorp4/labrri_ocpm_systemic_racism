from loguru import logger
import pandas as pd
from langdetect import detect, DetectorFactory
from nltk.tokenize import sent_tokenize
import spacy
from collections import Counter, defaultdict
import multiprocessing
from functools import partial

from scripts.database import Database
from scripts.language_detector import LanguageDetector

class LanguageDistributionAnalyzer:
    def __init__(self, db_path):
        """Initialize analyzer with database connection and required models."""
        if db_path is None:
            raise ValueError('db_path cannot be None')

        self.db_path = db_path
        
        # Load spaCy models
        self.nlp = {
            'fr': spacy.load('fr_core_news_sm'),
            'en': spacy.load('en_core_web_sm')
        }
        
        # Set seed for reproducibility in language detection
        DetectorFactory.seed = 0
        
        logger.info("LanguageDistributionAnalyzer initialized successfully")

    def _get_db_connection(self):
        """Create a new database connection."""
        return Database(self.db_path)

    def get_language_counts(self, category="All categories"):
        """Get the count of documents by language for a given category."""
        logger.info(f'Analyzing language distribution for {category}')
        
        db = self._get_db_connection()
        
        try:
            if category == "All categories":
                query = """
                    SELECT 
                        CASE 
                            WHEN d.language = 'fr' THEN 'French'
                            WHEN d.language = 'en' THEN 'English'
                            ELSE 'Other'
                        END as Language,
                        COUNT(*) as Count,
                        COUNT(*) * 100.0 / SUM(COUNT(*)) OVER() as Percentage
                    FROM documents d
                    GROUP BY 
                        CASE 
                            WHEN d.language = 'fr' THEN 'French'
                            WHEN d.language = 'en' THEN 'English'
                            ELSE 'Other'
                        END
                    ORDER BY Count DESC
                """
                params = {}
            else:
                query = """
                    SELECT 
                        CASE 
                            WHEN d.language = 'fr' THEN 'French'
                            WHEN d.language = 'en' THEN 'English'
                            ELSE 'Other'
                        END as Language,
                        COUNT(*) as Count,
                        COUNT(*) * 100.0 / SUM(COUNT(*)) OVER() as Percentage
                    FROM documents d
                    WHERE d.category = :category
                    GROUP BY 
                        CASE 
                            WHEN d.language = 'fr' THEN 'French'
                            WHEN d.language = 'en' THEN 'English'
                            ELSE 'Other'
                        END
                    ORDER BY Count DESC
                """
                params = {'category': category}

            df = db.df_from_query(query, params)
            
            if df is None or df.empty:
                logger.warning(f"No data found for category: {category}")
                return pd.DataFrame(columns=['Language', 'Count', 'Percentage'])

            # Ensure column names are correct
            df.columns = ['Language', 'Count', 'Percentage']
            logger.debug(f"Query result:\n{df}")
            
            return df

        except Exception as e:
            logger.error(f"Error retrieving language counts: {str(e)}")
            return pd.DataFrame(columns=['Language', 'Count', 'Percentage'])
        finally:
            logger.info("Database connection closed")

    def get_distribution_by_category(self):
        """Get language distribution across all categories."""
        db = self._get_db_connection()
        query = """
            SELECT d.category, d.language, COUNT(*) as count
            FROM documents d
            GROUP BY d.category, d.language
            ORDER BY d.category, count DESC
        """
        
        df = db.df_from_query(query)
        if df is None or df.empty:
            logger.warning("No data found for category distribution")
            return None
            
        pivot_df = df.pivot(index='category', columns='language', values='count').fillna(0)
        return pivot_df
    
    def _process_single_document(self, content_tuple):
        """Process a single document for language analysis."""
        doc_id, organization, language, content, category, document_type = content_tuple
        
        detector = LanguageDetector(self.nlp['fr'], self.nlp['en'])
        
        lang_counts = {'fr': 0, 'en': 0, 'other': 0}
        other_samples = []
        total_chars = 0
        code_switches = 0
        previous_lang = None

        if content and isinstance(content, str):
            processed_doc = self.nlp['fr'](content) if language == 'fr' else self.nlp['en'](content)
            
            for sent in processed_doc.sents:
                sent_text = sent.text.strip()
                detected_lang, metadata = detector.detect_language(sent_text)
                
                if detected_lang:
                    chars = len(sent_text)
                    if detected_lang in ['fr', 'en']:
                        lang_counts[detected_lang] += chars
                    else:
                        lang_counts['other'] += chars
                        if len(other_samples) < 5:
                            other_samples.append(f"{sent_text[:50]}... ({detected_lang})")
                    
                    total_chars += chars
                    if previous_lang and detected_lang != previous_lang:
                        code_switches += 1
                    previous_lang = detected_lang

        # Calculate percentages
        total = sum(lang_counts.values()) or 1
        return [
            doc_id,
            organization,
            language,
            round(lang_counts['en'] / total * 100, 2),
            round(lang_counts['fr'] / total * 100, 2),
            round(lang_counts['other'] / total * 100, 2),
            code_switches,
            category,
            document_type,
            '; '.join(other_samples)
        ]

    def analyze_language_content(self, where="All languages"):
        """Analyze detailed language content distribution including code-switching."""
        logger.info(f'Analyzing language content for {where}')
        
        try:
            db = self._get_db_connection()
            query = """
                SELECT d.id, d.organization, d.language, c.content, d.category, d.document_type 
                FROM documents d 
                INNER JOIN content c ON d.id = c.doc_id 
                WHERE c.content IS NOT NULL AND LENGTH(TRIM(c.content)) > 0
                {}
            """.format("" if where == "All languages" else "AND d.language = :lang")

            params = {'lang': where} if where != "All languages" else {}
            
            df = db.df_from_query(query, params)
            if df is None or df.empty:
                logger.warning(f"No data found for {where}")
                return pd.DataFrame()  # Return empty DataFrame instead of None

            # Convert DataFrame rows to tuples for multiprocessing
            content_tuples = [tuple(x) for x in df.to_numpy()]
            
            logger.debug(f"Processing {len(content_tuples)} documents")
            
            # Process documents in parallel
            with multiprocessing.Pool() as pool:
                results = list(pool.map(self._process_single_document, content_tuples))

            # Create DataFrame from results
            columns = [
                'Document ID', 'Organization', 'Declared Language', 'English (%)',
                'French (%)', 'Other (%)', 'Code Switches', 'Category',
                'Document Type', 'Other Samples'
            ]
            
            result_df = pd.DataFrame(results, columns=columns)
            
            # Ensure numeric columns are properly typed
            numeric_cols = ['English (%)', 'French (%)', 'Other (%)', 'Code Switches']
            for col in numeric_cols:
                result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
            
            logger.debug(f"Result DataFrame shape: {result_df.shape}")
            logger.debug(f"Numeric columns info:\n{result_df[numeric_cols].describe()}")

            # Calculate and append overall averages
            avg_columns = ['English (%)', 'French (%)', 'Other (%)', 'Code Switches']
            overall_averages = result_df[avg_columns].mean()
            overall_averages = pd.DataFrame(overall_averages).T
            overall_averages['Document ID'] = 'Overall Average'
            overall_averages['Organization'] = ''
            overall_averages['Declared Language'] = ''
            overall_averages['Category'] = ''
            overall_averages['Document Type'] = ''
            overall_averages['Other Samples'] = ''

            result_df = pd.concat([result_df, overall_averages], ignore_index=True)
            return result_df

        except Exception as e:
            logger.error(f"Error in analyze_language_content: {str(e)}", exc_info=True)
            return pd.DataFrame()  # Return empty DataFrame instead of None
        finally:
            logger.info("Database connection closed")

    def get_code_switching_analysis(self, category=None):
        """Analyze code-switching patterns in documents."""
        try:
            base_df = self.analyze_language_content("All languages")
            logger.debug(f"Initial DataFrame shape: {base_df.shape}")
            
            if base_df.empty:
                logger.warning("No data available for analysis")
                return self._get_empty_analysis_result()

            # Remove the 'Overall Average' row
            base_df = base_df[base_df['Document ID'] != 'Overall Average']
            logger.debug(f"DataFrame shape after removing overall average: {base_df.shape}")
                
            # Only filter by category if it's not "All categories"
            if category and category != "All categories":
                base_df = base_df[base_df['Category'] == category]
                if base_df.empty:
                    logger.warning(f"No data available for category: {category}")
                    return self._get_empty_analysis_result()
                logger.debug(f"DataFrame shape after category filter: {base_df.shape}")

            # Ensure numeric columns are properly typed
            numeric_columns = ['English (%)', 'French (%)', 'Code Switches']
            for col in numeric_columns:
                base_df[col] = pd.to_numeric(base_df[col], errors='coerce')

            # Calculate correlations
            corr_matrix = base_df[numeric_columns].corr()
            logger.debug(f"Correlation matrix:\n{corr_matrix}")

            correlations = {
                'en_fr': float(corr_matrix.loc['English (%)', 'French (%)']),
                'switches_en': float(corr_matrix.loc['Code Switches', 'English (%)']),
                'switches_fr': float(corr_matrix.loc['Code Switches', 'French (%)'])
            }

            # Handle any NaN values
            correlations = {k: 0.0 if pd.isna(v) else v for k, v in correlations.items()}
            
            logger.debug(f"Calculated correlations: {correlations}")

            # Calculate statistics for code switching
            analysis = {
                'avg_switches': float(base_df['Code Switches'].mean()),
                'max_switches': int(base_df['Code Switches'].max()),
                'min_switches': int(base_df['Code Switches'].min()),
                'by_category': base_df.groupby('Category')['Code Switches'].mean().to_dict(),
                'correlation': correlations,
                'sample_size': len(base_df)  # Add sample size for reference
            }

            logger.debug(f"Final analysis result: {analysis}")
            return analysis

        except Exception as e:
            logger.error(f"Error in get_code_switching_analysis: {str(e)}", exc_info=True)
            return self._get_empty_analysis_result()

    def _get_empty_analysis_result(self):
        """Return an empty analysis result structure."""
        return {
            'avg_switches': 0,
            'max_switches': 0,
            'min_switches': 0,
            'by_category': {},
            'correlation': {'en_fr': 0, 'switches_en': 0, 'switches_fr': 0}
        }

    def get_category_summary(self):
        """Get summary statistics by category."""
        df = self.analyze_language_content("All languages")
        if df is None:
            return None

        summary = df.groupby('Category').agg({
            'English (%)': ['mean', 'std'],
            'French (%)': ['mean', 'std'],
            'Other (%)': ['mean', 'std'],
            'Code Switches': ['mean', 'max', 'count']
        }).round(2)

        return summary

    def _process_single_document(self, doc):
        """Process a single document for language analysis."""
        doc_id, organization, language, content, category, document_type = doc
        
        detector = LanguageDetector(self.nlp['fr'], self.nlp['en'])
        
        lang_counts = {'fr': 0, 'en': 0, 'other': 0}
        other_samples = []
        total_chars = 0
        code_switches = 0
        previous_lang = None

        if content and isinstance(content, str):
            processed_doc = self.nlp['fr'](content) if language == 'fr' else self.nlp['en'](content)
            
            for sent in processed_doc.sents:
                sent_text = sent.text.strip()
                detected_lang, metadata = detector.detect_language(sent_text)
                
                if detected_lang:
                    chars = len(sent_text)
                    if detected_lang in ['fr', 'en']:
                        lang_counts[detected_lang] += chars
                    else:
                        lang_counts['other'] += chars
                        if len(other_samples) < 5:
                            other_samples.append(f"{sent_text[:50]}... ({detected_lang})")
                    
                    total_chars += chars
                    if previous_lang and detected_lang != previous_lang:
                        code_switches += 1
                    previous_lang = detected_lang

        # Calculate percentages
        total = sum(lang_counts.values()) or 1
        return [
            doc_id,
            organization,
            language,
            round(lang_counts['en'] / total * 100, 2),
            round(lang_counts['fr'] / total * 100, 2),
            round(lang_counts['other'] / total * 100, 2),
            code_switches,
            category,
            document_type,
            '; '.join(other_samples)
        ]
