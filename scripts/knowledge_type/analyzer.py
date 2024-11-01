from scripts.database import Database
import pandas as pd
from loguru import logger

class KnowledgeTypeAnalyzer:
    def __init__(self, db_path):
        self.db = Database(db_path)
        self.knowledge_types = ['Citoyen', 'Municipal', 'Communautaire']
        
    def _split_knowledge_types(self, df, knowledge_type_col='knowledge_type'):
        """Split comma-separated knowledge types into separate rows."""
        # Create a new dataframe with split knowledge types
        split_df = df.assign(
            knowledge_type=df[knowledge_type_col].str.split(',')
        ).explode('knowledge_type')
        
        # Clean up the split values
        split_df['knowledge_type'] = split_df['knowledge_type'].str.strip()
        
        return split_df
        
    def get_distribution(self):
        """Get basic distribution of knowledge types, properly handling comma-separated values."""
        query = """
            SELECT 
                knowledge_type,
                COUNT(*) as doc_count
            FROM documents
            WHERE knowledge_type IS NOT NULL
            GROUP BY knowledge_type
        """
        
        try:
            # Get raw data
            df = self.db.df_from_query(query)
            if df is None or df.empty:
                logger.warning("No knowledge type data found")
                return None
                
            # Split comma-separated values
            split_df = self._split_knowledge_types(df)
            
            # Calculate new distribution
            distribution = split_df.groupby('knowledge_type').agg(
                count=('doc_count', 'sum')
            ).reset_index()
            
            # Calculate percentages
            total = distribution['count'].sum()
            distribution['percentage'] = (distribution['count'] / total * 100).round(2)
            
            # Sort by count descending
            distribution = distribution.sort_values('count', ascending=False)
            
            logger.info(f"Processed {len(df)} original entries into {len(distribution)} unique knowledge types")
            return distribution
            
        except Exception as e:
            logger.error(f"Error getting knowledge type distribution: {str(e)}")
            return None
    
    def get_intersection_data(self):
        """Get data for intersection analysis of the three main knowledge types."""
        query = """
            SELECT id, knowledge_type
            FROM documents
            WHERE knowledge_type IS NOT NULL
        """
        
        try:
            df = self.db.df_from_query(query)
            if df is None or df.empty:
                logger.warning("No data found for intersection analysis")
                return None
                
            # Split comma-separated values and handle NaN
            df['knowledge_type'] = df['knowledge_type'].fillna('').str.strip()
            
            # Create binary columns for each knowledge type
            result = pd.DataFrame(index=df['id'])
            for kt in self.knowledge_types:
                # Explicitly convert to boolean and fill NaN with False
                result[kt] = df['knowledge_type'].str.contains(
                    kt, 
                    case=False, 
                    na=False,
                    regex=False
                ).astype(bool)
            
            # Verify data integrity
            logger.debug(f"Data shape: {result.shape}")
            logger.debug(f"Data columns: {result.columns}")
            logger.debug(f"Data head: {result.head()}")
            logger.debug(f"Data info:\n{result.info()}")
            logger.debug(f"NaN check:\n{result.isna().sum()}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting intersection data: {str(e)}")
            return None
    
    def get_cross_analysis(self, category):
        """Get cross-analysis between knowledge types and another category."""
        query = """
            SELECT 
                knowledge_type,
                {category},
                COUNT(*) as count
            FROM documents
            WHERE knowledge_type IS NOT NULL
                AND {category} IS NOT NULL
            GROUP BY knowledge_type, {category}
        """.format(category=category)
        
        try:
            df = self.db.df_from_query(query)
            if df is None or df.empty:
                return None
                
            # Split and process knowledge types
            split_df = self._split_knowledge_types(df)
            
            # Recalculate counts after splitting
            cross_analysis = split_df.groupby(['knowledge_type', category])['count'].sum().reset_index()
            
            logger.info(f"Processed cross analysis between knowledge types and {category}")
            return cross_analysis
            
        except Exception as e:
            logger.error(f"Error getting cross analysis: {str(e)}")
            return None
    
    def get_combination_statistics(self):
        """Get statistics about knowledge type combinations."""
        query = """
            SELECT knowledge_type, COUNT(*) as count
            FROM documents
            WHERE knowledge_type IS NOT NULL
            GROUP BY knowledge_type
        """
        
        try:
            df = self.db.df_from_query(query)
            if df is None or df.empty:
                return None
            
            # Count different combinations
            combinations = pd.DataFrame({
                'combination': df['knowledge_type'],
                'count': df['count']
            })
            combinations = combinations.sort_values('count', ascending=False)
            
            # Calculate statistics
            stats = {
                'total_documents': combinations['count'].sum(),
                'single_type': combinations[~combinations['combination'].str.contains(',', na=False)]['count'].sum(),
                'multiple_types': combinations[combinations['combination'].str.contains(',', na=False)]['count'].sum(),
                'unique_combinations': len(combinations)
            }
            
            logger.info("Generated combination statistics")
            return combinations, stats
            
        except Exception as e:
            logger.error(f"Error getting combination statistics: {str(e)}")
            return None, None