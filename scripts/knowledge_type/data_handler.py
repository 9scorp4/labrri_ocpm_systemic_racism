from loguru import logger
import pandas as pd

class KnowledgeTypeDataHandler:
    """Handles data processing and validation between analyzer and visualizer components."""
    
    EXPECTED_COLUMNS = {
        'basic': ['knowledge_type', 'count', 'percentage'],
        'detailed': ['knowledge_type', 'document_count', 'category'],
        'intersection': ['Citoyen', 'Municipal', 'Communautaire']
    }

    @staticmethod
    def validate_boolean_data(data):
        """
        Validate and clean boolean data for Venn diagram visualization.
        
        Args:
            data (pd.DataFrame): DataFrame containing boolean columns for each knowledge type
            
        Returns:
            pd.DataFrame or None: Cleaned DataFrame with proper boolean values or None if validation fails
        """
        try:
            if data is None or data.empty:
                logger.warning("Empty data provided for boolean validation")
                return None

            # Verify expected columns exist
            expected_columns = KnowledgeTypeDataHandler.EXPECTED_COLUMNS['intersection']
            missing_columns = [col for col in expected_columns if col not in data.columns]
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return None

            # Create a copy to avoid modifying original data
            clean_data = data.copy()

            # Ensure all required columns are boolean
            for col in expected_columns:
                # First fill NaN values with False
                clean_data[col] = clean_data[col].fillna(False)
                # Convert to boolean type
                clean_data[col] = clean_data[col].astype(bool)

            # Verify data integrity
            logger.debug(f"Cleaned data shape: {clean_data.shape}")
            logger.debug(f"Cleaned data columns: {clean_data.columns}")
            logger.debug(f"Cleaned data info:\n{clean_data.info()}")
            logger.debug(f"NaN check:\n{clean_data.isna().sum()}")

            return clean_data

        except Exception as e:
            logger.error(f"Error validating boolean data: {str(e)}")
            return None

    @staticmethod
    def validate_and_format_intersection_data(data):
        """
        Validate and format data for intersection analysis visualization.
        
        Args:
            data (pd.DataFrame): Raw intersection data from analyzer
            
        Returns:
            pd.DataFrame or None: Formatted data ready for visualization
        """
        try:
            # First apply boolean validation
            clean_data = KnowledgeTypeDataHandler.validate_boolean_data(data)
            if clean_data is None:
                return None

            # Calculate intersection statistics
            stats = {
                'total_documents': len(clean_data),
                'by_type': {
                    'Citoyen': clean_data['Citoyen'].sum(),
                    'Municipal': clean_data['Municipal'].sum(),
                    'Communautaire': clean_data['Communautaire'].sum()
                },
                'multiple_types': (clean_data[['Citoyen', 'Municipal', 'Communautaire']]
                                 .sum(axis=1) > 1).sum()
            }

            logger.info(f"Intersection statistics: {stats}")
            return clean_data, stats

        except Exception as e:
            logger.error(f"Error formatting intersection data: {str(e)}")
            return None, None

    @staticmethod
    def validate_and_format_distribution(data):
        """
        Validate and format distribution data for visualization.
        
        Args:
            data (pd.DataFrame): Raw distribution data from analyzer
            
        Returns:
            pd.DataFrame or None: Formatted data ready for visualization
        """
        try:
            if data is None or data.empty:
                logger.warning("Empty data provided for distribution validation")
                return None

            # Verify required columns
            required_columns = KnowledgeTypeDataHandler.EXPECTED_COLUMNS['basic']
            if not all(col in data.columns for col in required_columns):
                logger.error(f"Missing required columns. Found: {data.columns}")
                return None

            # Create a copy to avoid modifying original data
            formatted_data = data.copy()

            # Ensure numeric columns are float
            numeric_cols = ['count', 'percentage']
            for col in numeric_cols:
                formatted_data[col] = pd.to_numeric(formatted_data[col], errors='coerce')

            # Remove any rows with NaN values
            formatted_data = formatted_data.dropna()

            # Sort by count in descending order
            formatted_data = formatted_data.sort_values('count', ascending=False)

            logger.debug(f"Formatted distribution data:\n{formatted_data}")
            return formatted_data

        except Exception as e:
            logger.error(f"Error formatting distribution data: {str(e)}")
            return None

    @staticmethod
    def format_cross_analysis_data(data, category):
        """
        Format data for cross-analysis visualization.
        
        Args:
            data (pd.DataFrame): Raw cross-analysis data from analyzer
            category (str): Category used for cross-analysis
            
        Returns:
            dict or None: Formatted data ready for visualization
        """
        try:
            if data is None or data.empty:
                return None

            # Create pivot table for visualization
            pivot_data = pd.pivot_table(
                data,
                values='count',
                index='knowledge_type',
                columns=category,
                fill_value=0
            )

            # Calculate percentages
            percentage_data = pivot_data.div(pivot_data.sum(axis=0), axis=1) * 100

            # Calculate statistics
            stats = {
                'total_documents': data['count'].sum(),
                'unique_knowledge_types': len(data['knowledge_type'].unique()),
                'unique_categories': len(data[category].unique())
            }

            return {
                'counts': pivot_data,
                'percentages': percentage_data,
                'stats': stats
            }

        except Exception as e:
            logger.error(f"Error formatting cross-analysis data: {str(e)}")
            return None