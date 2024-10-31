from loguru import logger
import pandas as pd

class LanguageDistributionDataHandler:
    """Handles data processing and validation between analyzer and visualizer components."""
    
    EXPECTED_COLUMNS = {
        'basic': ['Language', 'Count', 'Percentage'],
        'detailed': ['Document ID', 'Organization', 'Declared Language', 
                    'English (%)', 'French (%)', 'Other (%)', 
                    'Code Switches', 'Category', 'Document Type', 'Other Samples'],
        'comparison': ['Category', 'French (%)', 'English (%)', 'Other (%)']
    }

    @staticmethod
    def validate_and_format_basic_distribution(data):
        """Validate and format data for basic distribution visualization."""
        try:
            if data is None or data.empty:
                logger.warning("Empty data received for basic distribution")
                return None

            # Create a copy to avoid modifying original data
            formatted_data = data.copy()
            
            # Ensure required columns exist
            required_columns = ['Language', 'Count', 'Percentage']
            if not all(col in formatted_data.columns for col in required_columns):
                logger.warning(f"Missing required columns. Found: {formatted_data.columns}")
                return None
            
            # Ensure numeric columns are float
            formatted_data['Count'] = pd.to_numeric(formatted_data['Count'], errors='coerce')
            formatted_data['Percentage'] = pd.to_numeric(formatted_data['Percentage'], errors='coerce')
            
            # Remove any rows with NaN values
            formatted_data = formatted_data.dropna()
            
            # Sort by count in descending order
            formatted_data = formatted_data.sort_values('Count', ascending=False)
            
            logger.debug(f"Formatted data:\n{formatted_data}")
            return formatted_data[required_columns]

        except Exception as e:
            logger.error(f"Error formatting basic distribution data: {str(e)}")
            return None

    @staticmethod
    def validate_and_format_comparison(data):
        """Validate and format data for category comparison visualization."""
        try:
            if data is None or data.empty:
                return None

            # Ensure we have the required columns
            required_columns = ['Category', 'French (%)', 'English (%)', 'Other (%)']
            
            # Handle MultiIndex DataFrames
            if isinstance(data.columns, pd.MultiIndex):
                # Extract mean values for language percentages
                formatted_data = pd.DataFrame({
                    'Category': data.index,
                    'French (%)': data[('French (%)', 'mean')],
                    'English (%)': data[('English (%)', 'mean')],
                    'Other (%)': data[('Other (%)', 'mean')]
                })
            else:
                formatted_data = data.copy()

            # Validate columns
            missing_cols = [col for col in required_columns if col not in formatted_data.columns]
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                return None

            # Remove empty categories and reset index
            formatted_data = formatted_data[
                formatted_data['Category'].notna() & (formatted_data['Category'] != '')
            ].reset_index(drop=True)

            return formatted_data[required_columns]

        except Exception as e:
            logger.error(f"Error formatting comparison data: {str(e)}")
            return None

    @staticmethod
    def process_detailed_analysis(data):
        """Process and validate data for detailed analysis visualization."""
        try:
            if data is None or data.empty:
                return None

            # Ensure numeric columns are properly typed
            numeric_columns = ['English (%)', 'French (%)', 'Other (%)', 'Code Switches']
            for col in numeric_columns:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')

            # Remove 'Overall Average' row for visualization
            if 'Document ID' in data.columns:
                data = data[data['Document ID'] != 'Overall Average']

            # Validate required columns
            required_cols = set(numeric_columns)
            if not required_cols.issubset(set(data.columns)):
                missing = required_cols - set(data.columns)
                logger.error(f"Missing required columns: {missing}")
                return None

            return data

        except Exception as e:
            logger.error(f"Error processing detailed analysis data: {str(e)}")
            return None