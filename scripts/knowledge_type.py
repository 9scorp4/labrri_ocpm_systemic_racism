import logging
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib_venn import venn3

from scripts.database import Database

class KnowledgeType:
    def __init__(self, db_path):
        """
        Initializes the KnowledgeType class.

        Args:
            db_path (str): The path to the database file.

        Raises:
            ValueError: If db_path is None.
            ValueError: If the database connection is None.

        Initializes the following instance variables:
            - db_path (str): The path to the database file.
            - db (Database): The database object.
            - conn (Connection): The database connection object.
        """
        logging.info('Initializing KnowledgeType')
        if db_path is None:
            raise ValueError('db_path cannot be None')
        self.db_path = db_path
        self.db = Database(self.db_path)
        self.conn = self.db.conn
        if self.conn is None:
            raise ValueError('Database connection is None')

    def df_from_query(self, query):
        """
        Retrieves a pandas DataFrame from a SQL query.

        Parameters:
            query (str): The SQL query to execute.

        Raises:
            ValueError: If the query is None.
            ValueError: If the database connection is None.
            ValueError: If the retrieved content_df is None.

        Returns:
            pandas.DataFrame: The DataFrame containing the results of the query.
        """
        if query is None:
            raise ValueError('query cannot be None')
        if self.conn is None:
            raise ValueError('Database connection is None')
        logging.info('Retrieving data from the database')
        content_df = pd.read_sql_query(query, self.conn)
        if content_df is None:
            raise ValueError('content_df is None')
        return content_df
    
    def all_docs(self):
        # Retrieve document content from the database
        df = self.df_from_query("SELECT knowledge_type FROM documents")
        
        # Count the frequency of each knowledge type
        df_original = df['knowledge_type'].value_counts().reset_index()
        df_original.columns = ['knowledge_type', 'count']

        # Split the knowledge types values by comma and count the frequency
        df_split = df['knowledge_type'].str.split(',', expand=True).stack().reset_index(level=1, drop=True)
        df_split = df_split.value_counts().reset_index()
        df_split.columns = ['knowledge_type', 'count']

        # Define the set sizes and overlaps from the df_original dataframe for the venn diagram
        # Define the set sizes and overlaps from the df_original dataframe for the venn diagram
        venn_values = {}
        if not df_original.empty:
            venn_values = {
                '100': df_original.loc[df_original['knowledge_type'] == 'Citoyen', 'count'].iloc[0] if not df_original.loc[df_original['knowledge_type'] == 'Citoyen', 'count'].empty else 0,
                '010': df_original.loc[df_original['knowledge_type'] == 'Communautaire', 'count'].iloc[0] if not df_original.loc[df_original['knowledge_type'] == 'Communautaire', 'count'].empty else 0,
                '001': df_original.loc[df_original['knowledge_type'] == 'Municipal', 'count'].iloc[0] if not df_original.loc[df_original['knowledge_type'] == 'Municipal', 'count'].empty else 0,
                '110': df_original.loc[df_original['knowledge_type'] == 'Communautaire,Citoyen', 'count'].iloc[0] if not df_original.loc[df_original['knowledge_type'] == 'Communautaire,Citoyen', 'count'].empty else 0,
                '101': df_original.loc[df_original['knowledge_type'] == 'Citoyen,Municipal', 'count'].iloc[0] if not df_original.loc[df_original['knowledge_type'] == 'Citoyen,Municipal', 'count'].empty else 0,
                '011': df_original.loc[df_original['knowledge_type'] == 'Communautaire,Municipal', 'count'].iloc[0] if not df_original.loc[df_original['knowledge_type'] == 'Communautaire,Municipal', 'count'].empty else 0,
                '111': df_original.loc[df_original['knowledge_type'] == 'Citoyen,Communautaire,Municipal', 'count'].iloc[0] if not df_original.loc[df_original['knowledge_type'] == 'Citoyen,Communautaire,Municipal', 'count'].empty else 0
            }

        # Create a figure with two subplots, one for the split dataframe and one for the venn diagram
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # Create a table for the split dataframe
        ax1.axis('off')
        ax1.axis('tight')
        ax1.table(cellText=df_split.values, colLabels=df_split.columns, loc='center')

        # Plot the venn diagram on the second subplot
        venn3(subsets=venn_values, set_labels=('Citoyen', 'Communautaire', 'Municipal'), ax=ax2)

        # Set the title of the entire figure
        fig.suptitle('Types de savoirs concernant les discriminations systémiques à Montréal', fontsize=16)

        # Adjust the layout
        plt.tight_layout()

        # Save the plot to a PNG file
        plt.savefig('results/knowledge_type/knowledge_types.png')

        # Show the plot
        plt.show()

        # Save both dataframes to CSV and XLSX files
        df_original.to_csv('results/knowledge_type/knowledge_types_original.csv', index=False)
        df_split.to_csv('results/knowledge_type/knowledge_types_split.csv', index=False)
        df_original.to_excel('results/knowledge_type/knowledge_types_original.xlsx', index=False)
        df_split.to_excel('results/knowledge_type/knowledge_types_split.xlsx', index=False)

    def crosstable(self, compare_with=None):
        """
        Creates a cross table of knowledge types.

        Args:
            compare_with (str, optional): The column to compare the knowledge types with. Defaults to None.

        Raises:
            ValueError: If the dataframe is empty or the compare_with column does not exist.
        """
        # Retrieve document content from the database
        df = self.df_from_query("SELECT * FROM documents")
        
        # Check if the dataframe is empty
        if df.empty:
            logging.error('DataFrame is empty')
            raise ValueError('DataFrame is empty')

        # Split the knowledge_type column into separate rows
        df['knowledge_type'] = df['knowledge_type'].str.split(",")

        # Explode the knowledge_type column
        expanded_df = df.explode('knowledge_type')

        # Check if the compare_with column exists in the expanded dataframe
        if compare_with is not None and compare_with not in expanded_df.columns:
            logging.error(f'Compare with column {compare_with} does not exist in the dataframe')
            raise ValueError(f'Compare with column {compare_with} does not exist in the dataframe')

        # Create a cross table
        if compare_with is not None:
            # Create a cross table using the specified column to compare the knowledge types with
            crosstable = pd.crosstab(expanded_df[compare_with], expanded_df['knowledge_type'], margins=True, margins_name="Total")
        else:
            # Create a cross table using the knowledge types as the columns
            crosstable = pd.crosstab(expanded_df['knowledge_type'], margins=True, margins_name="Total")

        # Save the cross table to a CSV and XLSX file
        crosstable.to_csv(f'results/knowledge_type/crosstable_knowledge_types_{compare_with}.csv', index=True)
        crosstable.to_excel(f'results/knowledge_type/crosstable_knowledge_types_{compare_with}.xlsx', index=True)

        # Print the cross table
        print(crosstable)

    def docs_list(self, knowledge_type):
        """
        Retrieve documents from the database based on the knowledge type and save them to a CSV and XLSX file.

        Args:
            knowledge_type (str): The knowledge type to filter the documents by.

        Raises:
            ValueError: If the dataframe is empty or no documents are found with the specified knowledge type.
        """
        # Retrieve document content from the database
        df = self.df_from_query("SELECT * FROM documents")

        # Check if the dataframe is empty
        if df.empty:
            logging.error('DataFrame is empty')
            raise ValueError('DataFrame is empty')

        # Filter the dataframe based on the knowledge type
        filtered_df = df[df['knowledge_type'] == knowledge_type]

        # Check if there are any filtered documents
        if filtered_df.empty:
            logging.warning(f'No documents found with knowledge type {knowledge_type}')
            raise ValueError(f'No documents found with knowledge type {knowledge_type}')

        # Save the filtered dataframe to a CSV and XLSX file
        filtered_df.to_csv(f'results/knowledge_type/docs_list_knowledge_type_{knowledge_type}.csv', index=False)
        filtered_df.to_excel(f'results/knowledge_type/docs_list_knowledge_type_{knowledge_type}.xlsx', index=False)

        # Notify the user
        print(f"Filtered dataframe saved to 'doc_list_knowledge_type_{knowledge_type}.csv' and 'docs_list_knowledge_type_{knowledge_type}.xlsx'")

    def __del__(self):
        self.conn.close()