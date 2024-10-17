from loguru import logger
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn3, venn2
import seaborn as sns
from collections import Counter
from sqlalchemy import text
from pathlib import Path

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
        """
        logger.info('Initializing KnowledgeType')
        if db_path is None:
            raise ValueError('db_path cannot be None')

        self.db_path = db_path
        self.db = Database(self.db_path)
        if self.db.engine is None:
            raise ValueError('Database connection is None')
    
    def all_docs(self):
        # Retrieve document content from the database
        query = text("SELECT knowledge_type FROM documents")
        with self.db.engine.connect() as connection:
            df = pd.read_sql(query, connection)
        
        # Count the frequency of each knowledge type
        df_original = df['knowledge_type'].value_counts().reset_index()
        df_original.columns = ['knowledge_type', 'count']

        # Calculate set sizes for Venn diagram
        venn_values = self._calculate_venn_values(df)

        # Create a figure with three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))

        # Create a table for the original dataframe
        ax1.axis('off')
        ax1.axis('tight')
        ax1.table(cellText=df_original.values, colLabels=df_original.columns, loc='center')
        ax1.set_title('Knowledge Type Distribution')

        # Plot the Venn diagram on the second subplot
        self._plot_venn_diagram(venn_values, ax2)

        # Create a bar chart for the third subplot
        self._plot_bar_chart(df_original, ax3)

        # Set the title of the entire figure
        fig.suptitle('Type de savoirs concernant les discrimination systémiques à Montréal', fontsize=16)

        # Adjust the layout, save the figure, and show it
        plt.tight_layout()
        plt.savefig('results/knowledge_type/knowledge_type_analysis.png')
        plt.show()

        # Save dataframes to CSV and XLSX files
        df_original.to_csv('results/knowledge_type/knowledge_type_distribution.csv', index=False)
        df_original.to_excel('results/knowledge_type/knowledge_type_distribution.xlsx', index=False)

    def _calculate_venn_values(self, df):
        # Split multi-value entries and count occurrences
        citoyen = set(df[df['knowledge_type'].str.contains('Citoyen', na=False)].index)
        communautaire = set(df[df['knowledge_type'].str.contains('Communautaire', na=False)].index)
        municipal = set(df[df['knowledge_type'].str.contains('Municipal', na=False)].index)

        # Calculate intersections
        c_com = len(citoyen.intersection(communautaire))
        c_m = len(citoyen.intersection(municipal))
        com_m = len(communautaire.intersection(municipal))
        c_com_m = len(citoyen.intersection(communautaire).intersection(municipal))

        # Calculate exclusive sets
        only_c = len(citoyen) - c_com - c_m + c_com_m
        only_com = len(communautaire) - c_com - com_m + c_com_m
        only_m = len(municipal) - c_m - com_m + c_com_m

        # Return the Venn diagram values
        return (only_c, only_com, only_m, c_com - c_com_m, c_m - c_com_m, com_m - c_com_m, c_com_m)

    def _plot_venn_diagram(self, venn_values, ax):
        venn3(subsets=venn_values, set_labels=('Citoyen', 'Communautaire', 'Municipal'), ax=ax)
        ax.set_title('Intersection of Knowledge Types')

    def _plot_bar_chart(self, df, ax):
        sns.barplot(x='knowledge_type', y='count', data=df, ax=ax, legend=False)
        ax.set_title('Knowledge Type Distribution')
        ax.set_xlabel('Knowledge Type')
        ax.set_ylabel('Count')
        plt.xticks(rotation=45, ha='right')

    def crosstable(self, compare_with=None):
        """
        Creates a cross table of knowledge types.

        Args:
            compare_with (str, optional): The column to compare the knowledge types with. Defaults to None.

        Raises:
            ValueError: If the dataframe is empty or the compare_with column does not exist.
        """
        # Retrieve document content from the database
        query = text("SELECT * FROM documents")
        with self.db.engine.connect() as connection:
            df = pd.read_sql(query, connection)
        
        # Check if the dataframe is empty
        if df.empty:
            logger.error('DataFrame is empty')
            raise ValueError('DataFrame is empty')

        # Split the knowledge_type column into separate rows
        df['knowledge_type'] = df['knowledge_type'].str.split(",")

        # Explode the knowledge_type column
        expanded_df = df.explode('knowledge_type')

        # Check if the compare_with column exists in the expanded dataframe
        if compare_with is not None and compare_with not in expanded_df.columns:
            logger.error(f'Compare with column {compare_with} does not exist in the dataframe')
            raise ValueError(f'Compare with column {compare_with} does not exist in the dataframe')

        # Create a cross table
        if compare_with is not None:
            # Create a cross table using the specified column to compare the knowledge types with
            crosstable = pd.crosstab(expanded_df[compare_with], expanded_df['knowledge_type'], margins=True, margins_name="Total")
        else:
            # Create a cross table using the knowledge types as the columns
            crosstable = pd.crosstab(expanded_df['knowledge_type'], columns='count', margins=True, margins_name="Total")

        # Save the cross table to a CSV and XLSX file
        crosstable.to_csv(f'results/knowledge_type/crosstable_knowledge_types_{compare_with}.csv', index=True)
        crosstable.to_excel(f'results/knowledge_type/crosstable_knowledge_types_{compare_with}.xlsx', index=True)

        # Visualize the crosstable
        plt.figure(figsize=(12, 8))
        sns.heatmap(crosstable.iloc[:-1, :-1], annot=True, cmap="YlGnBu", fmt='d')
        plt.title(f'Knowledge Type Distribution by {compare_with if compare_with else "Count"}')
        plt.tight_layout()
        plt.savefig(f'results/knowledge_type/crosstable_heatmap_{compare_with}.png')
        plt.close()

        # Print the cross table
        print(crosstable)

    def analyze_intersections(self):
        """
        Analyzes the intersections of different knowledge types.
        """
        query = text("SELECT knowledge_type FROM documents")
        with self.db.engine.connect() as connection:
            df = pd.read_sql(query, connection)

        # Split the knowledge types and create a set for each document
        knowledge_sets = df['knowledge_type'].str.split(',').apply(set)

        # Count the occurrences of each unique combination
        intersection_counts = Counter(frozenset(ks) for ks in knowledge_sets)

        # Prepare data for visualization
        labels = [', '.join(sorted(k)) for k in intersection_counts.keys()]
        sizes = list(intersection_counts.values())

        # Create a pie chart
        plt.figure(figsize=(12, 8))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('Intersections of Knowledge Types')
        plt.tight_layout()
        plt.savefig('results/knowledge_type/knowledge_type_intersections.png')
        plt.show()

        # Create a bar chart
        plt.figure(figsize=(12, 8))
        plt.bar(labels, sizes)
        plt.title('Intersections of Knowledge Types')
        plt.xlabel('Knowledge Type Combinations')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('results/knowledge_type/knowledge_type_intersections_bar.png')
        plt.show()

        # Save the intersection data
        intersection_df = pd.DataFrame({'Combination': labels, 'Count': sizes})
        intersection_df.to_csv('results/knowledge_type/knowledge_type_intersections.csv', index=False)
        intersection_df.to_excel('results/knowledge_type/knowledge_type_intersections.xlsx', index=False)

    def docs_list(self, knowledge_type):
        """
        Retrieve documents from the database based on the knowledge type and save them to a CSV and XLSX file.

        Args:
            knowledge_type (str): The knowledge type to filter the documents by.

        Raises:
            ValueError: If the dataframe is empty or no documents are found with the specified knowledge type.
        """
        # Retrieve document content from the database
        query = text("SELECT * FROM documents WHERE knowledge_type LIKE :kt")
        with self.db.engine.connect() as connection:
            df = pd.read_sql(query, connection, params={'kt': f'%{knowledge_type}%'})

        # Check if the dataframe is empty
        if df.empty:
            logger.warning(f'No documents found with knowledge type {knowledge_type}')
            raise ValueError(f'No documents found with knowledge type {knowledge_type}')

        # Save the filtered dataframe to a CSV and XLSX file
        df.to_csv(f'results/knowledge_type/docs_list_knowledge_type_{knowledge_type}.csv', index=False)
        df.to_excel(f'results/knowledge_type/docs_list_knowledge_type_{knowledge_type}.xlsx', index=False)

        # Notify the user
        print(f"Filtered dataframe saved to 'docs_list_knowledge_type_{knowledge_type}.csv' and 'docs_list_knowledge_type_{knowledge_type}.xlsx'")

    def __del__(self):
        self.db.engine.dispose()