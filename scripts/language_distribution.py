import logging
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from langdetect import detect
from nltk.tokenize import sent_tokenize

from scripts.database import Database

class LanguageDistributionChart:
    def __init__(self, db_path):
        """
        Initializes the LanguageDistributionChart class.

        Args:
            db_path (str): The path to the database file.

        Raises:
            ValueError: If db_path is None.
            FileNotFoundError: If the database file does not exist.

        Initializes the following instance variables:
            - db_path (str): The path to the database file.
            - db (Database): The database object.
            - conn (Connection): The database connection object.
            - cursor (Cursor): The database cursor object.
        """
        if db_path is None:
            raise ValueError('db_path cannot be None')

        self.db_path = db_path
        if not Path(self.db_path).exists():
            raise FileNotFoundError('Database file not found')

        self.db = Database(self.db_path)
        self.conn = self.db.conn
        if self.conn is None:
            raise ValueError('Database connection is None')

        self.cursor = self.conn.cursor()
        if self.cursor is None:
            raise ValueError('Database cursor is None')

    def count_graph(self, where):
        """
        Generates a count graph based on the specified category.

        Parameters:
            where (str): The category to generate the count graph for. It can be "All categories" or a specific category.

        Returns:
            None

        Raises:
            ValueError: If the retrieved data from the database is None.
            ValueError: If the category is not "All categories" or a specific category.

        This function retrieves data from the database based on the specified category. It then extracts the languages and counts from the retrieved data. Using this information, it creates a figure with two subplots: one for the pie chart and the other for the count table. The pie chart displays the counts of documents in each language, while the count table provides a tabular representation of the language counts. The first row of the table is bolded. The title of the entire figure is set based on the category. The layout is adjusted to reduce the gap between subplots. The resulting plot is saved in a PNG file and shown. Additionally, the results are saved in a CSV file.
        """
        logging.info('Retrieving data from the database')
        logging.info(f'Generating graph for {where}')

        if where == "All categories":
            query = "SELECT d.language, COUNT(d.id) FROM documents d GROUP by d.language"
        else:
            query = f"""
            SELECT d.language, COUNT(*) AS num_documents
            FROM documents d
            INNER JOIN content c ON d.id = c.doc_id
            WHERE d.category = '{where}'
            GROUP BY d.language
            """

        logging.info('Executing query')
        df = self.db.df_from_query(query)
        if df is None:
            raise ValueError('df is None')
        logging.info('Received data from the database')

        # Extract languages and counts from the retrieved data
        logging.info('Extracting languages and counts')
        languages = [entry[0] for entry in df.values]
        counts = [entry[1] for entry in df.values]

        # Create a figure with two subplots: one for the pie chart and the other for the count table
        logging.info('Creating figure')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Plot the pie chart
        logging.info('Plotting pie chart')
        ax1.pie(counts, labels=languages, autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')

        # Create a count table
        logging.info('Creating count table')
        ax2.axis('off')
        table_data = [["Language", "Count"]] + [[language, count] for language, count in zip(languages, counts)]
        table = ax2.table(cellText=table_data, loc='center', cellLoc='center')

        # Bold the first row of the table
        logging.info('Bolding first row of the table')
        for cell in table.get_celld().values():
            if cell.get_text().get_text() == "Language" or cell.get_text().get_text() == "Count":
                cell.get_text().set_fontweight('bold')

        # Set the title of the entire figure
        if where == "All categories":
            fig.suptitle("Nombre de documents par langue")
        else:
            fig.suptitle(f"Nombre de documents dans la catégorie {where} par langue")

        # Adjust layout to reduce gap between subplots
        logging.info('Adjusting layout')
        plt.subplots_adjust(wspace=0.1)

        # Save the results in a PNG file and show the plot
        logging.info('Saving and showing plot')
        plt.savefig(f'results/language_distribution/{where}_count.png')
        plt.show()

        # Save the results in a CSV file
        logging.info('Saving results in a CSV file')
        df.to_csv(f'results/language_distribution/{where}_count.csv', index=False)

    def language_percentage_distribution(self, where):
        """
        Calculates the percentage distribution of English and French sentences in the given documents.
        
        Args:
            where (str): The language to filter the documents by. If "All languages" is provided, all documents will be considered.
        
        Raises:
            ValueError: If the retrieved dataframe is None.
        
        Returns:
            None
        """
        # Retrieve data from the database
        logging.info('Retrieving data from the database')
        if where == "All languages":
            query = "SELECT d.organization, d.language, c.content FROM documents d INNER JOIN content c ON d.id = c.doc_id"
        else:
            query = f"SELECT d.organization, d.language, c.content FROM documents d INNER JOIN content c ON d.id = c.doc_id WHERE d.language = '{where}'"
        df = self.db.df_from_query(query)
        if df is None:
            raise ValueError('df is None')
        
        # Initialize lists to store data for the table
        rows = []

        for _, row in df.iterrows():
            doc_name = row['organization']
            doc_language = row['language']
            doc_text = row['content']

            # Tokenize the text
            sentences = sent_tokenize(doc_text)

            # Intialize counters
            en_count = 0
            fr_count = 0

            # Classify each sentence individually
            for sentence in sentences:
                # Check if the sentence contains sufficient length for language detection
                if len(sentence.strip()) > 18:
                    try:
                        language = detect(sentence)
                        if language == 'en':
                            en_count += 1
                        elif language == 'fr':
                            fr_count += 1
                    except Exception as e:
                        logging.error(f"Error: {e}")
                        raise ValueError(f"Error: {e}")
            
            # Append the data to the rows list
            rows.append([doc_name, doc_language, en_count, fr_count])
        
        # Create dataframe
        headers = ["Organization", "Language", "English Sentences", "French Sentences"]
        result_df = pd.DataFrame(rows, columns=headers)

        # Calculate percentages
        total_sentences = result_df["English Sentences"] + result_df["French Sentences"]
        result_df['English (%)'] = (result_df["English Sentences"] / total_sentences) * 100
        result_df['French (%)'] = (result_df["French Sentences"] / total_sentences) * 100

        # Calculate total counts for English and French sentences across all documents
        total_en_count = result_df["English Sentences"].sum()
        total_fr_count = result_df["French Sentences"].sum()
        
        # Calculate overall percentages
        overall_en_percentage = (total_en_count / (total_en_count + total_fr_count)) * 100
        overall_fr_percentage = (total_fr_count / (total_en_count + total_fr_count)) * 100
        
        # Add overall averages to the table
        result_df.loc[len(result_df)] = ["", "", "", "Overall Average", overall_en_percentage, overall_fr_percentage]
        
        # Save the results in a CSV file
        logging.info('Saving results in a CSV file')
        result_df.to_csv(f'results/language_distribution/{where}_percentage.csv', index=False)

        # Save the results in an Excel file
        logging.info('Saving results inb an Excel file')
        result_df.to_excel(f'results/language_distribution/{where}_percentage.xlsx', index=False)

        # Show the results
        logging.info('Showing results')
        print(result_df)
