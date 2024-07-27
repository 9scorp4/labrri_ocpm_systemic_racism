import logging
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
from scripts.database import Database

class WordFrequencyChart:
    def __init__(self, db_path):
        """
        Initializes the WordFrequencyChart class.

        Args:
            db_path (str): The path to the database file.

        Raises:
            ValueError: If db_path is None.
            ValueError: If the database connection is None.
            FileNotFoundError: If the stopwords cannot be downloaded.

        Initializes the following instance variables:
            - db_path (str): The path to the database file.
            - db (Database): The database object.
            - conn (Connection): The database connection object.
            - en_stopwords (set): The set of English stopwords.
            - fr_stopwords (set): The set of French stopwords.
            - custom_stopwords (set): The set of custom stopwords.
            - all_stopwords (set): The set of all stopwords.

        Returns:
            None
        """
        logging.info('Initializing WordFrequencyChart')
        if db_path is None:
            raise ValueError('db_path cannot be None')

        self.db_path = db_path
        self.db = Database(self.db_path)
        self.conn = self.db.conn
        if self.conn is None:
            raise ValueError('Database connection is None')

        try:
            nltk.download('stopwords')
        except LookupError:
            raise FileNotFoundError('Stopwords cannot be downloaded')

        self.en_stopwords = set(stopwords.words('english'))
        self.fr_stopwords = set(stopwords.words('french'))

        self.custom_stopwords = {
            'entre', 'ainsi', 'leurs', 'beaucoup', 'souvent', 'dire', 'plus', 'cette', 'fait', 'faire', 'donc',
            'aussi', 'parce', 'peut', 'avoir', 'autres', 'sténo', 'tout', 'alors', 'vraiment', 'bien', 'être',
            'quand', 'puis', 'très', 'faut', 'comme', 'ariane', 'émond', 'a', 'plus', 'comme', 'cette', 'ça', 'fait',
            'être', 'faire', 'mme', 'donc', 'aussi', 'autres', 'si', 'entre', 'bien', 'tout', 'g', 'peut', 'leurs',
            'o', 'gh', 'avoir', 'non', 'the', 'de', 'la', 'et', 'des', 'le', 'les', 'l', 'may', 'would', 'also', 'see',
            'one', 'http', 'à', 'du', 'like', 'coprésidente', 'well', 'non', 'think', 'see', 'xurukulasuriya', 'dexter',
            'plus', 'aussi', 'très', 'get', 'mme', 'novembre', 'séance', 'sténo', 'mmm', 'commissaire', 'coprésidente',
            'know', 'sarah', 'soirée', 'go', 'oui', 'holness', 'ça', 'émond', 'thierry', 'thuot', 'lindor', 'merci',
            'would', 'balarama', 'ariane', 'like', 'lot', 'donc', 'fait', 'si', 'comme', 'judy', 'ouellet', 'one',
            'years', 'parce', 'going', 'pinet', 'monsieur', 'avoir', 'dit'
        }

        self.all_stopwords = self.en_stopwords.union(self.fr_stopwords, self.custom_stopwords)
    
    def top_20_words_category(self, where: str) -> None:
        """
        Retrieves the top 20 most frequent words from the documents in a specified category, tokenizes the text, converts the words to lowercase, removes stop words, and plots the most frequent words.
        
        Parameters:
            self (WordFrequencyChart): The WordFrequencyChart instance.
        
        Returns:
            None
        
        Raises:
            ValueError: If the dataframe retrieved from the database is None or if any content in the dataframe is None.
            TypeError: If the input parameter where is not of type str.
        """
        if not isinstance(where, str):
            raise TypeError(f"Expected a string, got {type(where).__name__}")

        logging.info('Tokenizing and removing stopwords')
        # Retrieve document content from the database
        df = self.db.df_from_query(f"SELECT c.content FROM content c JOIN documents d ON c.doc_id = d.id WHERE category = '{where}'")
        if df is None:
            raise ValueError("df is None")

        # Check if any content is None
        if df['content'].isnull().any().any():
            raise ValueError("content contains None values")

        # Tokenize the text, convert to lowercase, and remove stop words
        word_freq = Counter()
        for content in df['content']:
            tokens = word_tokenize(content)
            words = [word.lower() for word in tokens if word.isalpha() and len(word) >= 4 and word.lower() not in self.all_stopwords]
            word_freq.update(words)

        # Get the 20 most common words and their frequencies
        most_common_words = word_freq.most_common(20)

        # Plot the most frequent words and save the results in a PNG file
        logging.info('Plotting the most frequent words')
        words, frequencies = zip(*most_common_words)
        plt.figure(figsize=(10, 6))
        plt.bar(words, frequencies)
        plt.title(f"Les 20 mots les plus fréquents parmi les documents de la catégorie {where}")
        plt.xlabel("Mots")
        plt.ylabel("Frequence")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'results/word_frequency/top_20_{where}.png')
        plt.show()

        # Save the results in a CSV file
        logging.info('Saving the results')
        df = pd.DataFrame(most_common_words, columns=['Word', 'Frequency'])
        df.to_csv(f'results/word_frequency/top_20_{where}.csv', index=False)

    def top_20_words_lang(self, lang):
        """
        Retrieves the text content of all documents in a specified language from the database, tokenizes the text, removes stopwords, counts the frequency of each word, extracts the 20 most common words, plots the word frequencies, and saves the results in a CSV file and a PNG file.

        Args:
            lang (str): The language of the documents to retrieve.

        Raises:
            ValueError: If the dataframe retrieved from the database is None or if any content in the dataframe is None.
        """
        logging.info('Tokenizing and removing stopwords')
        # Retrieve text content of all documents in the specified language
        query = f"""
        SELECT c.content
        FROM documents d
        INNER JOIN content C ON d.id = c.doc_id
        WHERE d.language = '{lang}'
        """
        df = self.db.df_from_query(query)
        if df is None:
            raise ValueError('df is None')

        # Check if any content is None
        if df['content'].isnull().any().any():
            raise ValueError('content contains None values')

        # Tokenize the text, convert to lowercase, and remove stop words
        all_words = []
        for content in df['content']:
            if content is None:
                raise ValueError('content is None')
            tokens = word_tokenize(content)
            words = [word.lower() for word in tokens if word.isalpha() and word.lower() not in self.all_stopwords]
            all_words.extend(words)

        # Count the frequency of each word
        word_freq = Counter(all_words)

        # Get the 20 most common words and their frequencies
        most_common_words = word_freq.most_common(20)

        # Extract word and frequency data for plotting
        words, frequencies = zip(*most_common_words)

        # Plot the most frequent words
        logging.info(f'Plotting the most frequent words in {lang} corpus')
        plt.figure(figsize=(10, 6))
        plt.bar(words, frequencies)
        plt.title(f"Les 20 mots les plus fréquents dans le corpus (lang = {lang})")
        plt.xlabel("Mots")
        plt.ylabel("Frequence")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'results/word_frequency/top_20_{lang}.png')
        plt.show()

        # Save the results in a CSV file
        logging.info('Saving the results')
        df = pd.DataFrame(most_common_words, columns=['Word', 'Frequency'])
        df.to_csv(f'results/word_frequency/top_20_{lang}.csv', index=False)

    def frequency_certain_words(self, words):
        """
        Retrieves the document content from the database, combines it into a single string, counts the frequency of each word in the specified list of words, creates a bar plot of the word frequencies, and saves the results in a CSV file and a PNG file.

        Args:
            words (list): A list of words to count the frequency of.

        Raises:
            ValueError: If the dataframe retrieved from the database is None or if any content in the dataframe is None.
        """
        # Retrieve document content from the database
        logging.info('Retrieving document content from the database')
        df = self.db.df_from_query("SELECT content FROM content")
        if df is None:
            logging.error('df is None')
            raise ValueError('df is None')

        # Check if any content is None
        if df['content'].isnull().any().any():
            raise ValueError('content contains None values')

        # Combine the content into a single string
        logging.info('Combining the content into a single string')
        content = ' '.join(df['content'])

        # Count the frequency of each word
        logging.info('Counting the frequency of each word')
        word_freq = {word: content.count(word) for word in words}

        # Create a bar plot
        logging.info('Creating a bar plot')
        plt.figure(figsize=(10, 6))
        plt.bar(word_freq.keys(), word_freq.values())
        plt.title("Frequence des mots sélectionnés")
        plt.xlabel("Mots")
        plt.ylabel("Frequence")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('results/word_frequency/frequency_certain_words.png')
        plt.show()

        # Save the results in a CSV file
        logging.info('Saving the results')
        df = pd.DataFrame(list(word_freq.items()), columns=['Word', 'Frequency'])
        df.to_csv('results/word_frequency/frequency_certain_words.csv', index=False)

    def __del__(self):
        """
        Closes the database connection if it is not None.
        """
        logging.info('Closing the database connection')
        if self.conn is not None:
            self.conn.close()
