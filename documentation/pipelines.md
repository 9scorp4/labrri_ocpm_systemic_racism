# Pipeline blueprints

This document provides an overview of the different pipelines used in this project. Each pipeline corresponds to a specific analysis task and is associated with one or more Python scripts or Jupyter notebooks.

## Table of contents

- [1. Update Database Pipeline](#1-update-database-pipeline)
- [2. General Analysis](#2-general-analysis)
- [3. Word Frequency Analysis](#3-word-frequency-analysis)
- [4. Language Distribution Analysis](#4-language-distribution-analysis)
- [5. Knowledge Type Analysis](#5-knowledge-type-analysis)
- [6. Topic Analysis](#6-topic-analysis-in-development)
- [7. Sentiment Analysis](#7-sentiment-analysis-in-development)
- [Notes](#notes)

## 1. Update Database Pipeline

**Main Script:** update_database.py

This pipeline is responsible with populating the database with fetched data from the OCPM documentation.

### Key Components

- *Database* class from scripts.database
    - Uses *csv_to_xlsx* method from class
- *ProcessDocuments* class from scripts.document_processing
    - Uses *process_documents* method from class

### Process

1. Initializes logging
2. Reads data from '*data/pdf_list.csv*'
3. Converts CSV to XLSX for handling in Nvivo software (not in this repo)
4. Processs PDF documents using *process_documents* method

## 2. General Analysis

**Main Notebook:** general_analysis.ipynb

This pipeline performs a general analysis on the connected data

### Key Components

Two main analyses:
1. **Organization category count:** How many participant organizations there are by category.
2. **Crosstables:** Cross-referencing of data columns.
    - Organization Category and Document Type
    - Organization Category and Clientele

### Process

1. Reads data from '*data/pdf_list.csv*'
2. Performs data analysis using pandas
3. Generates tables and visualizations
4. Saves results to Word and Excel files in the '*results/general/*' directory

## 3. Word Frequency Analysis

**Main Notebook:** word_frequency.ipynb

This pipeline analyzes the frequency of words in the collected documents.

### Key Components

- *Database* class from scripts.database
    - Used for connection to database
- *WordFrequencyChart* class from scripts.word_frequency
    - Methods:
        - *top_20_words_category*: Returns the top 20 most used words by a defined category.
        - *top_20_words_lang*: Returns the top 20 most used words by language.
        - *frequency_certain_words*: Returns the word frequency from a list of selected words

### Process

1. Initializes logging
2. Reads data from '*data/database.db*'
3. Generates tables and visualizations
4. Saves results to PNG and CSV files in the '*results/word_frequency*' directory

## 4. Language Distribution Analysis

**Main Notebook:** language_distribution.ipynb

This pipeline analyzes the distribution of both French and English languages in the collected documents.

### Key Components

- *Database* class from scripts.database
    - Used for connection to database
- *LanguageDistributionChart* class from scripts.language_distribution
    - Methods:
        - *count_graph*: Returns the number of documents distributed by language tag ('fr', 'en', 'bilingual')
        - *language_percentage_distribution*: Returns the distribution of French and English words inside each document (pertinent to establish a bilinguism threshold for the topic analysis pipeline)

### Process

1. Initializes logging
2. Reads data from '*data/database.db*'
3. Creates tables and visualizations
4. Saves results to CSV, XLSX and PNG files in the "*results/language_distribution*' directory

## 5. Knowledge Type Analysis

**Main Notebook:** knowledge_type.ipynb

This pipeline analyzes the types of municipal knowledge mobilized by the collected documents. For more information, consult the author's master thesis.

### Key Components:

- *Database* class from scripts.database
    - Used for connection to database
- *KnowledgeType* class from scripts.knowledge_type
    - Methods:
        - *all_docs*: Returns a Venn diagram depicting the distribution of documents by knowledge type
        - *cross_table*: Returns a crosstable counting the number of documents of a determined data column by their respective knowledge type

### Process

1. Initializes logging
2. Reads data from '*data/database.db*'
3. Generates tables and visualizations
4. Saves results to CSV and XLSX files in the '*results/knowledge_type*' directory

## 6. Topic Analysis (*IN DEVELOPMENT*)

**Main Notebook:** topic_analysis.ipynb

This pipeline tokenizes text by language and performs comparative analysis by topics.

### Key Components

- *main* method from scripts.topic_analysis_main
    - Uses the *Database* class from scripts.database
        - Methods:
            - *fetch_single*: Fetches a specified data item from the database
            - *fetch_all*: Fetches all data from the database
    - Uses the *Process* class from scripts.topic_analysis.text_processing
        - Methods:
            - *single_doc*: Tokenizes one single document.
            - *docs_parallel*: Processes all documents using parallel processing.
    - Use the *Analysis* class from scripts.topic_analysis.analysis
        - Methods:
            - *process_documents*: Processes a list of documents and performs topic analysis

### Process

1. Initializes logging
2. Reads data from *data/database.db*
3. *Visualization and saving procedures to be implemented*

## 7. Sentiment Analysis (*IN DEVELOPMENT*)

**Main Notebook:** sentiment_analysis.ipynb

This pipelines applies sentiment analysis to the collected documents.

### Key Components

- *Database* class from scripts.database
    - Used for connection to database
- *SentimentAnalysis* class from scripts.sentiment_analysis
    - Methods:
        - *analyze_docs_by_language*: Analyzes documents organized by language
        - *get_avg_sentiment_by_category*: Gets average sentiment values for a determined data column.
        - *analyze_all_docs*: Analyzes all documents for sentiment analysis. Basically, it runs three instances of *analyze_docs_by_language* with lang = 'fr', 'en', and 'bilingual'

### Process

1. Initializes logging
2. Reads data from '*data/database.db*'
3. Generates tables and visualizations
4. Saves results to CSV and XLSX files in the '*results/sentiment_analysis*' directory

## Notes

Some pipelines are still in development, such as Topic Analysis and Sentiment Analysis. These may not be fully functional. Feel free to propose your own contributions in order to fully implement these pipelines.

For more detailed information about each pipeline or script, please refer to the inline documentation within each file.