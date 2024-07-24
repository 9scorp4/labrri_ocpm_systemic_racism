# Pipeline blueprints

This document provides an overview of the different pipelines used in this project. Each pipeline corresponds to a specific analysis task and is associated with one or more Python scripts or Jupyter notebooks.

## 1. Update Database Pipeline

**Main Script:** update_database.py

This pipeline is responsible with populating the database with fetched data from the OCPM documentation.

### Key Components:

- *Database* class from scripts.database
    - Uses *csv_to_xlsx* method from class
- *ProcessDocuments* class from scripts.document_processing
    - Uses *process_documents* method from class

### Process:

1. Initializes logging
2. Read data from '*data/pdf_list.csv*'
3. Converts CSV to XLSX for handling in Nvivo software (not in this repo)
4. Processs PDF documents using *process_documents* method

## 2. General Analysis

**Main Notebook:** general_analysis.ipynb

This pipeline performs a general analysis on the connected data

### Key Components:

Two main analyses:
1. **Organization category count:** How many participant organizations there are by category.
2. **Crosstables:** Cross-referencing of data columns.
    - Organization Category and Document Type
    - Organization Category and Clientele

### Process:

1. Reads data from '*data/pdf_list.csv*'
2. Performs data analysis using pandas
3. Generates tables and visualizations
4. Saves results to Word and Excel files in the '*results/general/*' directory

## 3. Word Frequency Analysis

**Main Notebook:** word_frequency.ipynb

This pipeline analyzes the frequency of words in the collected documents.

### Key Components:

- *WordFrequencyChart* class from scripts.word_frequency
    - Methods:
        - *top_20_words_category*: Returns the top 20 most used words by a defined category.
        - *top_20_words_land*: Returns the top 20 most used words by language.
        - *frequency_certain_words*: Returns the word frequency from a list of selected words

### Process:

1. Initializes logging
2. Read data from '*data/database.db*'
3. Generates tables and visualizations
4. Saves results to PNG and CSV files in the '*results/word_frequency*' directory

## 4. Language Distribution Analysis

**Main Notebook:** language_distribution.ipynb

This pipeline analyzes the distribution of both French and English languages in the collected documents.

### Key Components:

- *LanguageDistributionChart* class from scripts.language_distribution
    - Methods:
        - *count_graph*: Returns the number of documents distributed by language tag ('fr', 'en', 'bilingual')
        - *language_percentage_distribution*: Returns the distribution of French and English words inside each document (pertinent to establish a bilinguism threshold for the topic analysis pipeline)

## 5. Knowledge Type Analysis

**Main Notebook:** knowledge_type.ipynb

This pipeline analyzes the types of municipal knowledge mobilized by the collected documents. For more information, consult the author's master thesis.

### Key Components:
- *KnowledgeType* class from scripts.knowledge_type
    - Methods:
        - *all_docs*: Returns a Venn diagram depicting the distribution of documents by knowledge type
        - *cross_table*: Returns a crosstable counting the number of documents of a determined data column by their respective knowledge type

### Process:

1. Initializes logging
2. Read data from '*data/database.db*'
3. Generates tables and visualizations
4. Saves results to CSV and XLSX files in the '*results/knowledge_type*' directory

## 6. Topic Analysis (IN DEVELOPMENT)

**Main Notebook:** topic_analysis.ipynb

This pipeline tokenizes text by language and performs comparative analysis by topics.

### Key Components:

- *main* method from scripts.topic_analysis_main
    - calls the methods *fetch_single* and *fetch_all* from the *Database* class from scripts.database
    - calls the *process_documents* method from the *Analysis* class from scripts.topic_analysis.text_processing
    - calls *Process* class from scripts.topic_analysis.analysis?? (review implementation)

## 7. Sentiment Analysis