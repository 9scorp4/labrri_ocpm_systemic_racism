# Pipeline Blueprints

This document provides an overview of the different pipelines used in this project. Each pipeline corresponds to a specific analysis task and is associated with one or more Python scripts or Jupyter notebooks.

## Table of Contents

1. [Update Database](#1-update-database)
2. [General Analysis](#2-general-analysis)
3. [Word Frequency Analysis](#3-word-frequency-analysis)
4. [Language Distribution Analysis](#4-language-distribution-analysis)
5. [Knowledge Type Analysis](#5-knowledge-type-analysis)
6. [Topic Analysis](#6-topic-analysis)
7. [Sentiment Analysis](#7-sentiment-analysis)

## 1. Update Database

**Main Script:** update_database.py

This pipeline is responsible for populating the PostgreSQL database with fetched data from the OCPM documentation.

### Key Components

- `Database` class from scripts.database
    - Uses `csv_to_xlsx` method from class
- `ProcessDocuments` class from scripts.document_processing
    - Uses `pdf_batch` method from class

### Process

1. Initializes logging
2. Reads data from 'data/pdf_list.csv'
3. Converts CSV to XLSX for handling in Nvivo software (not in this repo)
4. Processes PDF documents using `pdf_batch` method
5. Applies OCR to documents with missing or incomplete content
6. Updates the PostgreSQL database with new and modified documents

## 2. General Analysis

**Main Notebook:** general_analysis.ipynb

This pipeline performs a general analysis on the connected data.

### Key Components

- `Database` class from scripts.database
    - Used for connection to PostgreSQL database

### Process

1. Connects to the PostgreSQL database
2. Performs data analysis using pandas:
   - Organization category count
   - Crosstables:
     - Organization Category and Document Type
     - Organization Category and Clientele
3. Generates tables and visualizations
4. Saves results to Word and Excel files in the 'results/general/' directory

## 3. Word Frequency Analysis

**Main Notebook:** word_frequency.ipynb

This pipeline analyzes the frequency of words in the collected documents.

### Key Components

- `Database` class from scripts.database
    - Used for connection to PostgreSQL database
- `WordFrequencyChart` class from scripts.word_frequency
    - Methods:
        - `top_n_words`: Returns the top N most used words by category or language
        - `compare_categories`: Compares word frequency across categories
        - `compare_languages`: Compares word frequency across languages
        - `tfidf_analysis`: Performs TF-IDF analysis

### Process

1. Initializes logging
2. Connects to the PostgreSQL database
3. Performs word frequency analysis for different categories and languages
4. Generates tables and visualizations
5. Saves results to PNG, CSV, and XLSX files in the 'results/word_frequency' directory

## 4. Language Distribution Analysis

**Main Notebook:** language_distribution.ipynb

This pipeline analyzes the distribution of both French and English languages in the collected documents.

### Key Components

- `Database` class from scripts.database
    - Used for connection to PostgreSQL database
- `LanguageDistributionChart` class from scripts.language_distribution
    - Methods:
        - `count_graph`: Returns the number of documents distributed by language tag
        - `language_percentage_distribution`: Returns the distribution of French and English words inside each document
        - `analyze_code_switching`: Analyzes code-switching between languages

### Process

1. Initializes logging
2. Connects to the PostgreSQL database
3. Performs language distribution analysis
4. Creates tables and visualizations
5. Saves results to CSV, XLSX, and PNG files in the 'results/language_distribution' directory

## 5. Knowledge Type Analysis

**Main Notebook:** knowledge_type.ipynb

This pipeline analyzes the types of municipal knowledge mobilized by the collected documents.

### Key Components

- `Database` class from scripts.database
    - Used for connection to PostgreSQL database
- `KnowledgeType` class from scripts.knowledge_type
    - Methods:
        - `all_docs`: Returns a Venn diagram depicting the distribution of documents by knowledge type
        - `crosstable`: Returns a crosstable counting the number of documents of a determined data column by their respective knowledge type
        - `analyze_intersections`: Analyzes intersections of different knowledge types

### Process

1. Initializes logging
2. Connects to the PostgreSQL database
3. Performs knowledge type analysis
4. Generates tables and visualizations
5. Saves results to CSV, XLSX, and PNG files in the 'results/knowledge_type' directory

## 6. Topic Analysis

**Main Script:** populate_topics.py
**Main Notebook:** topic_analysis.ipynb

This pipeline tokenizes text by language and performs comparative analysis by topics.

### Key Components

- `Database` class from scripts.database
    - Used for connection to PostgreSQL database
- `Analysis` class from scripts.topic_analysis.analysis
    - Methods:
        - `analyze_docs`: Processes documents and performs topic analysis
        - `vectorize`: Vectorizes documents using TF-IDF
        - `_perform_topic_modeling`: Performs topic modeling using LDA, NMF, or LSA
- `TopicLabeler` class from scripts.topic_analysis.topic_labeler
    - Used for labeling topics

### Process

1. Initializes logging
2. Connects to the PostgreSQL database
3. Fetches documents from the database
4. Performs topic analysis using LDA, NMF, or LSA
5. Labels topics using the TopicLabeler
6. Calculates topic coherence scores
7. Filters topics based on coherence scores
8. Saves results to CSV and XLSX files in the 'results/topic_analysis' directory

## 7. Sentiment Analysis

**Main Notebook:** sentiment_analysis.ipynb

This pipeline applies sentiment analysis to the collected documents.

### Key Components

- `Database` class from scripts.database
    - Used for connection to PostgreSQL database
- `SentimentAnalysis` class from scripts.sentiment_analysis
    - Methods:
        - `analyze_sentiment`: Analyzes sentiment of a given text
        - `aspect_based_sentiment`: Performs aspect-based sentiment analysis
        - `analyze_docs_by_language`: Analyzes documents organized by language
        - `analyze_all_docs`: Analyzes all documents for sentiment analysis

### Process

1. Initializes logging
2. Connects to the PostgreSQL database
3. Performs sentiment analysis on documents
4. Conducts aspect-based sentiment analysis
5. Generates tables and visualizations
6. Saves results to CSV and XLSX files in the 'results/sentiment_analysis' directory

---

Note: This document reflects the current state of the project, including the transition from SQLite to PostgreSQL. Some pipelines may still be in development or undergoing refinement. For more detailed information about each pipeline or script, please refer to the inline documentation within each file.