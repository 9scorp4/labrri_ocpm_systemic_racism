# Pipeline Blueprints

This document provides detailed specifications for each analysis pipeline in the project. Each pipeline is designed to examine specific aspects of the OCPM documentation on systemic racism and discrimination.

## Table of Contents

1. [Update Database](#1-update-database)
2. [General Analysis Dashboard](#2-general-analysis-dashboard)
3. [Word Frequency Analysis](#3-word-frequency-analysis)
4. [Language Distribution Analysis](#4-language-distribution-analysis)
5. [Knowledge Type Analysis](#5-knowledge-type-analysis)
6. [Topic Analysis](#6-topic-analysis)
7. [Sentiment Analysis](#7-sentiment-analysis)

# Pipeline Blueprints

This document provides detailed specifications for each analysis pipeline in the project. Each component is designed to examine specific aspects of the OCPM documentation on systemic racism and discrimination.

## Table of Contents

1. [Update Database](#1-update-database)
2. [General Analysis Dashboard](#2-general-analysis-dashboard)
3. [Word Frequency Analysis](#3-word-frequency-analysis)
4. [Language Distribution Analysis](#4-language-distribution-analysis)
5. [Knowledge Type Analysis](#5-knowledge-type-analysis)
6. [Topic Analysis](#6-topic-analysis)
7. [Sentiment Analysis](#7-sentiment-analysis)

## 1. Update Database

**Main Script:** update_database.py  
**Purpose:** Database population and maintenance

### Components

#### Classes
- `Database` (scripts.database)
  - Database connection management
  - Data persistence operations
  - Migration handling

- `ProcessDocuments` (scripts.document_processing)
  - PDF processing
  - OCR handling
  - Text extraction

### Process Flow

1. **Initialization**
   - Configure logging
   - Establish database connection
   - Validate input files

2. **Document Processing**
   - Read CSV metadata
   - Process PDF documents
   - Apply OCR when needed

3. **Database Operations**
   - Update existing records
   - Insert new documents
   - Maintain data integrity

## 2. General Analysis Dashboard

**Main Component:** general_analysis_pipeline_dashboard.py
**Purpose:** Standalone dashboard for general document analysis and visualization

### Features

#### Document Analysis
- Total document count
- Document distribution by:
  - Organization
  - Document type
  - Category
  - Language
  - Clientele
  - Knowledge type

#### Cross-Analysis
- Organization vs Document Type
- Organization vs Category
- Document Type vs Language
- Category vs Clientele

#### Visualization Types
- Bar charts
- Pie charts
- Cross tables
- Heatmaps

### Dashboard Components

```python
class AnalysisDashboardApp:
    def __init__(self):
        self.columns = {
            'document_type': 'Document Type',
            'language': 'Language',
            'category': 'Category',
            'clientele': 'Clientele',
            'knowledge_type': 'Knowledge Type',
            'organization': 'Organization'
        }
        self.viz_types = ['Bar Chart', 'Pie Chart', 'Crosstable', 'Heatmap']
```

### Analysis Options

- **Single Distribution Analysis**
  - Select single variable
  - Choose visualization type
  - Apply filters

- **Cross Analysis**
  - Select primary and secondary variables
  - Generate contingency tables
  - Create comparative visualizations

### Data Export
- CSV export
- Excel export with formatting
- Interactive visualization export (HTML)
- PNG export for static visualizations

## 3. Word Frequency Analysis

**Main Components:**
- Dashboard: word_frequency_pipeline_dashboard.py
- Core Module: scripts/word_frequency/

### Components

#### Classes
- `WordFrequencyAnalyzer`
  - Term frequency calculation
  - TF-IDF analysis
  - Category comparisons

- `WordFrequencyVisualizer`
  - Frequency distribution plots
  - Comparison visualizations
  - Interactive word clouds

### Features

- Word frequency counting
- TF-IDF analysis
- Cross-category comparison
- Language-specific analysis

## 4. Language Distribution Analysis

**Main Components:**
- Dashboard: language_distribution_pipeline_dashboard.py
- Core Module: scripts/language_distribution/

### Components

#### Classes
- `LanguageDistributionAnalyzer`
  - Language detection
  - Code-switching analysis
  - Distribution calculation

- `LanguageDistributionVisualizer`
  - Distribution plots
  - Code-switching visualization
  - Comparative analysis plots

### Features

- Language detection
- Bilingual content analysis
- Code-switching patterns
- Distribution visualization

## 5. Knowledge Type Analysis

**Main Components:**
- Dashboard: knowledge_type_pipeline_dashboard.py
- Core Module: scripts/knowledge_type/

### Components

#### Classes
- `KnowledgeTypeAnalyzer`
  - Type classification
  - Intersection analysis
  - Distribution calculation

- `KnowledgeTypeVisualizer`
  - Venn diagrams
  - Distribution plots
  - Interactive visualizations

### Features

- Knowledge type classification
- Intersection analysis
- Cross-categorical distribution
- Pattern visualization

## 6. Topic Analysis

**Main Components:**
- Dashboard: topic_analysis_pipeline_dashboard.py
- Core Module: scripts/topic_analysis/

### Architecture

#### Manager Component (`TopicAnalysisManager`)
- Coordinates asynchronous operations
- Manages background processing thread
- Handles resource cleanup
- Provides async API for topic analysis operations

#### Analysis Component (`Analysis`)
- Performs topic modeling computations
- Supports multiple modeling approaches (LDA, NMF, LSA)
- Provides both synchronous and asynchronous interfaces
- Handles document preprocessing and vectorization

#### Topic Handler (`TopicHandler`)
- Manages topic persistence
- Provides async operations for topic management
- Handles topic similarity and updates

### Asynchronous Features
- Background event loop for non-blocking operations
- Parallel document processing
- Async database operations
- Resource management and cleanup
- Progress tracking and callback support

### Key Features
- Multiple modeling approaches
  - Latent Dirichlet Allocation (LDA)
  - Non-negative Matrix Factorization (NMF)
  - Latent Semantic Analysis (LSA)
- Topic coherence analysis
- Dynamic visualization
- Interactive exploration
- Asynchronous processing for large document sets

### Process Flow
1. Document preprocessing (async)
2. Vectorization (parallel)
3. Topic modeling (CPU-bound)
4. Coherence calculation
5. Topic filtering and labeling
6. Result persistence

### Usage Example
```python
# Initialize manager
manager = TopicAnalysisManager(db_path)

# Perform async analysis
topics_df = await manager.analyze_topics_async(
    doc_ids='all',
    method='lda',
    num_topics=20,
    coherence_threshold=-5.0
)

# Save results
saved_files = await manager.save_topics_async(topics_df)
```

## 7. Sentiment Analysis

**Main Components:**
- Dashboard: sentiment_analysis_pipeline_dashboard.py
- Core Module: scripts/sentiment_analysis/

### Components

#### Classes
- `SentimentAnalyzer`
  - Sentiment scoring
  - Aspect detection
  - Pattern analysis

- `SentimentVisualizer`
  - Sentiment distribution plots
  - Aspect-based visualization
  - Comparative analysis plots

### Features

- Document sentiment scoring
- Aspect-based analysis
- Cross-category comparison
- Pattern visualization

## Common Pipeline Features

### Error Handling
```python
try:
    # Pipeline operations
    pass
except DatabaseError:
    logger.error("Database operation failed")
    self._handle_db_error()
except ProcessingError:
    logger.error("Processing operation failed")
    self._handle_processing_error()
```

### Results Storage
```python
def save_results(self, results, pipeline_name):
    """Save pipeline results with consistent formatting."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = Path(f'results/{pipeline_name}/{timestamp}/')
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save results in appropriate formats
    self._save_data(results, save_path)
    self._save_visualizations(results, save_path)
    self._save_summary(results, save_path)
```

### Configuration
```python
PIPELINE_CONFIG = {
    'batch_size': 1000,
    'timeout': 3600,
    'workers': 3,
    'cache_results': True
}
```

Each pipeline maintains its own configuration while sharing common infrastructure for database access, logging, and result storage.

---

**Note:** This documentation reflects the current state of the project. For the most up-to-date information about each pipeline or script, please refer to the inline documentation within each file.