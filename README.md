# Projet OCPM: Racisme systémique

[<img src="documentation/images/labbri_logo_retina.png" alt="LABRRI" width="500"/>](https://labrri.net/)  [<img src="documentation/images/universite-de-montreal-logo-png-transparent.png" alt="Université de Montréal" width="200"/>](https://umontreal.ca/)

Welcome to Projet "OCPM: Racisme systémique", by Nicolas Arias Garcia (9scorp4) et al. All data used for analysis come from the documentation dossier of the Office de Consultation Publique de Montréal's Commission on Systemic Racism and Discrimination (available [here](https://ocpm.qc.ca/fr/r%26ds/documentation)).

## Table of Contents

- [About](#about)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data Processing Pipelines](#data-processing-pipelines)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

## About

This repository is part of a master's degree research project investigating municipal knowledge concerning systemic discriminations in Montreal. It provides a comprehensive suite of tools and pipelines for processing and analyzing data from the Office de Consultation Publique de Montréal's Commission on Systemic Racism and Discrimination.

## Features

- Database Management
  - PostgreSQL integration
  - Automated database updates and maintenance
  - Document versioning and tracking


- Document Processing
  - PDF text extraction and preprocessing
  - OCR (Optical Character Recognition) for image-based PDFs
  - Multi-format document support
  - Language detection and processing


- Analysis Pipelines
  - Word Frequency Analysis
    - Term frequency analysis
    - TF-IDF calculations
    - Category comparisons

  - Language Distribution Analysis
    - Bilingual content detection
    - Code-switching analysis
    - Language pattern recognition

  - Knowledge Type Analysis
    - Knowledge classification
    - Cross-categorical analysis
    - Intersection mapping

  - Topic Analysis
    - Multiple modeling approaches (LDA, NMF, LSA)
    - Topic coherence analysis
    - Dynamic topic visualization

  - Sentiment Analysis
    - Document sentiment scoring
    - Aspect-based analysis
    - Cross-category comparisons

## Project Structure

```
.
├── alembic/                # Database migrations
├── data/                   # Raw data and database files
├── documentation/          # Project documentation
├── logs/                   # Log files
├── results/                # Analysis results by pipeline
│   ├── knowledge_type/
│   ├── language_distribution/
│   ├── sentiment_analysis/
│   ├── topic_analysis/
│   └── word_frequency/
├── scripts/                # Core processing modules
│   ├── knowledge_type/     # Knowledge type analysis
│   ├── language_distribution/
│   ├── sentiment_analysis/
│   ├── topic_analysis/     # Topic modeling components
│   └── word_frequency/     # Word frequency analysis
├── static/                 # Static assets
├── tests/                  # Test suite
└── *_pipeline_dashboard.py # Pipeline-specific dashboards
```

## Installation

1. Clone the repository
```bash
git clone https://github.com/9scorp4/labrri_ocpm_systemic_racism.git
cd labrri_ocpm_systemic_racism
```
2. Set up a PostgreSQL database and update the connection details in `scripts/database.py`
```bash
# Create database and user
psql -U postgres
CREATE DATABASE labrri_ocpm_systemic_racism;
CREATE USER your_username WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE labrri_ocpm_systemic_racism TO your_username;
```
3. Create a virtual environment and install dependencies
```bash
# Using venv
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

# Or using conda
conda env create -f environment.yml
conda activate labrri_ocpm_systemic_racism
```
4. Install dependencies
```bash
pip install -r requirements.txt
```
5. Configure environment variables
```bash
# Create .env file
echo "DB_USER=your_username" >> .env
echo "DB_PASSWORD=your_password" >> .env
echo "DB_HOST=localhost" >> .env
echo "DB_PORT=5432" >> .env
echo "DB_NAME=labrri_ocpm_systemic_racism" >> .env
```

## Usage

1. Initialize the database
```bash
python update_database.py
```
2. Populate initial topics (if working with topic analysis)
```bash
python populate_topics.py
```
3. Run desired analysis pipeline
```bash
# General Analysis
python general_analysis_pipeline_dashboard.py

# Word Frequency Analysis
python word_frequency_pipeline_dashboard.py

# Language Distribution Analysis
python language_distribution_pipeline_dashboard.py

# Knowledge Type Analysis
python knowledge_type_pipeline_dashboard.py

# Topic Analysis
python topic_analysis_pipeline_dashboard.py

# Sentiment Analysis
python sentiment_analysis_pipeline_dashboard.py
```
4. Access analysis
```bash
# Results are saved in pipeline-specific directories
cd results/<pipeline_name>/
```

## Data Processing Pipelines

The project includes several specialized analysis pipelines, each designed to examine different aspects fo the documentation:

1. **General Analysis Pipeline**
    - Basic document statistics and classification
    - Cross-categorical analysis
    - Document distribution visualization

2. **Word Frequency Analysis Pipeline**
    - Term frequency analysis
    - TF-IDF analysis
    - Category-based comparisons

3. **Language Distribution Analysis Pipeline**
    - Language detection and classification
    - Code-switching analysis
    - Bilingual content distribution

4. **Knowledge Type Analysis Pipeline**
    - Knowledge classification
    - Intersection analysis
    - Category distribution

5. **Topic Analysis Pipeline**
    - Topic modeling (LDA/NMF/LSA)
    - Topic coherence evaluation
    - Topic visualization

6. **Sentiment Analysis Pipeline**
    - Document sentiment scoring
    - Aspect-based sentiment analysis
    - Cross-category sentiment comparison

For more information, see [pipeline blueprints](documentation/pipelines.md).

## Contributing

Please read our [contribution guidelines](CONTRIBUTING.md) before submitting pull requests.

## License

This work is licensed under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/?ref=chooser-v1).

### You are free to:
- Share — copy and redistribute the material in any medium or format.
- Adapt — remix, transform and build upon the material.

The licensor cannot revoke these freedoms as long as you follow the license terms.

### Under the following terms:
- Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
- NonCommercial — You may not use the material for commercial purposes.
- ShareAlike — If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.
- No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

### Notices:
You do not have to comply with the license for elements of the material in the public domain or where your use is permitted by an applicable exception or limitation.

No warranties are given. The license may not give you all of the permissions necessary for your intended use. For example, other rights such as publicity, privacy, or moral rights may limit how you use the material.

## Acknowledgements

I want to thank my colleagues at the Laboratoire de recherche en relations interculturelles (LABRRI) for their contributions to my master's research. By their names:
* Bob White
* Isabelle Comtois
* Maude Arsenault
* Roxane Archambault
* Fritz Gerald Louis

Special thanks to the contributors to this repository as well.

## Contact

### Nicolas Arias Garcia
- Email: ariasg.nicolas@gmail.com  •  nicolas.arias.garcia@umontreal.ca
- GitHub: [9scorp4](https://github.com/9scorp4)
- LinkedIn: [nicag](https://www.linkedin.com/in/nicag/)

For any questions or feedback, please open an issue on this repository or contact the maintainers directly.