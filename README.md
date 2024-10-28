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

This repository is the result of the author's master's degree research on municipal knowledge concerning systemic discriminations in Montreal. It contains a series of scripts and tools used to process and analyze data from the Office de Consultation Publique de Montréal's Commission on Systemic Racism and Discrimination.

## Features

- Database creation and management using PostgreSQL
- PDF text extraction and processing
- OCR (Optimal Character Recognition) for image-based PDFs
- Word frequency analysis
- Language distribution analysis
- Knowledge type analysis
- Topic analysis
- Sentiment analysis
- Multilingual support (French and English)

## Project Structure

```
.
├── data                    # Raw data and database files
├── documentation           # Project documentation
├── logs                    # Log files
├── results                 # Analysis results
├── scripts                 # Main scripts for data processing and analysis
│   ├── topic_analysis      # Scripts specific to topic analysis
├── static                  # Static files for web interface (if applicable)
├── CONTRIBUTING.md         # Contribution guidelines
├── LICENSE                 # Project license
├── README.md               # This file
├── api.py                  # API for the project (if applicable)
├── environment.yml         # Conda environment file
├── exceptions.py           # Custom exception classes
├── *.ipynb                 # Jupyter notebooks for various analyses
├── populate_topics.py      # Script to populate topics in the database
├── requirements.txt        # Python dependencies
└── update_database.py      # Script to update the database
```

## Installation

1. Clone the repository
```
git clone https://github.com/9scorp4/labrri_ocpm_systemic_racism.git
cd labrri_ocpm_systemic_racism
```
2. Set up a PostgreSQL database and update the connection details in `scripts/database.py`
3. Create a virtual environment and install dependencies
```
python -m venv venv
source venv/bin/activate    # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
```
OR
```
conda create -n labrri_ocpm_systemic_racism python=3.11
conda activate labrri_ocpm_systemic_racism
pip install -r requirements.txt
```
4. Set up environment variables for database connection
```
export DB_USER=your_username
export DB_PASSWORD=your_password
```

## Usage

1. Populate or update the database
```
python update_database.py
```
2. Interact with specific analysis pipelines using their respective Jupyter notebooks

## Data Processing Pipelines

1. `update_database.py`: Populate or update the database
2. `general_analysis_pipeline_dashboard.py`: Perform a general data analysis
3. `word_frequency_pipeline_dashboard.py`: Analyze by word frequency
4. `language_distribution.ipynb`: Analyze by language distribution
5. `knowledge_type.ipynb`: Analyze by knowledge types
6. `topic_analysis.ipynb`: Tokenize text by language and compare analysis by topics
7. `sentiment_analysis.ipynb`: Analyze by sentiment level

For more details on each pipeline, refer to the [pipeline blueprints](https://github.com/9scorp4/labrri_ocpm_systemic_racism/tree/main/documentation/pipelines.md).

## Contributing

We welcome contributions for this project. Please read the [contribution guidelines](CONTRIBUTING.md) for more information on how to get started.

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