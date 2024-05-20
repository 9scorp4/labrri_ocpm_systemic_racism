# Projet OCPM: Racisme systémique

[<img src="images/labbri_logo_retina.png" alt="LABRRI" width="500"/>](https://labrri.net/)  [<img src="images/universite-de-montreal-logo-png-transparent.png" alt="Université de Montréal" width="200"/>](https://umontreal.ca/)

Welcome to Projet "OCPM: Racisme systémique", by Nicolas Arias Garcia (9scorp4) et al. All data used for analysis come from the documentation dossier of the Office de Consultation Publique de Montréal's Commission on Systemic Racism and Discrimination (available [here](https://ocpm.qc.ca/fr/r%26ds/documentation)).

## Table of Contents

- [About](#about)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

## About

This repository is the result of the author's master's degree research on municipal knowledge concerning systemic discriminations in Montreal. It contains a series of scripts and tools used to process and analyze data from the Office de Consultation Publique de Montréal's Commission on Systemic Racism and Discrimination.

The 'data' folder contains raw data (PDF documents from the Commission), the database to store the processed data, the list of PDF files to retrieve content from (pdf_list.csv and pdf_list.xlsx) and the database schema (db_schema.pdf).

The 'scripts' folder contains all scripts used for general document and data operations. The scripts at the root folder execute their respective pipeline.

Results from all pipelines must be stored in the 'results' folder. For debugging, consult the 'logs' folder.

## Features

- Database creation and population.
- PDF Text processing.
- OCR Recognition.
- Word Frequency Analysis.
- Language Distribution Analysis.
- Topic Analysis By Language (IN DEVELOPMENT).

## Installation

Just download it as a ZIP file, extract and explore with your preferred IDE.

## Usage

- 'update_database.py' pipeline: Populate database using fetched data.
- 'general_analysis.ipynb' pipeline: Perform a general data analysis.
- 'word_frequency.ipynb' pipeline: Analyze by word frequency.
- 'language_distribution.ipynb' pipeline: Analyze by language distribution.
- 'knowledge_type.ipynb' pipeline: Analyze by knowledge types.
- 'topic_analysis.ipynb' pipeline: Tokenize text by language and compared analysis by topics.

## Contributing

Please read the [contribution guidelines](CONTRIBUTING.md) for more information.

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

Special thanks to the contributors to this repository as well:
* (ADD CONTRIBUTORS)

## Contact

### Nicolas Arias Garcia
- Email: ariasg.nicolas@gmail.com  •  nicolas.arias.garcia@umontreal.ca
- Web portfolio: IN CONSTRUCTION
- GitHub: [9scorp4](https://github.com/9scorp4)
- LinkedIn: [nicag](https://www.linkedin.com/in/nicag/)
