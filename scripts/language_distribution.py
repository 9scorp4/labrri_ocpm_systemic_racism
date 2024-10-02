from loguru import logger
import pandas as pd
import matplotlib.pyplot as plt
from langdetect import detect, DetectorFactory
from nltk.tokenize import sent_tokenize
import spacy
from collections import Counter
from tqdm import tqdm
import seaborn as sns
import multiprocessing
from functools import partial
from collections import defaultdict

from scripts.database import Database

# Set seed for reproducibility in language detection
DetectorFactory.seed = 0

def process_single_document(doc, nlp_fr, nlp_en):
    doc_id, organization, language, content, category, document_type = doc

    nlp = nlp_fr if language == 'fr' else nlp_en
    processed_doc = nlp(content)

    lang_counts = {
        'fr': 0,
        'en': 0,
        'other': 0
    }
    other_samples = defaultdict(list)
    total_chars = 0
    code_switches = 0
    previous_lang = None

    for sent in processed_doc.sents:
        if len(sent.text.strip()) > 18:
            try:
                sent_lang = detect(sent.text)
                chars = len(sent.text)

                if sent_lang in ['fr', 'en']:
                    lang_counts[sent_lang] += chars
                else:
                    lang_counts['other'] += chars
                    other_samples[sent_lang].append(sent.text[:50])
                
                total_chars += chars

                if previous_lang and sent_lang != previous_lang:
                    code_switches += 1
                previous_lang = sent_lang

            except Exception as e:
                logger.error(f"Error processing sentence in document {doc_id}: {e}")

    total_chars = sum(lang_counts.values())
    en_percentage = (lang_counts['en'] / total_chars) * 100 if total_chars > 0 else 0
    fr_percentage = (lang_counts['fr'] / total_chars) * 100 if total_chars > 0 else 0
    other_percentage = (lang_counts['other'] / total_chars) * 100 if total_chars > 0 else 0

    top_other_langs = sorted(other_samples.items(), key=lambda x: sum(len(s) for s in x[1]), reverse=True)[:3]

    return {
        'Document ID': doc_id,
        'Organization': organization,
        'Declared Language': language,
        'Category': category,
        'Document Type': document_type,
        'English (%)': en_percentage,
        'French (%)': fr_percentage,
        'Other (%)': other_percentage,
        'Code Switches': code_switches,
        'Other Lang 1': top_other_langs[0][0] if top_other_langs else None,
        'Other Lang 1 Sample': '; '.join(top_other_langs[0][1][:3]) if top_other_langs else None,
        'Other Lang 2': top_other_langs[1][0] if len(top_other_langs) > 1 else None,
        'Other Lang 2 Sample': '; '.join(top_other_langs[1][1][:3]) if len(top_other_langs) > 1 else None,
        'Other Lang 3': top_other_langs[2][0] if len(top_other_langs) > 2 else None,
        'Other Lang 3 Sample': '; '.join(top_other_langs[2][1][:3]) if len(top_other_langs) > 2 else None,
    }

class LanguageDistributionChart:
    def __init__(self, db_path):
        if db_path is None:
            raise ValueError('db_path cannot be None')

        self.db_path = db_path
        self.db = Database(self.db_path)
        
        # Load spaCy models for improved language detection and code-switching analysis
        self.nlp = {
            'fr': spacy.load('fr_core_news_sm'),
            'en': spacy.load('en_core_web_sm')
        }

        logger.info("LanguageDistributionChart initialized successfully")

    def count_graph(self, where):
        logger.info(f'Generating graph for {where}')

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

        df = self.db.df_from_query(query)
        if df is None or df.empty:
            logger.warning(f"No data found for {'category: ' + where if where != 'All categories' else 'all categories'}")
            return None

        languages = df.iloc[:, 0].tolist()
        counts = df.iloc[:, 1].tolist()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

        ax1.pie(counts, labels=languages, autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')
        ax1.set_title("Language Distribution")

        ax2.bar(languages, counts)
        ax2.set_title("Document Count by Language")
        ax2.set_xlabel("Language")
        ax2.set_ylabel("Number of Documents")

        for i, v in enumerate(counts):
            ax2.text(i, v, str(v), ha='center', va='bottom')

        plt.suptitle(f"Language Distribution for {'All Categories' if where == 'All categories' else 'Category: ' + where}")
        plt.tight_layout()
        plt.savefig(f'results/language_distribution/{where.replace(" ", "_").lower()}_count.png')
        plt.close()

        df.columns = ['Language', 'Count']
        df.to_csv(f'results/language_distribution/{where.replace(" ", "_").lower()}_count.csv', index=False)
        logger.info(f'Graph generated and saved for {where}')

        return df

    def improved_language_detection(self, text):
        try:
            if not text or not text.strip():
                return 'unknown'
            
            lang = detect(text)

            if lang == 'fr':
                doc = self.nlp_fr(text[:1000])
                fr_words = sum(1 for token in doc if token.lang_ == 'fr')
                if fr_words / len(doc) < 0.5:
                    lang = 'en'
            elif lang == 'en':
                doc = self.nlp_en(text[:1000])
                en_words = sum(1 for token in doc if token.lang_ == 'en')
                if en_words / len(doc) < 0.5:
                    lang = 'fr'

            return lang
        except Exception as e:
            logger.error(f"Failed to detect language. Error: {e}")
            return 'unknown'

    def analyze_code_switching(self, text):
        sentences = sent_tokenize(text)
        switches = 0
        prev_lang = None
        
        for sentence in sentences:
            curr_lang = self.improved_language_detection(sentence)
            if prev_lang and curr_lang != prev_lang and curr_lang != 'unknown':
                switches += 1
            prev_lang = curr_lang if curr_lang != 'unknown' else prev_lang
        
        return switches

    def language_percentage_distribution(self, where):
        logger.info(f'Analyzing language distribution for {where}')
        
        if where == "All languages":
            query = "SELECT d.id, d.organization, d.language, c.content, d.category, d.document_type FROM documents d INNER JOIN content c ON d.id = c.doc_id"
        else:
            query = f"SELECT d.id, d.organization, d.language, c.content, d.category, d.document_type FROM documents d INNER JOIN content c ON d.id = c.doc_id WHERE d.language = '{where}'"
        
        df = self.db.df_from_query(query)
        if df is None or df.empty:
            logger.warning(f"No data found for {where}")
            return None
        
        docs = list(df.itertuples(index=False, name=None))

        with multiprocessing.Pool() as pool:
            process_doc = partial(process_single_document, nlp_fr=self.nlp['fr'], nlp_en=self.nlp['en'])
            results = list(tqdm(pool.imap(process_doc, docs), total=len(docs), desc='Processing Documents'))

        result_df = pd.DataFrame(results)

        avg_columns = ['English (%)', 'French (%)', 'Other (%)', 'Code Switches']
        overall_averages = result_df[avg_columns].mean()
        overall_averages = pd.DataFrame(overall_averages).T
        overall_averages['Document ID'] = 'Overall Average'
        overall_averages['Organization'] = ''
        overall_averages['Declared Language'] = ''
        overall_averages['Category'] = ''
        overall_averages['Document Type'] = ''
        
        for i in range(1, 4):
            lang_col = f'Other Lang {i}'
            sample_col = f'Other Lang {i} Sample'
            print(f"\n{lang_col} samples:")
            for lang, sample in zip(result_df[lang_col], result_df[sample_col]):
                if pd.notnull(lang) and pd.notnull(sample):
                    print(f"{lang}: {sample}")

            overall_averages[lang_col] = result_df[lang_col].mode().iloc[0] if not result_df[lang_col].isna().all() else None

            samples = result_df[sample_col].dropna().tolist()[:3]
            overall_averages[sample_col] = ', '.join(samples) if samples else 'No samples'

        result_df = pd.concat([result_df, overall_averages], ignore_index=True)

        result_df.to_csv(f'results/language_distribution/{where.replace(" ", "_").lower()}_percentage.csv', index=False)
        result_df.to_excel(f'results/language_distribution/{where.replace(" ", "_").lower()}_percentage.xlsx', index=False)

        logger.info(f'Graph generated and saved for {where}')

        return result_df

    def visualize_language_distribution(self, data):
        logger.info('Visualizing language distribution')

        # Create a box plot for language percentages
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=data[['English (%)', 'French (%)', 'Other (%)']])
        plt.title('Distribution of Language Percentages')
        plt.ylabel('Percentage')
        plt.savefig('results/language_distribution/language_percentage_distribution.png')
        plt.close()

        # Create a histogram for code switches
        plt.figure(figsize=(12, 6))
        sns.histplot(data=data, x='Code Switches', kde=True)
        plt.title('Distribution of Code Switches')
        plt.xlabel('Number of Code Switches')
        plt.ylabel('Count')
        plt.savefig('results/language_distribution/code_switches_distribution.png')
        plt.close()

        # Create a grouped bar plot for average language percentages by category
        category_averages = data.groupby('Category')[['English (%)', 'French (%)', 'Other (%)']].mean()
        category_averages.plot(kind='bar', figsize=(12, 6))
        plt.title('Average Language Percentages by Category')
        plt.xlabel('Category')
        plt.ylabel('Percentage')
        plt.legend(title='Language')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('results/language_distribution/language_percentages_by_category.png')
        plt.close()

        logger.info('Language distribution visualizations completed')

    def analyze_all(self):
        logger.info('Starting comprehensive language distribution analysis')

        # Analyze overall distribution
        overall_dist = self.count_graph("All categories")

        # Analyze distribution by category
        categories = self.db.df_from_query("SELECT DISTINCT category FROM documents")['category'].tolist()
        for category in categories:
            self.count_graph(category)

        # Detailed language analysis for all languages
        detailed_analysis = self.language_percentage_distribution("All languages")

        # Visualize language distribution
        self.visualize_language_distribution(detailed_analysis)

        logger.info('Comprehensive language distribution analysis completed')
        return overall_dist, detailed_analysis

if __name__ == "__main__":
    db_path = "path/to/your/database.db"
    chart = LanguageDistributionChart(db_path)
    overall_dist, detailed_analysis = chart.analyze_all()
    print(overall_dist)
    print(detailed_analysis.describe())