from loguru import logger
from collections import Counter
from itertools import chain
import spacy
import nltk
from nltk.corpus import wordnet, stopwords
from nltk.util import ngrams
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

class TopicLabeler:
    def __init__(self, vectorizer, domain_terms, lang):
        self.vectorizer = vectorizer
        self.domain_terms = domain_terms
        self.lang = lang
        self.stopwords = set(stopwords.words('english') + stopwords.words('french'))

        if lang == 'fr':
            self.nlp = spacy.load('fr_core_news_md')
        elif lang == 'en':
            self.nlp = spacy.load('en_core_web_md')
        else:
            self.nlp = {
                'fr': spacy.load('fr_core_news_md'),
                'en': spacy.load('en_core_web_md')
            }

    def initialize_related_terms(self, doc_texts):
        """
        Initialize the term similarity matrix and feature names.

        Args:
            doc_texts (list): List of document texts.
        """
        if not hasattr(self.vectorizer, 'vocabulary_'):
            self.vectorizer.fit(doc_texts)

        self.feature_names = self.vectorizer.get_feature_names_out()
        tfidf_matrix = self.vectorizer.transform(doc_texts)

        # Calculate the cosine similarity between all terms
        term_similarity = cosine_similarity(tfidf_matrix.T)

        # Store the term similarity matrix
        self.term_similarity_matrix = term_similarity

    def label_topics(self, topics, doc_texts):
        """
        Label the topics by finding representative words in the documents.

        Args:
            topics (list): List of topics, where each topic is a list of words.
            doc_texts (list): List of document texts.

        Returns:
            list: List of labeled topics, where each topic is a tuple of (label, topic words).
        """
        labeled_topics = []
        all_topic_words = set(word for topic in topics for word in (topic if isinstance(topic, (list, tuple)) else [str(topic)]))

        for i, topic in enumerate(topics):
            if isinstance(topic, (list, tuple)):
                words = topic
            elif isinstance(topic, int):
                words = [str(topic)]
            else:
                words = list(topic)

            unique_words = [word for word in words if word not in self.stopwords and sum(word in t for t in topics) == 1]
            common_words = [word for word in words if word not in unique_words and word not in self.stopwords]

            if unique_words:
                main_theme = unique_words[0]
            elif common_words:
                main_theme = common_words[0]
            else:
                main_theme = words[0]

            additional_words = unique_words[1:3] if len(unique_words) > 1 else common_words[:2]
            label = f"Topic {i+1}: {main_theme.capitalize()}"
            if additional_words:
                label += f" ({', '.join(additional_words)})"
            
            labeled_topics.append((label, words))
        
        return labeled_topics

    def _generate_topic_label(self, topic_words, doc_texts, num_words=5, used_terms=None, topic_index=0, all_topics=None):
        if used_terms is None:
            used_terms = set()

        # Get top n-grams
        top_ngrams = self._get_top_ngrams(topic_words, num_words * 3)
        
        # Use TF-IDF scores to get important words from the documents
        tfidf_matrix = self.vectorizer.fit_transform(doc_texts)
        feature_names = self.vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.sum(axis=0).A1
        important_words = [feature_names[i] for i in tfidf_scores.argsort()[::-1][:num_words * 3]]
        
        # Combine and prioritize domain-specific terms
        label_candidates = list(set(top_ngrams + important_words))
        domain_specific = [word for word in label_candidates
                            if any(term.lower() in word.lower() for term in self.domain_terms)]
        if not domain_specific:
            domain_specific = label_candidates

        # Calculate term frequencies
        term_freq = Counter(domain_specific)
        if all_topics:
            other_topics = [t for i, t in enumerate(all_topics) if i != topic_index]
            other_topics_words = set(chain(*other_topics))
            uniqueness_score = {term: term_freq[term] / (1 + sum(term in topic for topic in other_topics_words))
                                for term in domain_specific}
            sorted_terms = sorted(uniqueness_score.items(), key=lambda x: x[1], reverse=True)
        else:
            sorted_terms = sorted(term_freq.items(), key=lambda x: x[1], reverse=True)

        # Generate label
        key_terms = []
        other_terms = []

        for term, score in sorted_terms:
            lower_term = term.lower()
            if len(key_terms) < 2 and lower_term not in used_terms and lower_term in self.domain_terms:
                key_terms.append(term)
                used_terms.add(lower_term)
                related_terms = self._find_related_terms(term)
                other_terms.extend([t for t in related_terms if t.lower() not in used_terms])
                used_terms.update([t.lower() for t in related_terms])
            elif len(other_terms) < num_words - 2 and lower_term not in used_terms:
                other_terms.append(term)
                used_terms.add(lower_term)

            if len(key_terms) + len(other_terms) >= num_words:
                break
            
        while len(key_terms) + len(other_terms) < min(num_words, len(sorted_terms)):
            for term, _ in sorted_terms:
                if term not in key_terms and term not in other_terms:
                    if len(key_terms) > 2:
                        key_terms.append(term)
                    else:
                        other_terms.append(term)
                    break
        
        if key_terms:
            main_theme = ' '.join(key_terms).capitalize()
            if other_terms:
                sub_themes = ', '.join(other_terms)
                label = f"Topic {topic_index + 1}: {main_theme} - {sub_themes}"
            else:
                label = f"Topic {topic_index + 1}: {main_theme}"
        else:
            label = f"Topic {topic_index + 1}: " + ', '.join(other_terms) if other_terms else f"Miscellaneous topic {topic_index + 1}"

        return label, used_terms

    def generate_label(self, top_words, docs):
        # Use NLP techniques to generate a more meaningful label
        combined_words = ' '.join(top_words)
        
        if isinstance(self.nlp, dict):
            en_doc = self.nlp['en'](combined_words)
            fr_doc = self.nlp['fr'](combined_words)

            if len(en_doc.ents) + len(list(en_doc.noun_chunks)) > len(fr_doc.ents) + len(list(fr_doc.noun_chunks)):
                doc = en_doc
                lang = 'en'
            else:
                doc = fr_doc
                lang = 'fr'
        else:
            doc = self.nlp(combined_words)
            lang = self.lang
        
        noun_phrases = [chunk.text for chunk in doc.noun_chunks]

        if noun_phrases:
            label = noun_phrases[0].capitalize()
        else:
            label = top_words[0].capitalize()

        domain_context = next((term for term in self.domain_terms if term.lower() in combined_words.lower()), '')

        if domain_context:
            label += f" ({domain_context})"

        if isinstance(self.nlp, dict):
            label += f" [{lang.upper()}]"

        return label

    def _calculate_term_score(self, term, topic_freq, all_topics_freq):
        tf = topic_freq[term]
        df = sum(1 for topic in all_topics_freq if term in topic)
        tfidf = tf * (np.log(len(all_topics_freq) / df) + 1)
        domain_relevance = 2 if term in self.domain_terms else 1
        return tfidf * domain_relevance

    def _get_top_ngrams(self, topic_words, n=5):
        preprocessed_words = self._preprocess_for_labeling(topic_words)
        if not preprocessed_words:
            return []
        
        unigrams = preprocessed_words
        bigrams = [' '.join(gram) for gram in ngrams(preprocessed_words, 2)]
        trigrams = [' '.join(gram) for gram in ngrams(preprocessed_words, 3)]

        all_ngrams = unigrams + bigrams + trigrams
        ngram_freq = Counter(all_ngrams)

        return [ngram for ngram, _ in ngram_freq.most_common(n)]

    def _find_related_terms(self, term, n=5):
        """
        Find related terms to the given term in the corpus.

        Args:
            term (str): The term to find related terms for.
            n (int): The number of related terms to find (default is 5).

        Returns:
            list: A list of related terms to the given term. If the term is not found in the corpus, an empty list is returned.
        """
        if self.term_similarity_matrix is None:
            logger.warning("Related term have not been initialized. Call initialize_related_terms() first.")
            return []

        try:
            # Find the index of the given term in the feature names
            term_index = np.where(self.feature_names == term)[0][0]

            # Get the similarity scores of the given term with all other terms in the corpus
            term_similarities = self.term_similarity_matrix[term_index]

            # Sort the indices of the similarity scores in descending order
            similar_indices = term_similarities.argsort()[::-1]

            # Get the top n related terms (excluding the given term itself)
            similar_terms = [self.feature_names[i] for i in similar_indices[1:n+1]]
            return similar_terms
        except IndexError:
            logger.warning(f"Term '{term}' not found in the corpus.")
            return []
        
    def _ensure_label_diversity(self, labeled_topics):
        logger.info(f"Ensuring label diversity for {len(labeled_topics)} topics...")
        
        # Step 1: Extract main themes from labels
        main_themes = [label.split(':')[1].strip().split('-')[0].strip() for label, _ in labeled_topics]

        # Step 2: Check for duplicates
        if len(set(main_themes)) == len(main_themes):
            logger.info("Labels are already diverse.")
            return labeled_topics
        
        # Step 3: Initialize NLTK's WordNet

        # Step 4: Function to get synonyms
        def get_synonyms(word):
            synonyms = set()
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonyms.add(lemma.name().replace('_', ' '))
            return list(synonyms)
        
        # Step 5: Generate alternative labels
        diverse_labels = []
        used_words = set()

        for i, (label, topic_words) in enumerate(labeled_topics):
            main_theme = main_themes[1]
            words = main_theme.split()

            # Try to fin alternative words for the main theme
            alternative_words = []
            for word in words:
                for word in used_words:
                    synonyms = get_synonyms(word)
                    alternative = next((s for s in synonyms if s not in used_words), word)
                else:
                    alternative = word
                alternative_words.append(alternative)
                used_words.add(alternative)
            
            new_main_theme = ' '.join(alternative_words)
            new_label = f"Topic {i+1}: {new_main_theme} - {', '.join(topic_words[:5])}"
            diverse_labels.append((new_label, topic_words))

        # Step 6: Check if we've improved diversity
        new_main_themes = [label.split(':')[1].strip().split('-')[0].strip() for label, _ in diverse_labels]
        if len(set(new_main_themes)) > len(set(main_themes)):
            logger.info("Successfully diversified labels.")
            return diverse_labels
        else:
            logger.warning("Unable to diverse labels further.")
            return labeled_topics

    def _preprocess_for_labeling(self, words):
        if not isinstance(words, list):
            words = [words]
        return [word.lower() for item in words
                for word in (item if isinstance(item, tuple) else [item])
                if isinstance(word, str) and word.isalpha()]