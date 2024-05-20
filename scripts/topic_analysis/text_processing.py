import logging
import concurrent.futures
from scripts.topic_analysis.tools import Tools

class Process:
    def __init__(self, lang=None):
        logging.info('Initializing Process')
        self.lang = lang
        try:
            self.tools = Tools(lang)
            self.stopwords_lang = self.tools.stopwords(lang)

            if lang == 'bilingual':
                self.nlp = {
                    'fr': self.tools.load_spacy('fr'),
                    'en': self.tools.load_spacy('en')
                }
            else:
                self.nlp = self.tools.load_spacy(lang)
        except Exception as e:
            logging.error(f"Failed to load spacy model. Error: {e}", exc_info=True)
            raise e

    def single_sentence(self, sentence, lang):
        try:
            lang = lang or self.lang
            logging.debug(f"Processing sentence:\n{sentence}")

            text = sentence.text
            if not text:
                logging.warning("Skipping empty sentence")
                return None
            logging.debug("Sentence loadad!")

            logging.debug("Tokenizing...")
            try:
                if lang == 'bilingual':
                    fr_doc = self.nlp['fr'](text)
                    en_doc = self.nlp['en'](text)
                    tokens = [token.text for token in fr_doc] + [token.text for token in en_doc]
                else:
                    doc = self.nlp(text)
                    tokens = [token.text for token in doc]
                logging.debug(f"Tokens before lemmatization:\n{tokens}")
            except Exception as e:
                logging.error(f"Failed to tokenize. Error: {e}", exc_info=True)
                return None
            
            logging.debug("Lemmatizing...")
            lemmatized_tokens = [self.tools.lemmatize(token, lang) for token in tokens if token.lower() not in self.stopwords_lang]

            if all(isinstance(token, list) for token in lemmatized_tokens):
                lemmatized_sentence = ' '.join(lemmatized_tokens)
            else:
                lemmatized_sentence = lemmatized_tokens

            logging.debug(f"Lemmatized sentence:\n{lemmatized_sentence}")
            logging.debug(f"Total tokens: {len(lemmatized_tokens)}")
            logging.debug(f"Processed sentence:\n{lemmatized_sentence}")

            return lemmatized_sentence
        except Exception as e:
            logging.error(f"Failed to process sentence. Error: {e}", exc_info=True)
            return None

    def single_doc(self, doc, lang):
        try:
            lang = lang or self.lang
            sentences = []

            if lang == 'bilingual':
                fr_doc = self.nlp['fr'](doc)
                en_doc = self.nlp['en'](doc)

                for sent in fr_doc.sents:
                    sentences.append(self.single_sentence(sent, 'fr'))
                for sent in en_doc.sents:
                    sentences.append(self.single_sentence(sent, 'en'))
            else:
                doc = self.nlp(doc)
                for sent in doc.sents:
                    sentences.append(self.single_sentence(sent, lang))
            
            processed_sentences = [sentence for sentence in sentences if sentence]
            logging.debug("Document processed successfully!")
            return processed_sentences
        except Exception as e:
            logging.error(f"Failed to process document. Error: {e}", exc_info=True)
            return []
        
    def docs_parallel(self, docs, lang, pbar=None):
        logging.info("Starting parallel processing...")
        postprocessed_docs = []

        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                processed_docs = list(executor.map(lambda doc: self.single_doc(doc, lang), docs))
                for doc in processed_docs:
                    if lang == 'bilingual':
                        if isinstance(doc, dict):
                            filtered_sentences = [word for word in doc.get(lang, []) if word not in self.stopwords_lang]
                        elif isinstance(doc, list):
                            filtered_sentences = [word for word in doc if isinstance(word, list) or (word not in set(tuple(self.stopwords_lang)) and word not in tuple(self.stopwords_lang))]
                    else:
                        filtered_sentences = [word for word in doc if (isinstance(word, dict) and tuple(word.items()) not in set(tuple(self.stopwords_lang.items())) and word not in self.stopwords_lang) or (isinstance(word, str) and word not in self.stopwords_lang)]

                    postprocessed_docs.extend(filtered_sentences)

                    if pbar:
                        pbar.update(len(filtered_sentences))
        except Exception as e:
            logging.error(f"Failed to process documents. Error: {e}", exc_info=True)
            return []
        
        return postprocessed_docs