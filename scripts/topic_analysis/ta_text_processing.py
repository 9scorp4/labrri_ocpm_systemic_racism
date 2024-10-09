from loguru import logger
import concurrent.futures

from scripts.topic_analysis.tools import Tools
from exceptions import ProcessingError

class Process:
    def __init__(self, lang=None):
        logger.info('Initializing Process')
        self.lang = lang
        try:
            self.tools = Tools(lang)
            self.stopwords_lang = self.tools.stopwords_lang

            if lang == 'bilingual':
                self.nlp = {
                    'fr': self.tools.load_spacy('fr'),
                    'en': self.tools.load_spacy('en')
                }
            else:
                self.nlp = self.tools.load_spacy(lang)
        except Exception as e:
            logger.error(f"Failed to load spacy model. Error: {e}")
            raise e

    def single_sentence(self, sentence, lang):
        try:
            lang = lang or self.lang
            logger.debug(f"Processing sentence:\n{sentence}")

            text = sentence.text
            if not text:
                logger.warning("Skipping empty sentence")
                return None
            logger.debug("Sentence loaded!")

            logger.debug("Tokenizing...")
            tokens = self._tokenize(text, lang)

            logger.debug("Lemmatizing...")
            lemmatized_tokens = self._lemmatize(tokens, lang)

            lemmatized_sentence = ' '.join(lemmatized_tokens) if isinstance(lemmatized_tokens[0], str) else lemmatized_tokens

            logger.debug(f"Processed sentence:\n{lemmatized_sentence}\n")

            return lemmatized_sentence
        except Exception as e:
            logger.error(f"Failed to process sentence. Error: {e}")
            raise ProcessingError("Failed to process sentence", error=e)
        
    def _tokenize(self, text, lang):
        try:
            if lang == 'bilingual':
                fr_doc = self.nlp['fr'](text)
                en_doc = self.nlp['en'](text)
                return [token.text for token in fr_doc] + [token.text for token in en_doc]
            else:
                doc = self.nlp(text)
                return [token.text for token in doc]
        except Exception as e:
            logger.error(f"Failed to tokenize text. Error: {e}")
            raise ProcessingError("Failed to tokenize text", error=e)
        
    def _lemmatize(self, tokens, lang):
        try:
            return [self.tools.lemmatize(token, lang) for token in tokens if token.lower() not in self.stopwords_lang]
        except Exception as e:
            logger.error(f"Failed to lemmatize tokens. Error: {e}")
            raise ProcessingError("Failed to lemmatize tokens", error=e)
        
    def single_doc(self, doc, lang):
        try:
            lang = lang or self.lang
            sentences = self._split_sentences(doc, lang)
            processed_sentences = [self.single_sentence(sent, lang) for sent in sentences if sent]
            logger.debug("Document processed successfully!")
            return [sentence for sentence in processed_sentences if sentence]
        except Exception as e:
            logger.error(f"Failed to process document. Error: {e}")
            raise ProcessingError("Failed to process document", error=e)
        
    def _split_sentences(self, doc, lang):
        try:
            if lang == 'bilingual':
                fr_doc = self.nlp['fr'](doc)
                en_doc = self.nlp['en'](doc)
                return list(fr_doc.sents) + list(en_doc.sents)
            else:
                doc = self.nlp(doc)
                return list(doc.sents)
        except Exception as e:
            logger.error(f"Failed to split sentences. Error: {e}", exc_info=True)
            raise ProcessingError("Failed to split sentences", error=e)
        
    def docs_parallel(self, docs, lang, pbar=None):
        logger.info(f"Parallel processing documents with lang={lang}")
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                processed_docs = list(executor.map(lambda doc: self.single_doc(doc, lang), docs))

            post_processed_docs = self._postprocess_docs(processed_docs, lang)

            if pbar:
                pbar.update(len(post_processed_docs))

            return post_processed_docs
        except Exception as e:
            logger.error(f"Failed to process documents. Error: {e}")
            raise ProcessingError("Failed to process documents in parallel", error=e)
        
    def _postprocess_docs(self, processed_docs, lang):
        try:
            post_processed_docs = []
            for doc in processed_docs:
                if lang == 'bilingual':
                    filtered_sentences = self._filter_bilingual_sentences(doc)
                else:
                    filtered_sentences = self._filter_sentences(doc)
                post_processed_docs.extend(filtered_sentences)
            return post_processed_docs
        except Exception as e:
            logger.error(f"Failed to postprocess documents. Error: {e}")
            raise ProcessingError("Failed to postprocess documents", error=e)
        
    def _filter_bilingual_sentences(self, doc):
        return [word for word in doc if isinstance(word, list) or (word not in set(tuple(self.stopwords_lang)) and word not in tuple(self.stopwords_lang))]

    def _filter_sentences(self, doc):
        return [word for word in doc if (isinstance(word, dict) and tuple(word.items()) not in set(tuple(self.stopwords_lang.items())) and word not in self.stopwords_lang) or (isinstance(word, str) and word not in self.stopwords_lang)]