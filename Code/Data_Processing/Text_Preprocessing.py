from nltk.stem import PorterStemmer
import spacy
import pandas as pd


# INITIALISATIONS
nlp = spacy.load('en_core_web_sm')
stemmer = PorterStemmer()


# 1. TEXT PROCESSING


# REMOVE ALL PUNCTUATION
def separate_punc(doc_text):
    # GRAB THE TOKENS IF NOT A NEWLINE OR PUNCTUATION
    return [token.text.lower() for token in nlp(doc_text) if token.text not in '\n\n \n\n\n!"-#$%&()--.*+,-/:;<=>?@[\\]^_`{|}~\t\n ']


# REMOVE STOPWORDS - SPACY

stop_words = nlp.Defaults.stop_words


def remove_stopwords(text):
    words = text.lower().split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)


# STEMMING - NLTK
def stem_text(text):
    return ' '.join([stemmer.stem(word) for word in text.split()])


# LEMMATIZATION - SPACY
def lemmatize_text(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc])


df['REVIEW_TEXT'][:1000] = df['REVIEW_TEXT'][:1000].apply(
    lambda d: lemmatize_text(d))
