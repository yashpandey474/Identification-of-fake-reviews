import pandas as pd
import numpy as np
# FRE SCORE
from textstat import flesch_reading_ease
# SENTIMENT ANALYSIS USING VADER
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#POS TAGGING
import spacy
#VECTORISING TEXT AND CREATING PIPELINE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
#COSINE SIMILARITY BETWEEN REVIEWS
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd
import numpy as np

nlp = spacy.load('en_core_web_sm')
df = pd.read_csv("amazon_reviews_training.csv")

# 1. FRE READABILITY SCORE
def add_readability_score(df):
    df['READABILITY_FRE'] = df['REVIEW_TEXT'].apply(
        lambda d: flesch_reading_ease(d))

# 2. SENTIMENT CATEGORY & RATING CATEGORY
sentiment_threshold = 0.2
rating_threshold = 2.5


def add_sentiment_category(df, threshold):

    def assign_sentiment_category(score):
        if score > threshold:
            return 'positive'
        else:
            return 'negative'

    df['SENTIMENT_CATEGORY'] = df['SENTIMENT_SCORE'].apply(
        assign_sentiment_category)


def add_rating_category(df, threshold):

    def assign_rating_category(rating):
        if rating > threshold:
            return 'positive'
        else:
            return 'negative'

    df['RATING_CATEGORY'] = df['RATING'].apply(assign_rating_category)

#3. COHERENCE COLUMN [BETWEEN RATING AND TEXT]
def add_coherence_column(df):
    df['COHERENT'] = df['SENTIMENT_CATEGORY'] == df['RATING_CATEGORY']

#4. VADER SENTIMENT SCORE
def add_vader_sentiment_score(df):
    sid = SentimentIntensityAnalyzer()

    df['SENTIMENT_SCORE'] = df['REVIEW_TEXT'].apply(
        lambda d: sid.polarity_scores(d)['compound'])

#5. TITLE LENGTH 
def add_title_length(df):
    df['TITLE_LENGTH'] = df['REVIEW_TITLE'].apply(lambda d: len(d))


# 6. POS TAGS: VERBS, NOUNS, ADJECTIVES, ADVERBS
def add_pos_tags(df):
    def count_pos(Pos_counts, pos_type):
        pos_count = Pos_counts.get(pos_type, 0)
        return pos_count

    def pos_counts(text):
        doc = nlp(text)
        Pos_counts = doc.count_by(spacy.attrs.POS)
        return Pos_counts

    poscounts =  df['REVIEW_TEXT'].apply(pos_counts)
    df['NUM_NOUNS'] = df['REVIEW_TEXT'].apply(
        lambda text: count_pos(poscounts, spacy.parts_of_speech.NOUN))
    df['NUM_VERBS'] = df['REVIEW_TEXT'].apply(
        lambda text: count_pos(poscounts, spacy.parts_of_speech.VERB))
    df['NUM_ADJECTIVES'] = df['REVIEW_TEXT'].apply(
        lambda text: count_pos(poscounts, spacy.parts_of_speech.ADJ))
    df['NUM_ADVERBS'] = df['REVIEW_TEXT'].apply(
        lambda text: count_pos(poscounts, spacy.parts_of_speech.ADV))

# 7. AVERAGE RATING OF PRODUCT
def add_average_rating(df):
    average_ratings = df.groupby('PRODUCT_ID')['RATING'].mean()
    df['AVERAGE_RATING'] = df['PRODUCT_ID'].map(average_ratings)

# 8. RATING DEVIATION FROM AVERAGE RATIONG
def add_rating_deviation(df):
    df['RATING_DEVIATION'] = abs(df['RATING'] - df['AVERAGE_RATING'])

# 9. TOTAL NUMBER OF REVIEWS FOR PRODUCT
def add_total_reviews(df):
    num_reviews = df.groupby('PRODUCT_ID').size()
    df['NUM_REVIEWS'] = df['PRODUCT_ID'].map(num_reviews)


# 10. NUMBER OF TIMES ENTITIES WERE MENTIONED

def add_named_entities(df):
    def count_entities(text):
        doc = nlp(text)
        ent_count = len([ent.text for ent in doc.ents])
        return ent_count

    df['NUM_NAMED_ENTITIES'] = df['REVIEW_TEXT'].apply(count_entities)


#11. LENGTH OF REVIEW TEXT

def add_review_length(df):
    df['REVIEW_LENGTH'] = df['REVIEW_TEXT'].apply(lambda d: len(d))

#12. MAXIMUM COSINE SIMILARITY WITH ANOTHER REVIEW


def add_max_similarity(df):
    tfidfvectoriser = TfidfVectorizer()
    tfidf_matrix = tfidfvectoriser.fit_transform(df['REVIEW_TEXT'])

    cosine_similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    max_similarities = []
    for i, row in enumerate(cosine_similarity_matrix):
        max_similarity = max(row[:i].tolist() + row[i+1:].tolist())
        max_similarities.append(max_similarity)

    df['MAX_SIMILARITY'] = max_similarities


