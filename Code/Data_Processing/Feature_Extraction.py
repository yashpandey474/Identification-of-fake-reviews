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
df['READABILITY_FRE'] = df['REVIEW_TEXT'].apply(
    lambda d: flesch_reading_ease(d))

# 2. SENTIMENT CATEGORY & RATING CATEGORY
sentiment_threshold = 0.2
rating_threshold = 2.5

def assign_sentiment_category(score):
    if score > sentiment_threshold:
        return 'positive'
    else:
        return 'negative'

def assign_rating_category(rating):
    if rating > rating_threshold:
        return 'positive'

    else:
        return 'negative'


df['SENTIMENT_CATEGORY'] = df['SENTIMENT_SCORE'].apply(
    assign_sentiment_category)
df['RATING_CATEGORY'] = df['RATING'].apply(assign_rating_category)

#3. COHERENCE COLUMN [BETWEEN RATING AND TEXT]
df['COHERENT'] = df['SENTIMENT_CATEGORY'] == df['RATING_CATEGORY']

#4. VADER SENTIMENT SCORE
sid = SentimentIntensityAnalyzer()

df['SENTIMENT_SCORE'] = df['REVIEW_TEXT'].apply(
    lambda d: sid.polarity_scores(d)['compound'])

#5. TITLE LENGTH 
df['TITLE_LENGTH'] = df['REVIEW_TITLE'].apply(lambda d: len(d))


# 6. POS TAGS: VERBS, NOUNS, ADJECTIVES, ADVERBS
def count_nouns(Pos_counts):
    noun_count = Pos_counts.get(spacy.parts_of_speech.NOUN, 0)
    return noun_count


def count_verbs(Pos_counts):
    verb_count = Pos_counts.get(spacy.parts_of_speech.VERB, 0)
    return verb_count


def count_adjectives(Pos_counts):
    adjective_count = Pos_counts.get(spacy.parts_of_speech.ADJ, 0)
    return adjective_count


def count_adverbs(Pos_counts):
    adverb_count = Pos_counts.get(spacy.parts_of_speech.ADV, 0)
    return adverb_count


def pos_counts(text):
    doc = nlp(text)
    Pos_counts = doc.count_by(spacy.attrs.POS)
    return Pos_counts

df['NUM_NOUNS'] = df['REVIEW_TEXT'].apply(count_nouns)
df['NUM_VERBS'] = df['REVIEW_TEXT'].apply(count_verbs)
df['NUM_ADJECTIVES'] = df['REVIEW_TEXT'].apply(count_adjectives)
df['NUM_ADVERBS'] = df['REVIEW_TEXT'].apply(count_adverbs)

# 7. AVERAGE RATING OF PRODUCT
average_ratings = df.groupby('PRODUCT_ID')['RATING'].mean()

df['AVERAGE_RATING'] = df['PRODUCT_ID'].map(average_ratings)

# 8. RATING DEVIATION FROM AVERAGE RATIONG
df['RATING_DEVIATION'] = abs(df['RATING']-df['AVERAGE_RATING'])

# 9. TOTAL NUMBER OF REVIEWS FOR PRODUCT
num_reviews = df.groupby('PRODUCT_ID').size()

df['NUM_REVIEWS'] = df['PRODUCT_ID'].map(num_reviews)


# 10. NUMBER OF TIMES ENTITIES WERE MENTIONED
def count_ent(text):
    doc = nlp(text)
    ent_count = len([ent.text for ent in doc.ents])
    return ent_count


df['NUM_NAMED_ENTITIES'] = df['REVIEW_TEXT'].apply(count_ent)


#11. LENGTH OF REVIEW TEXT
df['REVIEW_LENGTH'] = df['REVIEW_TEXT'].apply(lambda d: len(d))

#12. MAXIMUM COSINE SIMILARITY WITH ANOTHER REVIEW

tfidfvectoriser = TfidfVectorizer()
tfidf_matrix = tfidfvectoriser.fit_transform(df['REVIEW_TEXT'])

# CREATE COSINE SIMILARITY MATRIX
cosine_similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# FIND MAX SIMILARITIES
max_similarities = []

for i, row in enumerate(cosine_similarity_matrix):
    # ALL OTHER BUT WITH ITSELF [ALWAYS 1]
    max_similarity = max(row[:i].tolist() + row[i+1:].tolist())
    max_similarities.append(max_similarity)
    
df['MAX_SIMILARITY'] = max_similarities


