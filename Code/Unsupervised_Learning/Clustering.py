# IMPORT MODULES
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, FunctionTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

from keras.models import Sequential, load_model
from keras.layers import Dense

import numpy as np
import pandas as pd
import spacy

#READ THE FILE INTO A DATAFRAME
df = pd.read_csv("Datasets/amazon_reviews_training.csv")
df = df.dropna()

#FEATURES AND LABELS
X = df.drop("LABEL_ENCODED", axis = 1)
Y = df['LABEL_ENCODED']


#PIPELINE FOR VECTORISING AND SCALING
text_pipeline = Pipeline([
    ('selector', FunctionTransformer(
        lambda x: x['REVIEW_TEXT', 'REVIEW_TITLE'], validate=False)),
    ('tfidf', TfidfVectorizer()),
])
numeric_pipeline = Pipeline([
    ('selector', FunctionTransformer(
        lambda x: x[['NUM NOUNS', 'NUM VERBS', 'NUM ADVERBS', 'NUM ADJECTIVES', 'VERIFIED_PURCHASE', 'RATING', 'REVIEW_LENGTH', 'SENTIMENT SCORE']], validate=False)),
    ('scaler', StandardScaler()),
])

feature_union = FeatureUnion([
    ('text_features', text_pipeline),
    ('numeric_features', numeric_pipeline)
])

#PROCESS FEATURE DATA


# NUMBER OF CLUSTERS
n_clusters = 5

# CREATE AND FIT THE K-MEANS MODEL
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X)


# GET THE CLUSTER LABELS
cluster_labels = kmeans.labels_

#NEW COLUMN IN DATASET
df['cluster_label'] = cluster_labels
#CHECK THE VALUES AND FREQUENCIES
df['cluster_label'].value_counts()


# EXTRACT THE COUNTS OF FAKE AND NON-FAKE FOR DIFFERENT CLUSTERS
label_counts = df1.groupby('cluster_label')[
    'LABEL'].value_counts().unstack(fill_value=0)
fake_counts = label_counts['Fake']
not_fake_counts = label_counts['Not Fake']

#PRINT THE COUNTS
print("Fake Counts:")
print(fake_counts)
print("\nNot Fake Counts:")
print(not_fake_counts)
