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

# READ THE FILE INTO A DATAFRAME
df = pd.read_csv("Datasets/amazon_reviews_training.csv")
df = df.dropna()

# FEATURES AND LABELS
X = df.drop("LABEL_ENCODED", axis=1)
Y = df['LABEL_ENCODED']

# TRAIN-TEST SPLIT [80-20]
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(
    X, Y, test_size=0.2, random_state=42
    )

# TRAIN THE MODEL
classifier = LogisticRegression()
classifier.fit(X_TRAIN, Y_TRAIN)

# PREDICTIONS ON THE TEST DATA
predictions = classifier.predict(X_TEST)

# CHECK PRECISION, RECALL, F1 SCORE
print(classification_report(Y_TEST, predictions))
