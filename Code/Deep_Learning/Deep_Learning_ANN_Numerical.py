# IMPORT THE MODULES
import numpy as np
import pandas as pd
import spacy
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer


# READ THE FILE AS DATAFRAME
dataset = pd.read_csv("Datasets/amazon_reviews_training.csv")

# CONSIDER ONLY NUMERICAL FEATURES
features = ['VERIFIED_PURCHASE', 'REVIEW_LENGTH', 'RATING', 'SENTIMENT_SCORE',
            'TITLE_LENGTH', 'RATING_DEVIATION', 'NUM_REVIEWS',
            'COHERENT_ENCODED', 'AVG_WORD_LENGTH'
            'NUM_NOUNS', 'NUM_VERBS', 'NUM_ADJECTIVES', 'NUM_ADVERBS']

X = dataset[features].values
Y = dataset['LABEL_ENCODED'].values

# SPLIT DATASET INTO TRAINING AND TESTING SET
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(
    X, Y, test_size=0.2, random_state=42)

# FEATURE SCALING - NECESSARY FOR DEEP LEARNING
sc = StandardScaler()
X_TRAIN = sc.fit_transform(X_TRAIN)
X_TEST = sc.transform(X_TEST)

# INITIALISING THE ANN
ann = Sequential()
# ADDING HIDDEN LAYER
ann.add(Dense(units=16, activation='relu'))
# ADDING SECOND HIDDEN LAYER
ann.add(Dense(units=16, activation='relu'))
# ADDING OUTPUT LAYER
# ->1: BINARY CLASSIFICATION; SIGMOID: PROBABILITY
ann.add(Dense(units=1, activation='sigmoid'))

# COMPILE THE MODEL
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# FIT THE MODEL
ann.fit(X_TRAIN, Y_TRAIN, epochs=350, verbose=1, batch_size=32)

# PREDICT
threshold = 0.5
predictions = ann.predict(X_TEST)
predictions = predictions > threshold

# CLASSIFICATION REPORT
print(classification_report(Y_TEST, predictions))
# ACCURACY
accuracy = accuracy_score(Y_TEST, predictions)*100
print(f"ACCURACY = {accuracy}%")
