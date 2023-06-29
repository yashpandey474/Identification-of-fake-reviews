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
# REMOVE THE TEXTUAL FEATURES
del X['REVIEW_TEXT']
del X['REVIEW_TITLE']
Y = df['LABEL_ENCODED']

# TRAIN-TEST SPLIT [80-20]
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# SCALE ALL VALUES
scaler_object = MinMaxScaler()
scaler_object.fit(X_TRAIN)
# SCALED FEATURES
scaled_X_TRAIN = scaler_object.transform(X_TRAIN)
scaled_X_TEST = scaler_object.transform(X_TEST)

# INSTANTIATE DEEP LEARNING MODEL
model = Sequential()

# ADD LAYERS
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# COMPILE THE MODEL
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# FIT MODEL ON SCALED DATA
model.fit(scaled_X_TRAIN, Y_TRAIN, epochs=400, verbose=2)

# MAKE PREDICTIONS ON TESTING DATA
predictions = np.argmax(model.predict(scaled_X_TEST), axis=1)

# PRINT THE CLASSIFICATION REPORT
print(classification_report(Y_TEST, predictions))
