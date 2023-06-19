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
df = pd.read_csv("Datasets/amazon_reviews_training.csv")

# CONSIDER ONLY NUMERICAL FEATURES [NUMPY ARRAYS]
features_text = df['REVIEW_TEXT'].values
features_numeric = df[['REVIEW_LENGTH', 'TITLE_LENGTH',
                      'NUM_NOUNS', 'NUM_VERBS', 'NUM_ADVERBS',
                       'NUM_ADJECTIVES', 'TITLE_LENGTH', 'SENTIMENT_SCORE',
                       'VERIFIED_PURCHASE', 'RATING', 'RATING_DEVIATION']].values

labels = df['LABEL_ENCODED'].values

# TRAIN-TEST SPLIT
X_text_train, X_text_test, X_numeric_train, X_numeric_test, y_train, y_test = train_test_split(
    features_text, features_numeric, labels, test_size=0.2, random_state=42
)

# VEFCTORISE THE TEXTUAL FEATURES
tfidf = TfidfVectorizer()
X_text_train_tfidf = tfidf.fit_transform(X_text_train)
X_text_test_tfidf = tfidf.transform(X_text_test)

# Standardize the numeric features
scaler = StandardScaler()
X_numeric_train = scaler.fit_transform(X_numeric_train)
X_numeric_test = scaler.transform(X_numeric_test)

# Concatenate the TF-IDF matrix and numeric features
X_train = np.hstack((X_text_train_tfidf.toarray(), X_numeric_train))
X_test = np.hstack((X_text_test_tfidf.toarray(), X_numeric_test))

# INSTANTIATE THE DEEP LEARNING MODEL
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, activation='relu',
                          input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# COMPILE THE DEEP LEARNING MODEL
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# FIT THE MODEL ON TRAINING DATA
model.fit(X_train, y_train, verbose=1, epochs=400, batch_size=32)

# MAKE PREDICTIONS ON TESTING DATA
threshold = 0.5
predictions = model.predict(X_test)
predictions = predictions>threshold

# PRINT CLASSIFICATION REPORT
print(classification_report(y_test, predictions))

# PRINT ACCURACY OF PREDICTION
accuracy = accuracy_score(y_test, predictions)*100
print(f"Accuracy = {accuracy}%")
