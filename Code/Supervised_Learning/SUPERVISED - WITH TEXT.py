#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
#SCALE THE NUMERICAL DATA
from sklearn.preprocessing import StandardScaler
#TRAIN-TEST SPLIT
from sklearn.model_selection import train_test_split
#TRAIN THE ML MODEL
from sklearn.linear_model import LogisticRegression
#TFIDF VECTORISER
from sklearn.feature_extraction.text import TfidfVectorizer
#EVALUATE THE MODEL
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[2]:


df = pd.read_csv("Datasets/amazon_reviews_2.csv")


# In[3]:


features_text = df['REVIEW_TEXT']
features_numeric = df[['RATING', 'VERIFIED_PURCHASE',
        'NUM_NOUNS', 'NUM_VERBS', 'NUM_ADJECTIVES',
       'NUM_ADVERBS', 'REVIEW_LENGTH', 'SENTIMENT_SCORE', 'TITLE_LENGTH',
       'AVERAGE_RATING', 'RATING_DEVIATION', 'NUM_REVIEWS', 'READABILITY_FRE',
       'SENTIMENT_CATEGORY_ENCODED', 'RATING_CATEGORY_ENCODED',
       'COHERENT_ENCODED', 'AVG_WORD_LENGTH',
       'NUM_NAMED_ENTITIES', 'CAPITAL_CHAR_COUNT', 'PUNCTUATION_COUNT'
                      ]]

labels = df['LABEL_ENCODED']

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


# In[6]:


classifier = LogisticRegression()
classifier.fit(X_train, y_train)

#MAKE PREDICTIONS
predictions = classifier.predict(X_test)
print(classification_report(y_test, predictions))


# In[ ]:




