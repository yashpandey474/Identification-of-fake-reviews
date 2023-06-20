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
#EVALUATE THE MODEL
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[2]:


df = pd.read_csv("Datasets/amazon_reviews_2.csv")


# In[3]:


del df['Unnamed: 0.1']
del df['Unnamed: 0']
del df['Unnamed: 0.2']
df.columns


# In[97]:


features = ['RATING', 'VERIFIED_PURCHASE',
        'NUM_NOUNS', 'NUM_VERBS', 'NUM_ADJECTIVES',
       'NUM_ADVERBS', 'REVIEW_LENGTH', 'SENTIMENT_SCORE', 'TITLE_LENGTH',
       'AVERAGE_RATING', 'RATING_DEVIATION', 'NUM_REVIEWS', 'READABILITY_FRE',
       'SENTIMENT_CATEGORY_ENCODED', 'RATING_CATEGORY_ENCODED',
       'COHERENT_ENCODED', 'AVG_WORD_LENGTH',
       'NUM_NAMED_ENTITIES', 'CAPITAL_CHAR_COUNT', 'PUNCTUATION_COUNT'
                   ]


# In[98]:


X = df[features]
Y = df['LABEL_ENCODED']

sc = StandardScaler()
X_scaled = sc.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y,
                                                   test_size = 0.2,
                                                   random_state = 42)
classifier = LogisticRegression()
classifier.fit(X_train, Y_train)


# In[99]:


#MAKE PREDICTIONS
predictions = classifier.predict(X_test)
print(classification_report(Y_test, predictions))


# In[ ]:




