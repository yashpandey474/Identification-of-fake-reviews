#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import time
# SENTIMENT ANALYSIS USING VADER
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import LinearSVC, SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.linear_model import RidgeClassifier, SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer


# In[5]:


df = pd.read_csv("../Datasets/amazon_reviews_unlabelled.csv")


# In[6]:


df.columns


# In[7]:


df.dropna(inplace = True)


# In[8]:


#DEFINE RULE FOR DUPLICATE DETECTION
def find_duplicates(df):
    duplicates = []
    
    for i in range(len(df)):
        for j in range(i+1, len(df)):
            ngram1 = set(df.iloc[i]['NGRAMS'])
            ngram2 = set(df.iloc[j]['NGRAMS'])
            
            intersection = len(ngram1.intersection(ngram2))
            union = len(ngram1.union(ngram2)) + 1e-10
            
            similarity_score = float(intersection)/union

            if(similarity_score) > 0.95:
                print(f"SIMILARITY = {similarity_score*100}%")
                duplicates.append(i)
                duplicates.append(j)
    return set(duplicates)


# In[9]:


duplicates = find_duplicates(df[:100])


# In[10]:


len(duplicates)


# In[11]:


duplicates


# In[12]:


#ADD LABEL: DUPLICATE OR NOT [ALL CONSIDERED AS POSITIVELY FAKE]
df['DUPLICATE'] = [1 if index in duplicates else 0 for index in df.index]


# In[20]:


#TRAIN MODELS TO DETECT DUPLICATES BASED ON EXTRACTED FEATURES
# DICTIONARY WITH NAME AND COMMAND TO INSTANTIATE DIFFERENT MODELS
classifiers = {}
classifiers.update({"XGBClassifier": XGBClassifier(eval_metric='logloss',objective='binary:logistic')})
#classifiers.update({"CatBoostClassifier": CatBoostClassifier(silent=True)})
classifiers.update({"LinearSVC": LinearSVC(max_iter=10000)})
#classifiers.update({"MultinomialNB": MultinomialNB()})
#classifiers.update({"LGBMClassifier": LGBMClassifier()})
classifiers.update({"RandomForestClassifier": RandomForestClassifier()})
classifiers.update({"DecisionTreeClassifier": DecisionTreeClassifier()})
classifiers.update({"ExtraTreeClassifier": ExtraTreeClassifier()})
classifiers.update({"AdaBoostClassifier": AdaBoostClassifier()})
classifiers.update({"KNeighborsClassifier": KNeighborsClassifier()})
classifiers.update({"RidgeClassifier": RidgeClassifier()})
classifiers.update({"SGDClassifier": SGDClassifier()})
classifiers.update({"BaggingClassifier": BaggingClassifier()})
classifiers.update({"BernoulliNB": BernoulliNB()})
classifiers.update({"LogisticRegression": LogisticRegression()})
classifiers.update({"SVM": SVC()})


# In[21]:


features = [
    'RATINGS',
    'VERIFIED',  'MAX_REVIEWS_DAY',
    'HELPFUL_VOTES','REVIEW_SENTIMENT', 'AVERAGE_RATING',
    'RATING_DEVIATION', 'REVIEW_LENGTH', 'TITLE_LENGTH',
    'TOTAL_USER_REVIEWS',  'REVIEW_DATE_DIFF',
       'AVG_WORD_LENGTH', 'TOTAL_PRODUCT_REVIEWS', 'READABILITY_FRE',
       'CAPITAL_CHAR_COUNT', 'PUNCTUATION_COUNT', 'REVIEW_WORD_COUNT',
       'SENTIMENT_SCORE_TITLE', 'NUM_NAMED_ENTITIES', 'LEXICAL_DIVERSITY',
       'WORD_COUNT', 'RATING_CATEGORY', 'SENTIMENT_CATEGORY', 'COHERENCE',
        'TOTAL_VERIFIED_REVIEWS',
       'TOTAL_USER_HELPFUL_VOTES'
]
for col in features:
    df[col] = df[col].astype(int)


# In[22]:


#CURRENTLY TEST ONLY WITH NUMERICAL FEATURES
X = df[features][:100]

Y = df['DUPLICATE'][:100]


# In[23]:


X_train, X_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size = 0.2,
    random_state = 42
)


# In[24]:


# CREATE A DATAFRAME OF MODELS WITH RUN TIME AND AUC SCORES
df_models = pd.DataFrame(
    columns=['model', 'run_time', 'accuracy', 'precision', 'f1_score'])

print("TRAINING DATA AND TESTING DATA CREATED")
i = 1

for key in classifiers:
    # STARTING TIME
    start_time = time.time()
    # CURRENT CLASSIFIER
    clf = classifiers[key]
    # TRAIN CLASSIFIER ON TRAINING DATA
    clf.fit(X_train, y_train)
    # MAKE PREDICTIONS USING CURRENT CLASSIFIER
    predictions = clf.predict(X_test)
    # CALCULATE ACCURACY
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    f1score = f1_score(y_test, predictions)

    row = {'model': key,
           'run_time': format(round((time.time() - start_time)/60, 2)),
           'accuracy': accuracy,
           'precision': precision,
           'f1_score': f1score
           }

    df_models = df_models._append(row, ignore_index=True)
    print(f"{i} MODEL DONE")
    i+=1;

df_models = df_models.sort_values(by='accuracy', ascending=False)


# In[25]:


#PRINT THE MODELS
df_models


# In[ ]:




