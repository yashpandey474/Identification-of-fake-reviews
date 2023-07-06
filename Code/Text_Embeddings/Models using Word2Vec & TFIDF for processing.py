#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import numpy as np
import os
import time
# SENTIMENT ANALYSIS USING VADER
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import LinearSVC, SVC
from gensim.models import Word2Vec
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.linear_model import RidgeClassifier, SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA


# In[3]:


df = pd.read_csv("Datasets/amazon_reviews_labelled.csv")


# In[6]:


features_text = df['PREPROCESSED_REVIEW_TEXT']
features_numeric = df[['RATING', 'VERIFIED_PURCHASE',
                       'NUM_NOUNS', 'NUM_VERBS', 'NUM_ADJECTIVES',
                       'NUM_ADVERBS', 'REVIEW_LENGTH', 'SENTIMENT_SCORE', 'TITLE_LENGTH',
                       'AVERAGE_RATING', 'RATING_DEVIATION', 'NUM_REVIEWS', 'READABILITY_FRE',
                       'SENTIMENT_CATEGORY_ENCODED', 'RATING_CATEGORY_ENCODED',
                       'COHERENT_ENCODED', 'AVG_WORD_LENGTH',
                       'NUM_NAMED_ENTITIES', 'CAPITAL_CHAR_COUNT', 'PUNCTUATION_COUNT'
                       ]]

labels = df['LABEL_ENCODED']


# In[44]:


# TRAIN-TEST SPLIT
X_text_train, X_text_test, X_numeric_train, X_numeric_test, y_train, y_test = train_test_split(
    features_text, features_numeric, labels, test_size=0.2, random_state=42
)


# In[23]:


#FUNCTION FOR COMBINING WORD2VEC AND TFIDF

def vectorize_text(X_text_train, X_text_test):
    #TOKENIZE THE TEXT
    tokenized_text_train = [t.split() for t in X_text_train]
    tokenized_text_test = [t.split() for t in X_text_test]
    
    #LOAD PRE-TRAINED WORD2VEC MODEL
    model_path = "word2vec_model.bin"
    if os.path.isfile(model_path):
        w2v_model = Word2Vec.load(model_path)
    
    else:
        #TRAIN THE WORD2VEC MODEL
        w2v_model = Word2Vec(sentences=tokenized_text_train,vector_size=100,window=5,min_count=1)
        #SAVE THE MODEL
        w2v_model.save("word2vec_model.bin")
        
    #W2V VECTORISATION
    w2v_vectors_train = []
    
    for review in tokenized_text_train:
        review_vectors = [w2v_model.wv[word] for word in review if word in w2v_model.wv]
        
        #AVERAGE METHOD
        if len(review_vectors) > 0:
            review_vector = np.mean(review_vectors, axis=0)  # Average the word vectors
            w2v_vectors_train.append(review_vector)
            
    #W2V VECTORISATION
    w2v_vectors_test = []
    
    for review in tokenized_text_test:
        review_vectors = [w2v_model.wv[word] for word in review if word in w2v_model.wv]
        
        #AVERAGE METHOD
        if len(review_vectors) > 0:
            review_vector = np.mean(review_vectors, axis=0)  # Average the word vectors
            w2v_vectors_test.append(review_vector)
            
    #VECTORISE USING TF-IDF
    vect = TfidfVectorizer()
    tfidf_vectors_train = vect.fit_transform(X_text_train).toarray()
    tfidf_vectors_test = vect.transform(X_text_test).toarray()
    
    
    #CONCATENATE
    final_vectors_train = np.concatenate(
        (w2v_vectors_train, tfidf_vectors_train),
        axis = 1 #COMBINE COLUMNS -> HORIZONTAL
    )
    final_vectors_test = np.concatenate(
        (w2v_vectors_test, tfidf_vectors_test),
        axis = 1 #COMBINE COLUMNS -> HORIZONTAL
    )
        
    return final_vectors_train, final_vectors_test


# In[45]:


X_text_train, X_text_test = vectorize_text(X_text_train, X_text_test)


# In[46]:


#APPLY FEATURE SCALING TO MATRICES
sc_numeric = MinMaxScaler()
X_numeric_train = sc_numeric.fit_transform(X_numeric_train)
X_numeric_test = sc_numeric.transform(X_numeric_test)

sc_text= MinMaxScaler()
X_text_train = sc_text.fit_transform(X_text_train)
X_text_test = sc_text.transform(X_text_test)


# In[ ]:


#APPLY DIMENSIONALITY REDUCTION TO TEXTUAL MATRICES [LARGE NUMBER OF COLUMNS]
components = 5000
pca = PCA(n_components=components)

X_text_train = pca.fit_transform(X_text_train)
X_text_test = pca.transform(X_text_test)


# In[ ]:


#CONCATENATE THE TEXTUAL FEATURE AND NUMERIC FEATURE MATRIX
X_train = np.hstack((X_text_train, X_numeric_train))
X_test = np.hstack((X_text_test, X_numeric_test))


# In[29]:


# DICTIONARY WITH NAME AND COMMAND TO INSTANTIATE DIFFERENT MODELS
classifiers = {}
classifiers.update({"XGBClassifier": XGBClassifier(eval_metric='logloss',objective='binary:logistic',use_label_encoder=False)})
classifiers.update({"CatBoostClassifier": CatBoostClassifier(silent=True)})
classifiers.update({"LinearSVC": LinearSVC(max_iter=10000)})
classifiers.update({"MultinomialNB": MultinomialNB()})
classifiers.update({"LGBMClassifier": LGBMClassifier()})
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


# In[ ]:


#TRAIN 1 MODEL
clf = LogisticRegression(max_iter = 10000)
clf.fit(X_train, y_train)

#MAKE PREDICTIONS
predictions = clf.predict(X_test)
print(classification_report(y_test, predictions))


# In[ ]:




