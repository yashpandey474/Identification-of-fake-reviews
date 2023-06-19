#IMPORT MODULES
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

# READ THE FILE AS A DATAFRAME
df = pd.read_csv("Datasets/amazon_reviews_training.csv")

# DICTIONARY WITH NAME AND COMMAND TO INSTANTIATE DIFFERENT MODELS
classifiers = {}
classifiers.update({"XGBClassifier": XGBClassifier(eval_metric='logloss',
                                                   objective='binary:logistic',
                                                   use_label_encoder=False
                                                   )})
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

#GET THE FEATURES AND LABELS
print("THE DATASET CONTAINS FOLLOWING COLUMNS")
print(df.columns)
print('\n\n\n\n\n')

features = ['RATING', 'VERIFIED_PURCHASE', 'REVIEW_LENGTH',
            'TITLE_LENGTH', 'SENTIMENT_SCORE', 'COHERENT_ENCODED',
            'RATING_DEVIATION', 'READABILITY_FRE', 'NUM_NOUNS',
            'NUM_VERBS', 'NUM_ADJECTIVES', 'NUM_ADVERBS',
            'AVERAGE_RATING', 'NUM_REVIEWS', 'READABILITY_FRE'
            ]
X = df[features]
Y = df['LABEL_ENCODED']

# PERFORM THE TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=1, shuffle=True)

# FEATURE SCALING
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# CREATE A DATAFRAME OF MODELS WITH RUN TIME AND AUC SCORES
df_models = pd.DataFrame(
    columns=['model', 'run_time', 'accuracy', 'precision', 'f1_score'])

for key in classifiers:
    # STARTING TIME
    start_time = time.time()
    # CURRENT CLASSIFIER
    clf = classifiers[key]
    #TRAIN CLASSIFIER ON TRAINING DATA
    clf.fit(X_train_scaled, y_train)
    # MAKE PREDICTIONS USING CURRENT CLASSIFIER
    predictions = clf.predict(X_test_scaled)
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

df_models = df_models.sort_values(by='accuracy', ascending=False)

# PRINT THE MODELS WITH AUC SCORES [SORTED]
print(df_models)
