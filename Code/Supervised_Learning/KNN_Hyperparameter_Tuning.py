from sklearn.neighbors import KNeighborsClassifier

# IMPORT MODULES
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
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
from scipy import sparse

# HYPERPARAMETER TUNING TO FIND MOST EFFICIENT ARGUMENT FOR KNN
from sklearn.model_selection import GridSearchCV

# READ FILE INTO DATAFRAME
df = pd.read_csv("Datasets/amazon_reviews_training.csv")

# ONLY NUMERICAL FEATURES CONSIDERED
X = df.drop(
    ["LABEL_ENCODED", "REVIEW_TEXT",
     "REVIEW_TITLE"], axis=1
)
Y = df['LABEL_ENCODED']

# TRAIN-TEST SPLIT
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

#RANGE TO SEARCH FOR VALUE OF K
param_grid = {'n_neighbors': range(1,25)}
# KNN CLASSIFIER
knn_classifier = KNeighborsClassifier()
# GRID SEARCH OVER THE PARAMETER RANGE [cross-validation]
grid_search = GridSearchCV(knn_classifier, param_grid, cv=5)
grid_search.fit(X_TRAIN, Y_TRAIN)

# BEST NUMBER OF NEIGHBORS
best_n_neighbors = grid_search.best_params_['n_neighbors']
# NEW KNN CLASSIFIER
best_knn_classifier = KNeighborsClassifier(n_neighbors=best_n_neighbors)
# FIT ON TRAINING DATA
best_knn_classifier.fit(X_TRAIN, Y_TRAIN)

# MAKE PREDICTIONS ON TESTING DATA
predictions = best_knn_classifier.predict(X_TEST)

# CLASSIFICATION REPORT
print(classification_report(Y_TEST, predictions))
