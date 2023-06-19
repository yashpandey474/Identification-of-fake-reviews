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

# READ FILE INTO DATAFRAME
df = pd.read_csv("Datasets/amazon_reviews_training.csv")

# ONLY NUMERICAL FEATURES CONSIDERED
X = df.drop(
    ["LABEL_ENCODED", "REVIEW_TEXT", 
             "REVIEW_TITLE"], axis = 1
            )
Y = df['LABEL_ENCODED']

# TRAIN-TEST SPLIT
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(
    X, Y, test_size=0.2, random_state=42
    )

# INSTANTIATE AND TRAIN THE KNN CLASSIFIER
knn = K(n_estimators=50)

knn.fit(X_TRAIN, Y_TRAIN)

# PREDICTIONS ON THE TESTING DATA
predictions = knn.predict(X_TEST)

# PRINT CLASSIFICATION REPORT
print(classification_report(Y_TEST, predictions))
