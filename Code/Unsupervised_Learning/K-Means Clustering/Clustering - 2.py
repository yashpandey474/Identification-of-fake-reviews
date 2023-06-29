# UNSUPERVISED LEARNING FOR THE 2ND DATASET
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

#DATA FRAME FOR DATASET
df = pd.read_csv('amazon_reviews2.csv',sep=',')

# LABEL NOT PRESENT IN THIS DATASET [ONLY USING VALID FEATURES]
X = df[['title_length', 'review_length', 'total_user_reviews','review_date_diff','total_reviews','rating_deviation','verified', 'ratings', 'review_sentiment', 'max_reviews_day']]

# FEATURE SCALING
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# NUMBER OF CLUSTERS TO DIVIDE INTO
n_clusters = 2

# CREATE AND FIT THE K-MEANS MODEL
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X_scaled)

# GET THE CLUSTER LABELS
cluster_labels = kmeans.labels_
df['cluster_label'] = cluster_labels

# FIND THE FREQUENCIES FOR CLUSTERS [NO WAY TO VERIFY]
df['cluster_label'].value_counts()
