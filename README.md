# Identification-of-fake-reviews

# Project Summary
This project aims to implement machine learning to suitable identify fake reviews. We use the following approach for the project

1. __Feature Extraction__: From the initial features of the dataset used: RATING, VERIFIED_PURCHASE, PRODUCT_ID, PRODUCT_CATEGORY, REVIEW_TEXT, REVIEW_TITLE, we have extracted the following features to use for training various machine learning models:
   
     I.  FRE Readability Score (READABILITY_FRE): This feature calculates the Flesch Reading Ease score for each review text. The Flesch Reading Ease score is a measure of how easy or difficult it is to read a piece of text. Higher scores indicate easier readability.

      II.  Sentiment Category (SENTIMENT_CATEGORY) and Rating Category (RATING_CATEGORY): These features assign a sentiment category and rating category to each review based on predefined thresholds. The sentiment category is determined based on the sentiment score using the VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analysis tool, and the rating category is determined based on the numeric rating. Scores above the sentiment threshold are categorized as "positive," while scores below are categorized as "negative." Ratings above the rating threshold are categorized as "positive," and ratings below are categorized as "negative."

      III. Coherence (COHERENT): This feature checks if the sentiment category and rating category are the same, indicating whether the sentiment expressed in the text aligns with the numeric rating given.

      IV. VADER Sentiment Score (SENTIMENT_SCORE): This feature calculates the sentiment score using the VADER sentiment analysis tool. The sentiment score represents the overall sentiment of the review text, ranging from -1 (negative) to 1 (positive).

      V. Title Length (TITLE_LENGTH): This feature calculates the length of the review title in terms of the number of characters.

      VI. Part-of-Speech (POS) Tags: These features count the number of verbs (NUM_VERBS), nouns (NUM_NOUNS), adjectives (NUM_ADJECTIVES), and adverbs (NUM_ADVERBS) in the review text. Part-of-speech tagging is a linguistic task that labels words with their corresponding part of speech.

      VII. Average Rating (AVERAGE_RATING): This feature calculates the average rating of the product based on all the reviews.

      IX. Rating Deviation (RATING_DEVIATION): This feature calculates the absolute difference between the rating given in the review and the average rating of the product. It measures how much the rating deviates from the average rating.

      X.  Number of Reviews (NUM_REVIEWS): This feature counts the total number of reviews for each product.

      XI. Number of Named Entities (NUM_NAMED_ENTITIES): This feature counts the number of named entities mentioned in the review text. Named entities are specific entities such as names of people, organizations, locations, etc.

      XII. Review Length (REVIEW_LENGTH): This feature calculates the length of the review text in terms of the number of characters.

      XIV. Maximum Cosine Similarity (MAX_SIMILARITY): This feature calculates the maximum cosine similarity between the review text and all other reviews. Cosine similarity measures the similarity between two vectors, in this case, the vector representations of the review texts. The maximum similarity indicates how similar the review text is to other reviews.

2. __Data Pre-Processing__: The features can be divided into textual and behavioural features. The behavioural features are numerical or binary. We follow the following technique for preprocessing
   
      I. Textual Features [REVIEW_TEXT, REVIEW_TITLE] :

         A. Stop-words cleaning
         B. Stemming
         C. Lemmatisation
         D. TF-IDF Vectorization: Using sklearn's TfidfVectorizer to convert the review text and review title text into matrix of TF-IDF scores for unique words for each review

     II. Categorical Features [Eg - LABEL]:
         A. If only two values, such as LABEL [Fake/Not Fake], use LabelEncoder() to encode into numerical values 0/1
         B. If more than two values [none here]: Convert into one-hot encoded values

     III. Other numerical features:
         A. Standard Scaling [after train-test split]: Using StandardScaler, fit and transform the features for training [X_train] and only transform the features used for testing [x_test]

3. __Train-Test Split__: The features and labels are divided into matrices: X_train, X_test, Y_train, Y_test for training and testing the model

4. __Train The ML Model__

5. __Evaluate the model__: Get the predictions on the scaled testing feature values and compare with the actual labels (y_test) in the following ways:
       A. Accuracy_score
       B. Classification_report
       C. Precision_score

     
