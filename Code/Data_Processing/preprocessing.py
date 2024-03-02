import spacy
import string

# STEPS OF DATA PRE-PROCESSING: NORMALIZATION, TOKENIZATION, REMOVAL OF STOP-WORDS, LEMMATIZATION

#1. NORMALIZATION

def remove_punctuation(text):

    # REMOVE PUNCTUATION MARKS
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    return text

def tokenize(text):
    # CONVERT TO ARRAY OF WORDS: ALSO HANDLES EMPTY STRINGS
    words = text.split()
    words = [word for word in words if word]
    return words

def remove_stopWords(text):
    words = text.split()
    
    en = spacy.load('en_core_web_sm')
    # ENGLISH STOP WORDS FROM SPACY MODULE
    sw_spacy = en.Defaults.stop_words
    # REMOVING STOP WORDS FROM ARRAY OF WORDS
    words = [word for word in words if word not in sw_spacy]

    return ' '.join(words)

def lemmatize_text(words):
    nlp = spacy.load('en_core_web_sm')

    # CREATE A NEW ARRAY OF WORDS
    lemmatized_words = []
    # FOR EACH WORD
    for word in words:
        # PROCESS TO LEMMATIZED FORM
        doc = nlp(word)
        # LEMMATIZED WORD FORMED
        lemmatized = " ".join([token.lemma_ for token in doc])
        # ADD TO NEW ARRAY
        lemmatized_words.append(lemmatized)
        
    return lemmatized_words

def preprocess_text(text):
    
    # CONVERSION INTO LOWER-CASE
    text = text.lower()
    
    #1. PUNCTUATION REMOVED
    text = remove_punctuation(text)

    #2. TOKENIZATION
    words = tokenize(text)

    # 3. REMOVAL OF STOP WORDS
    words = remove_stopWords(words)

    #4. LEMMATIZATION
    words = lemmatize_text(words)
    
    #5. CONVERT BACK TO TEXT
    text = ' '.join(words)
    
    return text

def preprocess_reviews(reviews):
    # PERFOMING DATA PREPROCESSING ON THE REVIEWS
    reviews_preprocessed = []
    for text in reviews:
        # PUNCTUATION REMOVED; STOP WORDS REMOVED; ALL WORDS LEMMATIZED
        text = preprocess_text(text)
        reviews_preprocessed.append(text)
    return reviews_preprocessed
