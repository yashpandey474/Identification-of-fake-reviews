def calculate_jaccard_similarity(text1, text2):
    ngrams1 = list(ngrams(text1.split(), 2))
    ngrams2 = list(ngrams(text2.split(), 2))

    set1 = set(ngrams1)
    set2 = set(ngrams2)
    
    intersection = len(list(set1.intersection(set2)))
    union = len(list(set1.union(set2)))
    similarity_score = float(intersection)/union
    return similarity_score
