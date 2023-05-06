import math
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def tokenize(text):
    tokens = word_tokenize(text.lower())
    return [stemmer.stem(token) for token in tokens if token not in stop_words]

def calculate_tf(tokens):
    tf = defaultdict(int)
    for token in tokens:
        tf[token] += 1
    return tf

def calculate_idf(tokenized_documents):
    idf = defaultdict(float)
    N = len(tokenized_documents)
    for tokens in tokenized_documents:
        for token in set(tokens):
            idf[token] += 1
    for token, count in idf.items():
        idf[token] = math.log(N / count)
    return idf

def calculate_tfidf(tokens, idf):
    tfidf = defaultdict(float)
    tf = calculate_tf(tokens)
    for token, freq in tf.items():
        tfidf[token] = freq * idf[token]
    return tfidf

def search(query, documents, idf):
    tokenized_query = tokenize(query)
    query_tfidf = calculate_tfidf(tokenized_query, idf)
    document_tfidfs = []
    for document in documents:
        tokenized_document = tokenize(document)
        document_tfidf = calculate_tfidf(tokenized_document, idf)
        document_tfidfs.append(document_tfidf)
    scores = []
    for i, document_tfidf in enumerate(document_tfidfs):
        score = sum(query_tfidf.get(token, 0) * document_tfidf.get(token, 0) for token in set(tokenized_query) & set(document_tfidf))
        scores.append((i, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [i for i, score in scores]

