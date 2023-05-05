import json
import openai
import numpy as np
import pandas as pd
import re
from openai.embeddings_utils import cosine_similarity
import transformers
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
from sklearn.feature_extraction.text import strip_accents_unicode, strip_tags
from sklearn.feature_extraction.text import _analyze
from fuzzywuzzy import fuzz, process
import logging
import time


openai.api_key = "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"
logging.getLogger("transformers").setLevel(logging.CRITICAL)
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


def load_openai_classes_and_embeddings():
    """Loads the courses_normalized.json file and computes the embeddings for each course. Saves the embeddings to a new file called courses_embeddings.json."""

    with open('courses_normalized.json', 'r') as f:
        courses = json.load(f)

    count = 1
    for course in courses:
        if count > 10:
            break
        try:
            course["combined"] = "Title: " + course["Name"] + "; Description: " + course["Desc"] + "; Professors: " + \
                course["Profs"] + "; StartTime: " + \
                course["StartTime"] + "; EndTime: " + course["EndTime"]
            course["embedding"] = openai.Embedding.create(
                input=course["combined"], engine="text-embedding-ada-002")["data"][0]["embedding"]
            print("Embedding number " + str(count) + " computed successfully!")
            count += 1
        except:
            continue

    with open('courses_embeddings.json', 'w') as f:
        json.dump(courses, f)


def load_bert_classes_and_embeddings():
    """Loads the courses_normalized.json file and computes the embeddings for each course. Saves the embeddings to a new file called courses_embeddings.json."""

    with open('courses_normalized.json', 'r') as f:
        courses = json.load(f)

    count = 1
    for course in courses:
        try:
            course["combined"] = "Title: " + course["Name"] + "; Description: " + course["Desc"] + "; Professors: " + \
                course["Profs"] + "; StartTime: " + \
                course["StartTime"] + "; EndTime: " + course["EndTime"]

            course["embedding"] = bert_encode(course["combined"])

            print("Embedding number " + str(count) + " computed successfully!")
            count += 1
        except:
            continue

    with open('courses_bert_embeddings.json', 'w') as f:
        json.dump(courses, f)


def calculate_similarity_openai(query, query_embedding):
    """Calculates the cosine similarity between the query and the query embedding."""

    try:
        similarity = cosine_similarity(
            np.array(query), np.array(query_embedding))
        if isinstance(similarity, np.ndarray):
            return similarity[0]
        else:
            return similarity
    except Exception as e:
        print(f"Error calculating similarity: {e}")
        return -1


def calculate_similarity_bert(query, query_embedding):
    """Calculates the cosine similarity between the query and the query embedding."""

    try:
        query = np.array(query) if not isinstance(query, np.ndarray) else query
        query_embedding = np.array(query_embedding) if not isinstance(
            query_embedding, np.ndarray) else query_embedding

        similarity = cosine_similarity(query, query_embedding)
        if isinstance(similarity, np.ndarray):
            return similarity[0]
        else:
            return similarity
    except:
        pass


def bert_search(query, pprint=True, n=3):
    """Searches the courses_embeddings.json file for the query and returns the top n results sorted by cosine similarity using BERT."""

    df = pd.read_json('courses_embeddings.json')
    query_embedding = bert_encode(query)

    df["similarity"] = df["embedding"].apply(
        lambda x: calculate_similarity_bert(x, query_embedding))

    result = df.sort_values("similarity", ascending=False).head(n)

    if pprint:
        for _, r in result.iterrows():
            print("Course Name: " + re.sub(r'<\/?p>', '', str(r["Name"])))
            print("Description: " + re.sub(r'<\/?p>', '', str(r["Desc"])))
            print("Professors: " + re.sub(r'<\/?p>', '', str(r["Profs"])))
            print("Start Time: " + re.sub(r'<\/?p>', '', str(r["StartTime"])))
            print("End Time: " + re.sub(r'<\/?p>', '', str(r["EndTime"])))
            print()

    return result


def fuzzy_search(query, pprint=True, n=3):
    """Searches the courses_combined.json file with a fuzzy search and returns the top n results."""

    df = pd.read_json('courses_combined.json')

    df["similarity"] = df["combined"].apply(
        lambda x: fuzz.token_sort_ratio(query, x))

    result = df.sort_values("similarity", ascending=False).head(n)

    if pprint:
        for _, r in result.iterrows():
            print("Course Name: " + re.sub(r'<\/?p>', '', str(r["Name"])))
            print("Description: " + re.sub(r'<\/?p>', '', str(r["Desc"])))
            print("Professors: " + re.sub(r'<\/?p>', '', str(r["Profs"])))
            print("Start Time: " + re.sub(r'<\/?p>', '', str(r["StartTime"])))
            print("End Time: " + re.sub(r'<\/?p>', '', str(r["EndTime"])))
            print()

    return result


def tfidf_search(query, pprint=True, n=3):
    """Searches the courses_normalized.json file for the query and returns the top n results sorted by cosine similarity using tf-idf."""

    df = pd.read_json('courses_embeddings.json')
    df = df.dropna()

    corpus = df['combined']

    vectorizer = TfidfVectorizer(
        analyzer='word', ngram_range=(1, 3), min_df=0.2)

    tfidf_matrix = vectorizer.fit_transform(corpus)

    query_matrix = vectorizer.transform([query])

    result_indices = []
    try:
        cosine_similarities = sklearn_cosine_similarity(
            query_matrix, tfidf_matrix).flatten()
        result_indices = cosine_similarities.argsort()[::-1][:n]
    except:
        print("No courses found matching your query.")
        return

    result = courses.iloc[result_indices]  # get the results using the indices
    if pprint:
        for _, r in result.iterrows():
            print("Course Name: " + re.sub(r'<\/?p>', '', str(r["Name"])))
            print("Description: " + re.sub(r'<\/?p>', '', str(r["Desc"])))
            print("Professors: " + re.sub(r'<\/?p>', '', str(r["Profs"])))
            print("Start Time: " + re.sub(r'<\/?p>', '', str(r["StartTime"])))
            print("End Time: " + re.sub(r'<\/?p>', '', str(r["EndTime"])))
            print()

    return result


def bert_encode(text):
    """Encodes the text using BERT."""

    tokens = tokenizer(text, padding=True, truncation=True,
                       return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**tokens).last_hidden_state[:, 0, :].numpy()
    return embeddings[0]


def openai_search(query, pprint=True, n=3):
    """Searches the courses_embeddings.json file for the query and returns the top n results sorted by cosine similarity."""

    df = pd.read_json('courses_embeddings.json')
    df = df[:100]
    query_embedding = openai.Embedding.create(
        input=query, engine="text-embedding-ada-002")["data"][0]["embedding"]

    df["similarity"] = df["embedding"].apply(
        lambda x: calculate_similarity_bert(x, query_embedding))

    result = df.sort_values("similarity", ascending=False).head(n)

    if pprint:
        for _, r in result.iterrows():
            print("Course Name: " + re.sub(r'<\/?p>', '', str(r["Name"])))
            print("Description: " + re.sub(r'<\/?p>', '', str(r["Desc"])))
            print("Professors: " + re.sub(r'<\/?p>', '', str(r["Profs"])))
            print("Start Time: " + re.sub(r'<\/?p>', '', str(r["StartTime"])))
            print("End Time: " + re.sub(r'<\/?p>', '', str(r["EndTime"])))
            print()

    return df


if __name__ == "__main__":
    start_time = time.time()
    load_bert_classes_and_embeddings()
    end_time = time.time()
    print("Time to compute BERT embeddings: " + str(end_time - start_time))
    print()

    while (True):
        print()
        search_type = input(
            "Enter 1 for OpenAI, 2 for BERT, and 3 for fuzzy search: ")
        print()
        if (search_type == "1"):
            query = input("Enter your search query: ")
            print()
            print("Results using OpenAI:")
            print()
            openai_result = openai_search(query)

        elif (search_type == "2"):
            query = input("Enter your search query: ")
            print()
            print("Results using BERT:")
            print()
            bert_result = bert_search(query)

        elif (search_type == "3"):
            query = input("Enter your search query: ")
            print()
            print("Results using fuzzy matching:")
            print()
            fuzzy_result = fuzzy_search(query)

        elif (search_type == "4"):
            query = input("Enter your search query: ")
            print()
            print("Results using TF-IDF retrieval:")
            print()
            tfidf_result = tfidf_search(query)

        else:
            print()
            print("Please select a valid search option.")
            print()
