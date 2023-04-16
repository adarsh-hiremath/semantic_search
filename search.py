import json
import openai
import numpy as np
import pandas as pd
import re
from openai.embeddings_utils import cosine_similarity

openai.api_key = "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"


def load_classes_and_embeddings():
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


def calculate_similarity(query, query_embedding):
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


def openai_search(query, pprint=True, n=3):
    """Searches the courses_embeddings.json file for the query and returns the top n results sorted by cosine similarity."""

    df = pd.read_json('courses_embeddings.json')
    df = df[:100]
    query_embedding = openai.Embedding.create(
        input=query, engine="text-embedding-ada-002")["data"][0]["embedding"]

    df["similarity"] = df["embedding"].apply(
        lambda x: calculate_similarity(x, query_embedding))

    result = df.sort_values("similarity", ascending=False).head(n)

    if pprint:
        for _, r in result.iterrows():
            print("Course Name: " + re.sub(r'<\/?p>', '', r["Name"]))
            print("Description: " + re.sub(r'<\/?p>', '', r["Desc"]))
            print("Professors: " + re.sub(r'<\/?p>', '', r["Profs"]))
            print("Start Time: " + re.sub(r'<\/?p>', '', r["StartTime"]))
            print("End Time: " + re.sub(r'<\/?p>', '', r["EndTime"]))
            print()

    return df


if __name__ == "__main__":
    # load_classes_and_embeddings()
    query = input("Enter your search query: ")
    print()
    result = openai_search(query)
