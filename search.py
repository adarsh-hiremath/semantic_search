import json
import sys
import openai

openai.api_key = "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"

def load_classes(): 
    return json.load(open('courses_normalized.json'))

def clean_up(classes_json):
    for i in range(len(classes_json)):
        try: 
            classes_json[i].pop("Subject")
            classes_json[i].pop("Number")
            classes_json[i].pop("ClassNumber")
            classes_json[i].pop("CourseId")
            classes_json[i].pop("Profs")
            classes_json[i].pop("F22")
            classes_json[i].pop("Days")
            classes_json[i].pop("M")
            classes_json[i].pop("T")
            classes_json[i].pop("W")
            classes_json[i].pop("R")
            classes_json[i].pop("F")
            classes_json[i].pop("StartTime")
            classes_json[i].pop("EndTime")
            classes_json[i].pop("HasSection")
            classes_json[i].pop("AH")
            classes_json[i].pop("Location")
            classes_json[i].pop("URL")
        except: 
            continue
    return classes_json

def search(query, classes_json):
    documents = []
    for class_info in classes_json:
        try:
            document = {
                "id": class_info["id"],
                "text": f"{class_info['name']} - {class_info['desc']}"
            }
            documents.append(document)
        except KeyError as e:
            print(f"Error processing class_info: {class_info}. Missing key: {e}")

    response = openai.Answer.create(
        search_model="davinci",
        model="davinci",
        question=query,
        documents=documents,
        examples_context="",
        max_responses=1,
        lls_model="text-davinci-002",
        return_prompt=True,
        return_metadata=True,
        stop=None,
        temperature=0.5,
    )

    most_relevant_id = response['choices'][0]['metadata']['id']

    return most_relevant_id

if __name__ == "__main__": 
    classes_json = load_classes()
    search_query = input("Enter your search query: ")
    result = search(search_query, classes_json)

    print(f"The most relevant class ID for '{search_query}' is {result}.")


