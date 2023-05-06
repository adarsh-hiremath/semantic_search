import random
import string

def random_query_string(min_length, max_length):
    length = random.randint(min_length, max_length)
    return ''.join(random.choices(string.ascii_letters + ' ', k=length))

topics = [
    "math",
    "history",
    "chemistry",
    "physics",
    "biology",
    "computer science",
    "literature",
    "economics",
    "philosophy",
    "psychology",
]

questions = [
    "What is the syllabus for {} class?",
    "Best books for {} class?",
    "Recommended resources for {}?",
    "How difficult is the {} class?",
    "What are the prerequisites for {} class?",
    "What topics are covered in {} class?",
    "Professors teaching {} class?",
    "What is the workload of {} class?",
    "Any tips for succeeding in {} class?",
    "What projects are done in {} class?",
]

random_queries = []

for _ in range(1000):
    topic = random.choice(topics)
    question = random.choice(questions)
    query = question.format(topic)
    query = random_query_string(0, 3) + query + random_query_string(0, 3)
    random_queries.append(query.strip())

for query in random_queries:
    print(query)
