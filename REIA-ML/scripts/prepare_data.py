import json
from transformers import RagTokenizer
import torch

def read_questions_from_file(file_path):
    with open(file_path, 'r') as file:
        questions = file.readlines()
    return [question.strip() for question in questions]

def prepare_dataset(questions_file, faqs_file, output_file):
    # Read questions from a text file
    questions = read_questions_from_file(questions_file)

    # Read data from JSON file
    with open(faqs_file, 'r') as file:
        data = json.load(file)

    # Prepare question-context-answer triplets
    qa_pairs = []
    for entry in data:
        context = entry["category"]
        title = entry["title"]
        for question in questions:
            answer = entry['content']  # For simplicity, we use the title as the answer
            qa_pairs.append({"question": question, "context": context, "answer": answer})

    # Save prepared dataset to a file
    with open(output_file, 'w') as file:
        json.dump(qa_pairs, file)

    print(f"Prepared dataset saved to {output_file}")

if __name__ == "__main__":
    # Paths to input files
    questions_file = '../dataset/questions.txt'
    faqs_file = '../dataset/faq_data.json'

    # Output file for prepared dataset
    output_file = '../dataset/prepared_dataset.json'

    # Prepare dataset
    prepare_dataset(questions_file, faqs_file, output_file)
