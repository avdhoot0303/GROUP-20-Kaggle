import json

def preprocess_faq_data(data_path):
  """
  Loads FAQ data from a JSON file, preprocesses it, and returns a list of dictionaries.

  Args:
      data_path (str): Path to the JSON file containing FAQ data.

  Returns:
      list: A list of dictionaries, where each dictionary represents a preprocessed FAQ entry.
  """

  with open(data_path, 'r') as f:
    faq_data = json.load(f)

  # Preprocess each FAQ entry
  processed_data = []
  for entry in faq_data:
    processed_entry = {}

    # Clean text (lowercase, remove special characters, etc.)
    processed_entry['title'] = clean_text(entry['title'].lower())
    processed_entry['content'] = clean_text(entry['content'].lower())

    # Tokenization (optional)
    # You can use libraries like nltk or spaCy for tokenization
    # processed_entry['title_tokens'] = tokenize(processed_entry['title'])
    # processed_entry['content_tokens'] = tokenize(processed_entry['content'])

    # Keep category information
    processed_entry['category'] = entry['category']

    processed_data.append(processed_entry)

  return processed_data

def clean_text(text):
  """
  Cleans text by removing special characters, converting to lowercase, etc.

  Args:
      text (str): The text to be cleaned.

  Returns:
      str: The cleaned text.
  """

  # Implement your desired cleaning logic here
  # You can use regular expressions or string manipulation techniques
  cleaned_text = text.strip()  # Remove leading/trailing whitespace
  # Add more cleaning steps as needed (e.g., removing punctuation, stopwords)
  return cleaned_text

# Example usage
faq_data = preprocess_faq_data("../dataset/faq_data.json")

# Print the first 2 processed entries
print(faq_data[:2])
def preprocess_user_questions(questions_file):
  """
  Loads and preprocesses user questions from a text file.

  Args:
      questions_file (str): Path to the text file containing user questions.

  Returns:
      list: A list of preprocessed user questions.
  """

  with open(questions_file, 'r') as f:
    questions = f.readlines()

  # Preprocess each question
  processed_questions = []
  for question in questions:
    processed_question = clean_text(question.strip().lower())  # Clean and lowercase
    processed_questions.append(processed_question)

  return processed_questions

# Example usage
user_questions = preprocess_user_questions("../dataset/questions.txt")

# Print the first 2 processed questions
print(user_questions[:2])

def combine_data(faq_data, user_questions):
  """
  Combines FAQ data and user questions into a single dataset.

  Args:
      faq_data (list): List of dictionaries representing preprocessed FAQ entries.
      user_questions (list): List of preprocessed user questions.

  Returns:
      list: A list of dictionaries representing the combined dataset (question-answer pairs).
  """

  combined_data = []
  for question in user_questions:
    # Find a matching FAQ answer (replace with your matching logic if needed)
    answer = None
    for entry in faq_data:
      if question in entry['title'] or question in entry['content']:
        answer = entry['content']
        break

    # Add a "no_answer" label if no match is found
    if answer is None:
      answer = "no_answer"

    combined_data.append({'question': question, 'answer': answer})

  return combined_data

# Example usage
combined_dataset = combine_data(faq_data, user_questions)

# Print the first 2 entries of the combined dataset
print(combined_dataset[:2])


from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, RagConfig, RagRetriever, RagTokenizer

# Define model names and paths (replace with your choices)
model_name = "facebook/bart-base"  # Example pre-trained model
tokenizer_name = model_name

# Load tokenizer and model config
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
config = RagConfig.from_pretrained(model_name)

# Create retriever and generator models
retriever = RagRetriever.from_pretrained(model_name, config=config)
generator = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Prepare dataset for training (replace with your actual data)
train_dataset = [
    {"question": "What is Powerwall?", "answer": "combined_data[0]['answer']"},  # Replace with actual answer access
    # ... add more question-answer pairs for training
]

# Training setup (replace with appropriate hyperparameters)
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./outputs",
    overwrite_output_dir=True,
    per_device_train_batch_size=8,
    save_steps=500,
    save_total_limit=2,
)

# Fine-tune retriever and generator (replace with library's training function)
# Specific function names might vary depending on the version
retriever.train(training_args, train_dataset=train_dataset)
generator.train(training_args, train_dataset=train_dataset)

# Save the fine-tuned models
retriever.save_pretrained("./retriever")
generator.save_pretrained("./generator")

# ... (Load the models for inference when your chatbot is ready)

