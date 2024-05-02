import json
import torch
from transformers import RagTokenForGeneration,Trainer, TrainingArguments, RagTokenizer

def read_prepared_dataset(file_path):
    with open(file_path, 'r') as file:
        dataset = json.load(file)
    return dataset

if __name__ == "__main__":
    # Path to prepared dataset
    prepared_dataset_file = '../dataset/prepared_dataset.json'

    # Load prepared dataset
    dataset = read_prepared_dataset(prepared_dataset_file)

    # Tokenization (using Hugging Face Transformers library)
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-base")

    # Tokenize question-context pairs
    tokenized_data = tokenizer(
        [pair["question"] for pair in dataset],
        [pair["context"] for pair in dataset],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512  # Adjust as needed
    )

    # Create PyTorch Dataset
    class QADataset(torch.utils.data.Dataset):
        def __init__(self, encodings, answers):
            self.encodings = encodings
            self.answers = answers

        def __getitem__(self, idx):
            item = {key: tensor[idx] for key, tensor in self.encodings.items()}
            item["labels"] = torch.tensor(int(self.answers[idx]))
            return item

        def __len__(self):
            return len(self.encodings.input_ids)

    # Create dataset instance
    dataset = QADataset(tokenized_data, answers=[pair["answer"] for pair in dataset])
    model = RagTokenForGeneration.from_pretrained("facebook/rag-token-base")


    # Split dataset into training, validation, and testing sets (if needed)

    # Define training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=4,
        num_train_epochs=3,
        output_dir="./logs",
    )

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,  # Use the dataset you've prepared
    )

    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model
    output_dir = "../models/fine_tuned_model"
    trainer.save_model(output_dir)
