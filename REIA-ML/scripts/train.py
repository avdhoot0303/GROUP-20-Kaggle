from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import torch
import json

# Load knowledge base data
with open("../dataset/faq_data.json", "r") as f:
    knowledge_base = json.load(f)

# Prepare data for fine-tuning
training_data = []
for item in knowledge_base:
    context = item['title']
    response = item['content']
    training_data.append((context, response))

# Define custom dataset
class KnowledgeBaseDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, response = self.data[idx]
        input_text = f"{context} {self.tokenizer.eos_token} {response}"
        input_ids = self.tokenizer.encode(input_text, max_length=self.max_length, truncation=True, return_tensors="pt")
        return input_ids

# Load pre-trained model and tokenizer
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Fine-tuning parameters
batch_size = 8
max_length = 512
num_epochs = 3
learning_rate = 5e-5

# Create DataLoader for training data
dataset = KnowledgeBaseDataset(training_data, tokenizer, max_length)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch.to(model.device)
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}: Average Loss {loss.item()}")

# Save fine-tuned model and tokenizer
model.save_pretrained("fine_tuned_model")
tokenizer.save_pretrained("fine_tuned_model")
