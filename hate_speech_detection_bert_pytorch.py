import torch
from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
import re
from tqdm import tqdm

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained BERT model and tokenizer
bert_model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class HateSpeechDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.text = self.data['tweet'].dropna()
        self.text = self.text.apply(lambda x: clean_text(x))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.text.iloc[idx]
        encoded_text = tokenizer.encode_plus(
            text,
            max_length=60,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        input_ids = encoded_text['input_ids'].squeeze()
        attention_mask = encoded_text['attention_mask'].squeeze()
        label = self.data['class'].iloc[idx]

        return input_ids, attention_mask, torch.tensor(label)

def clean_text(text):
    text = re.sub(r"@[A-Za-z0-9_-]+", 'USR', text)
    text = re.sub(r"http\S+", 'URL', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower()

def collate_fn(batch):
    input_ids, attention_masks, labels = zip(*batch)
    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)
    labels = torch.stack(labels)
    return input_ids, attention_masks, labels

class HateSpeechClassifier(nn.Module):
    def __init__(self):
        super(HateSpeechClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.1)
        self.linear1 = nn.Linear(768, 256)
        self.linear2 = nn.Linear(256, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[0][:, 0, :]  # Use [CLS] token
        x = self.dropout(pooled_output)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

def train_model(model, train_data, dev_data, epochs=10, learning_rate=5e-5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct_predictions = 0
        total_samples = 0
        pbar = tqdm(train_data, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for batch in pbar:
            optimizer.zero_grad()
            input_ids, attention_mask, labels = [t.to(device) for t in batch]

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            pbar.set_postfix(loss=epoch_loss / len(train_data), accuracy=correct_predictions / total_samples)

        scheduler.step()

        # Evaluate on dev set
        model.eval()
        dev_correct = 0
        dev_total = 0
        with torch.no_grad():
            for batch in dev_data:
                input_ids, attention_mask, labels = [t.to(device) for t in batch]
                outputs = model(input_ids, attention_mask)
                predicted = torch.argmax(outputs, dim=1)
                dev_correct += (predicted == labels).sum().item()
                dev_total += labels.size(0)

        dev_acc = dev_correct / dev_total
        print(f"Epoch {epoch+1} - Dev Accuracy: {dev_acc:.4f}")

# Load your hate speech dataset
data = pd.read_csv("Detection.csv")

# Split data into train and dev sets
train_data, dev_data = train_test_split(data, test_size=0.2, random_state=42)

# Create datasets
train_dataset = HateSpeechDataset(train_data)
dev_dataset = HateSpeechDataset(dev_data)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
dev_loader = DataLoader(dev_dataset, batch_size=32, collate_fn=collate_fn)

# Initialize the model
model = HateSpeechClassifier()
model = model.to(device)

# Train the model
train_model(model, train_loader, dev_loader)

# Save the model state
torch.save(model.state_dict(), 'hate_speech_model.pt')