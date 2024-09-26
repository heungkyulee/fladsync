import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import MobileBertTokenizer, MobileBertForSequenceClassification, AdamW
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm  # Import tqdm for progress bars

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 1: Load the dataset
data_path = "models/training_data.csv"  # Your uploaded file
df = pd.read_csv(data_path, sep=',', quotechar='"', names=["col1", "col2", "col3", "IABCategory", "title", "description", "col6"], on_bad_lines='skip')

# Step 2: Preprocess data
# Replace NaN in title and description with empty strings
df = df.dropna(subset=['title'])
df = df.dropna(subset=['description'])

# Combining title and description into one input column
df['input'] = df['title'] + ' ' + df['description']

# Ensure 'IABCategory' contains valid strings before splitting
df['IABCategory'] = df['IABCategory'].apply(lambda x: x.split(',') if isinstance(x, str) else [])

# Step 3: Tokenizer and MultiLabel Binarizer
tokenizer = MobileBertTokenizer.from_pretrained('google/mobilebert-uncased')
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(df['IABCategory'])

# Tokenize and encode the inputs
class ContentDataset(Dataset):
    def __init__(self, inputs, labels, tokenizer, max_len=256):
        self.inputs = inputs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

# Step 4: Split data into training and testing sets
train_inputs, test_inputs, train_labels, test_labels = train_test_split(df['input'], labels, test_size=0.2, random_state=42)

train_dataset = ContentDataset(train_inputs.values, train_labels, tokenizer)
test_dataset = ContentDataset(test_inputs.values, test_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Step 5: Define the Model
model = MobileBertForSequenceClassification.from_pretrained('google/mobilebert-uncased', num_labels=len(mlb.classes_))
model = model.to(device)

# Step 6: Training Loop with tqdm
optimizer = AdamW(model.parameters(), lr=2e-5)
epochs = 4

def train_model(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(data_loader, desc="Training")  # Add tqdm progress bar
    for batch in progress_bar:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        # Update progress bar with the current loss
        progress_bar.set_postfix({'loss': loss.item()})

    return total_loss / len(data_loader)

def evaluate_model(model, data_loader, device):
    model.eval()
    total_loss = 0
    progress_bar = tqdm(data_loader, desc="Evaluating")  # Add tqdm progress bar
    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            # Update progress bar with the current loss
            progress_bar.set_postfix({'loss': loss.item()})

    return total_loss / len(data_loader)

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    train_loss = train_model(model, train_loader, optimizer, device)
    val_loss = evaluate_model(model, test_loader, device)
    print(f"Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

# Step 7: Save the model for inference
model.save_pretrained("saved_model/")
tokenizer.save_pretrained("saved_model/")
mlb_filename = "mlb_classes.npy"
np.save(mlb_filename, mlb.classes_)  # Saving the MultiLabelBinarizer classes for future use in inference
