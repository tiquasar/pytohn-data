
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
from collections import Counter

# Dataset preparation
class TextDataset(Dataset):
    def __init__(self, text, sequence_length=50):
        self.text = text
        self.sequence_length = sequence_length
        self.tokens = self.tokenize_text()

        # Building the vocabulary
        self.vocab = sorted(set(self.tokens))
        self.token_to_idx = {token: idx for idx, token in enumerate(self.vocab)}
        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}

        # Encoding text
        self.encoded_text = [self.token_to_idx[token] for token in self.tokens]

    def tokenize_text(self):
        # Simple tokenization (by space); customize as needed
        return self.text.split()

    def __len__(self):
        return len(self.encoded_text) - self.sequence_length

    def __getitem__(self, idx):
        return (
            torch.tensor(self.encoded_text[idx:idx+self.sequence_length]),
            torch.tensor(self.encoded_text[idx+1:idx+self.sequence_length+1])
        )

# Model definition
class TextPredictor(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, n_layers=2):
        super(TextPredictor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out)
        return out

# Example usage
text = "Your text data goes here. This is just an example text." # Replace with your text data
dataset = TextDataset(text)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Instantiate the model
model = TextPredictor(len(dataset.vocab))

# Training loop
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10  # Number of epochs to train for

for epoch in range(epochs):
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output.transpose(1, 2), targets)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

# Save the model (Optional)
torch.save(model.state_dict(), 'text_predictor_model.pth')






import pandas as pd
from ctgan import CTGANSynthesizer

def load_data_chunk(file_path, chunksize, skiprows):
    """Load a chunk of data from a CSV file."""
    chunk = pd.read_csv(file_path, chunksize=chunksize, skiprows=skiprows)
    return next(chunk)

def save_model_state(ctgan, filepath):
    """Save the CTGAN model state."""
    ctgan.save(filepath)

def load_model_state(filepath):
    """Load a CTGAN model state from a file."""
    ctgan = CTGANSynthesizer()
    ctgan = ctgan.load(filepath)
    return ctgan

def incremental_training_with_state_save(file_path, chunksize, model_path, epochs=1):
    """Train a CTGAN model incrementally on data chunks and save/reuse model state."""
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        total_rows = sum(1 for _ in open(file_path)) - 1  # Exclude header
        total_chunks = (total_rows // chunksize) + (0 if total_rows % chunksize == 0 else 1)
        
        for chunk_idx in range(total_chunks):
            print(f"Training on chunk {chunk_idx+1}/{total_chunks}")
            
            if chunk_idx == 0 and epoch == 0:
                # Initialize a new model for the very first chunk and first epoch
                ctgan = CTGANSynthesizer(epochs=1)
            else:
                # Load the model from the previous state for subsequent chunks/epochs
                ctgan = load_model_state(model_path)
            
            skiprows = 1 + chunk_idx * chunksize
            chunk = load_data_chunk(file_path, chunksize, range(1, skiprows))
            
            # Preprocess your chunk as necessary here
            
            # Train the model on the current chunk
            ctgan.fit(chunk)
            
            # Save the model state after training on the chunk
            save_model_state(ctgan, model_path)
            
            print(f"Finished training on chunk {chunk_idx+1}/{total_chunks}")
    
    print("Finished training on all chunks.")
    # Optionally return or load the final model state
    final_model = load_model_state(model_path)
    return final_model

# Parameters
file_path = 'your_data.csv'  # Your CSV file path
chunksize = 10000  # Adjust based on your memory capacity
model_path = 'ctgan_model.pkl'  # Path to save/load the model state
epochs = 5  # Number of times to iterate through all chunks

# Train the model incrementally and save/reuse the model state
final_model = incremental_training_with_state_save(file_path, chunksize, model_path, epochs)

# After training, you can use final_model to generate synthetic data

