
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
