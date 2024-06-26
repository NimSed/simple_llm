"""
Minimal character-level Vanilla RNN model in pytroch.

Nima Sedaghat
rawdataspeaks.org
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.functional import one_hot

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 100
learning_rate = 0.01

# Dataset is a simple text file.
with open('./data/janeausten.txt', 'r') as f:
    text = f.read()

# Create a character to index mapping
chars = list(set(text))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# Constants
input_size = len(chars)
hidden_size = 100
output_size = len(chars)

# Convert text to tensor
text_as_int = torch.tensor([char_to_idx[ch] for ch in text], dtype=torch.long).to(device)
text_as_onehot = one_hot(torch.tensor(text_as_int), num_classes=input_size).float()

# Create a custom single-layer RNN model without using the built-in RNN module
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)

    def forward(self, x, hidden):
        combined = torch.cat((x, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

# Instantiate the model
rnn = RNNModel(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(rnn.parameters(), lr=learning_rate)

# Generate text
def generate_text(rnn, seed_text='Gutenberg', predict_len=200):
    with torch.no_grad():
        hidden = rnn.init_hidden().to(device)
        seed_text_as_int = torch.tensor([char_to_idx[ch] for ch in seed_text], dtype=torch.long).to(device)
        seed_text_as_onehot = one_hot(seed_text_as_int, num_classes=input_size).float()

        for i in range(len(seed_text) - 1):
            _, hidden = rnn(seed_text_as_onehot[i].view(1, -1), hidden)

        predicted_text = seed_text
        x = seed_text_as_onehot[-1].view(1, -1)

        for i in range(predict_len):
            output, hidden = rnn(x, hidden)
            _, topi = output.topk(1)
            predicted_char = idx_to_char[topi.item()]
            predicted_text += predicted_char
            x = one_hot(topi, num_classes=input_size).float().view(1, -1)

    return predicted_text

print(generate_text(rnn))

# Train the model
for epoch in range(num_epochs):
    hidden = rnn.init_hidden().to(device)
    total_loss = 0

    for i in range(0, len(text_as_int) - 1):
        x = text_as_onehot[i].view(1, -1)
        y = text_as_int[i + 1].view(1)
        inputs = x
        outputs, hidden = rnn(inputs, hidden)
        hidden = hidden.detach()

        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if (i + 1) % 10000 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(text_as_int) - 1}], Loss: {loss.item()}')

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(text_as_int)}')
    print(generate_text(rnn))
