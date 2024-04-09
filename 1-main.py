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

torch.autograd.set_detect_anomaly(True)

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
text_as_int = np.array([char_to_idx[ch] for ch in text])

# One-hot encode the input characters
def one_hot_encode(indices, vocab_size):
    encoded = np.zeros((len(indices), vocab_size), dtype=np.float32)
    for i, idx in enumerate(indices):
        encoded[i, idx] = 1.0
    return encoded

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
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
rnn = RNNModel(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(rnn.parameters(), lr=learning_rate)

def validate(rnn, text_as_int):
    hidden = rnn.init_hidden()
    loss = 0
    for i in range(len(text_as_int) - 1):
        x = torch.tensor(text_as_int[i], dtype=torch.float32).view(1, 1)
        y = torch.tensor(text_as_int[i+1], dtype=torch.float32).view(1, 1)
        output, hidden = RNNModel(x, hidden)
        loss += criterion(output, y)
    return loss

# Train the model
for epoch in range(num_epochs):
    hidden = rnn.init_hidden().to(device)
    total_loss = 0

    for i in range(0, len(text_as_int) - 1):
        x = one_hot_encode([text_as_int[i]], input_size)
        y = torch.tensor([text_as_int[i + 1]], dtype=torch.long).to(device)
        inputs = torch.from_numpy(x).to(device)
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

# Generate text
with torch.no_grad():
    hidden = rnn.init_hidden()
    start = np.random.randint(0, len(text_as_int) - input_size)
    input = torch.tensor(text_as_int[start:start+input_size]).float()
    print(idx_to_char[text_as_int[start]], end='')
    for i in range(100):
        output, hidden = rnn(input, hidden)
        output = torch.argmax(output)
        print(idx_to_char[output.item()], end='')
        input = output
    print()
