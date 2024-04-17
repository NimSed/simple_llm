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
num_epochs = 10000
seq_len = 200
learning_rate = 0.001
batch_size = 1024

# A piecewise text dataloader.
class piecewise_text_dataset(torch.utils.data.Dataset):
    def __init__(self, seq_len):
        with open('./data/janeausten.txt', 'r') as f:
            self.text = f.read()

        # temporarily pick only the first 1000 characters
        self.text = self.text[:2000]

        self.seq_len = seq_len

        # Create a character to index mapping
        self.chars = list(set(self.text))
        self.chars.sort()
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

    def __len__(self):
        return len(self.text) - self.seq_len

    def seq_to_int(self, seq):
        return torch.tensor([self.char_to_idx[ch] for ch in seq], dtype=torch.long)

    def seq_to_onehot(self, seq):
        return one_hot(self.seq_to_int(seq), num_classes=len(self.chars))

    def __getitem__(self, idx):
        seq = self.text[idx:idx+self.seq_len+1]
        return self.seq_to_onehot(seq[:-1]), self.seq_to_int(seq[1:])


# Create a dataset and a dataloader
dataset = piecewise_text_dataset(seq_len)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# constants
input_size = len(dataset.chars)
hidden_size = 1000
output_size = len(dataset.chars)

# Create a custom single-layer RNN model without using the built-in RNN module
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)

    def forward(self, x, hidden):
        combined = torch.cat((x, hidden), 1)
        hidden = torch.tanh(self.i2h(combined))
        output = self.i2o(combined)
        return output, hidden

    def init_hidden(self, batch_size=1):
        return torch.zeros(batch_size, self.hidden_size)

# Instantiate the model
rnn = RNNModel(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(rnn.parameters(), lr=learning_rate)

# Generate text
def generate_text(rnn, seed_text='Gutenberg', predict_len=seq_len):
    with torch.no_grad():
        hidden = rnn.init_hidden().to(device)
        seed_text_as_int = torch.tensor([dataset.char_to_idx[ch] for ch in seed_text], dtype=torch.long).to(device)
        seed_text_as_onehot = one_hot(seed_text_as_int, num_classes=input_size).float()

        for i in range(len(seed_text) - 1):
            _, hidden = rnn(seed_text_as_onehot[i].view(1, -1), hidden)

        predicted_text = seed_text
        x = seed_text_as_onehot[-1].view(1, -1)

        for i in range(predict_len):
            output, hidden = rnn(x, hidden)
            _, topi = output.topk(1)
            predicted_char = dataset.idx_to_char[topi.item()]
            predicted_text += predicted_char
            x = one_hot(topi, num_classes=input_size).float().view(1, -1)

    return predicted_text

# Train the model
print(generate_text(rnn, seed_text='The'))
for epoch in range(num_epochs):
    for i, (x, y) in enumerate(dataloader):
        total_loss = 0
        hidden = rnn.init_hidden(batch_size).to(device)

        # Forward pass
        for j in range(seq_len):
            x_batch = x[:, j, :].to(device)
            y_batch = y[:, j].to(device)

            output, hidden = rnn(x_batch, hidden)
            hidden = hidden.detach()

            loss = criterion(output, y_batch)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (i+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataset)//batch_size}], Loss: {total_loss/seq_len:.4f}')
            print(generate_text(rnn, seed_text='The'))


    print(f'Epoch {epoch + 1}, Loss: {total_loss/seq_len:.4f}')
    print(generate_text(rnn, seed_text='The'))
