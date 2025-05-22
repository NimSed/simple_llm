"""
Minimal character-level transformer-based lm in pytorch.

Nima Sedaghat
rawdataspeaks.org
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.functional import one_hot

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 10000
seq_len = 256
learning_rate = 0.001
batch_size = 512
workers = 20

d_model = 128
n_head = 8
num_layers = 8

# A piecewise text dataloader.
class piecewise_text_dataset(torch.utils.data.Dataset):
    def __init__(self, seq_len):
        with open('./data/janeausten.txt', 'r') as f:
            self.text = f.read()

        # temporarily pick only the first 1000 characters
        #self.text = self.text[:2000]

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
        return one_hot(self.seq_to_int(seq), num_classes=len(self.chars)).float()

    def __getitem__(self, idx):
        seq = self.text[idx:idx+self.seq_len+1]
        return self.seq_to_onehot(seq[:-1]), self.seq_to_int(seq[1:])


# Create a dataset and a dataloader
dataset = piecewise_text_dataset(seq_len)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=workers)

# constants
input_size = len(dataset.chars)
output_size = len(dataset.chars)

# A decoder-only minimal transformer model
class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_head=4, num_layers=2):
        super().__init__()
        #self.embed = nn.Embedding(vocab_size, d_model)
        self.embed = nn.Linear(vocab_size, d_model)
        self.pos_embed = nn.Embedding(1024, d_model)  # Simple positional encoding
        decoder_layer = nn.TransformerDecoderLayer(d_model, n_head, dim_feedforward=128)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0)
        #x = x.permute([0,2,1])
        x = self.embed(x)
        x = x + self.pos_embed(pos)
        # Transformer expects (seq_len, batch, d_model)
        x = x.transpose(0, 1)
        # Create a causal mask to prevent attending to future positions
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        # No encoder, so use zeros as memory (for decoder-only)
        memory = torch.zeros(1, x.size(1), x.size(2), device=x.device)
        out = self.decoder(x, memory, tgt_mask=tgt_mask)
        out = out.transpose(0, 1)  # (batch, seq_len, d_model)
        logits = self.fc(out)
        return logits

# Instantiate the model
model = MiniTransformer(vocab_size=input_size, d_model=d_model, n_head=n_head, num_layers=num_layers).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Sampling function
def generate_text(model, seed_text='Gutenberg', length=40):
    model.eval()

    predicted_text = seed_text
    with torch.no_grad():
        for _ in range(length):
            x = dataset.seq_to_onehot(predicted_text[-seq_len:]).to(device)
            logits = model(x.unsqueeze(0))
            next_id = torch.argmax(logits[0, -1]).item()
            predicted_text += dataset.idx_to_char[next_id]
            print(predicted_text[-seq_len:])

    return predicted_text

# Train the model
print(generate_text(model, seed_text='The'))

for epoch in range(num_epochs):
    for i, (x, y) in enumerate(dataloader):
        model.train()
        total_loss = 0

        x, y = x.to(device), y.to(device)
        # Forward pass
        output = model(x)

        loss = criterion(output.view(-1, output_size), y.view(-1))

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (i+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataset)//batch_size}], Loss: {total_loss/seq_len:.4f}')
            print(generate_text(model, seed_text='The'))


    print(f'Epoch {epoch + 1}, Loss: {total_loss/seq_len:.4f}')
    print(generate_text(model, seed_text='The'))
