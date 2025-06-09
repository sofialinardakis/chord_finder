import torch
import torch.nn as nn

class ChordPredictor(nn.Module):
    def __init__(self, vocab_size=128, embed_dim=64, hidden_dim=128, chord_size=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size * chord_size)
        self.chord_size = chord_size
        self.vocab_size = vocab_size

    def forward(self, x):
        x = self.embed(x)
        _, (h, _) = self.lstm(x)
        out = self.fc(h[-1])
        out = out.view(-1, self.chord_size, self.vocab_size)
        return out
