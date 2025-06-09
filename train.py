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
        out = self.fc(h[-1])  # batch x (vocab_size * chord_size)
        out = out.view(-1, self.chord_size, self.vocab_size)
        return out  # logits per chord note

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y = torch.load("training_data.pt")
    X, y = X.to(device), y.to(device)

    model = ChordPredictor().to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 100
    batch_size = 64
    n = len(X)

    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            batch_X, batch_y = X[idx], y[idx]

            optimizer.zero_grad()
            output = model(batch_X)  # (batch, chord_size, vocab_size)

            loss = 0
            for note_pos in range(model.chord_size):
                loss += criterion(output[:, note_pos, :], batch_y[:, note_pos])
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / (n / batch_size)
        print(f"Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "model.pth")
    print("Training finished. Model saved.")

if __name__ == "__main__":
    train()
