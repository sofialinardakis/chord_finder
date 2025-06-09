import torch
from train import ChordPredictor

def predict_chord(melody_seq):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChordPredictor()
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()
    model.to(device)

    with torch.no_grad():
        input_tensor = torch.tensor([melody_seq], dtype=torch.long).to(device)
        output = model(input_tensor)  # (1, chord_size, vocab_size)
        preds = torch.argmax(output, dim=2).squeeze(0)  # chord_size
        # Remove padded notes (-1 not predicted here)
        chord_notes = preds.cpu().tolist()
        chord_notes = [note for note in chord_notes if note >= 0 and note < 128]
        return chord_notes

if __name__ == "__main__":
    test_seq = [60, 62, 64, 65]
    chord = predict_chord(test_seq)
    print("Predicted chord notes:", chord)
