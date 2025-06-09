import os
import mido
import torch

def prepare_training_data():
    sequences_X = []
    sequences_y = []

    midi_folder = "rnb"

    for filename in os.listdir(midi_folder):
        if not filename.endswith(".mid"):
            continue
        filepath = os.path.join(midi_folder, filename)
        try:
            mid = mido.MidiFile(filepath)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

        notes = []
        for track in mid.tracks:
            for msg in track:
                if msg.type == 'note_on' and msg.velocity > 0:
                    notes.append(msg.note)

        # Simple sequence generation: pairs of notes and chords (dummy example)
        # Replace with your real sequence preparation logic
        for i in range(len(notes) - 5):
            seq_in = notes[i:i+5]
            seq_out = notes[i+5:i+9] if i+9 <= len(notes) else [0,0,0,0]
            if len(seq_out) < 4:
                seq_out += [0] * (4 - len(seq_out))

            sequences_X.append(seq_in)
            sequences_y.append(seq_out)

    if not sequences_X:
        print("No training data prepared.")
        return

    X = torch.tensor(sequences_X, dtype=torch.long)
    y = torch.tensor(sequences_y, dtype=torch.long)

    torch.save((X, y), "training_data.pt")
    print(f"Prepared {len(sequences_X)} training sequences.")

if __name__ == "__main__":
    prepare_training_data()
