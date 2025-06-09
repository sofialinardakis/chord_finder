import mido
import torch
from train import ChordPredictor
import torch.nn.functional as F

def midi_to_sequence(mid):
    """
    Convert midi notes in the melody track into a sequence of note numbers.
    Simplified: takes first track's note_on notes as melody.
    """
    notes = []
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                notes.append(msg.note)
    return notes

def sequence_to_tensor(seq):
    """
    Converts list of note integers to a torch tensor batch.
    Here batch size 1, sequence length = len(seq)
    """
    return torch.tensor(seq, dtype=torch.long).unsqueeze(0)

def predict_chords(model, input_seq_tensor, chord_size=4):
    model.eval()
    with torch.no_grad():
        output = model(input_seq_tensor)  # shape: (1, chord_size, 128)
        # Pick top note from each chord position
        chords = []
        for note_pos in range(chord_size):
            probs = F.softmax(output[0, note_pos], dim=0)
            note = torch.argmax(probs).item()
            chords.append(note)
    return chords

def overlay_chords_on_midi(input_midi_path, output_midi_path, model_path='model.pth',
                           melody_velocity_scale=0.3, chord_velocity_scale=1.0):
    mid = mido.MidiFile(input_midi_path)
    new_mid = mido.MidiFile()

    # Lower melody velocity
    for i, track in enumerate(mid.tracks):
        new_track = mido.MidiTrack()
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                new_msg = msg.copy(velocity=int(msg.velocity * melody_velocity_scale))
                new_track.append(new_msg)
            else:
                new_track.append(msg)
        new_mid.tracks.append(new_track)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChordPredictor()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Convert melody notes to input sequence tensor
    melody_notes = midi_to_sequence(mid)
    if len(melody_notes) == 0:
        raise ValueError("No melody notes found in input MIDI.")
    input_seq = sequence_to_tensor(melody_notes).to(device)

    # Predict chords
    chord_notes = predict_chords(model, input_seq)

    # Create chord track, place chords at start for demo (improve timing as needed)
    chord_track = mido.MidiTrack()
    chord_time = 0
    for note in chord_notes:
        chord_track.append(mido.Message('note_on', note=note, velocity=int(90 * chord_velocity_scale), time=chord_time))
        chord_time = 0
    # Add note_off messages with a fixed duration
    chord_duration = 480
    for note in chord_notes:
        chord_track.append(mido.Message('note_off', note=note, velocity=0, time=chord_duration))
        chord_duration = 0

    new_mid.tracks.append(chord_track)
    new_mid.save(output_midi_path)

if __name__ == "__main__":
    # Example run
    input_file = "rnb/AUD_Lr0966.mid"
    output_file = "output_with_chords.mid"
    overlay_chords_on_midi(input_file, output_file)
    print(f"Output saved to {output_file}")
