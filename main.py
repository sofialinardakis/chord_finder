import prepare
import train
import torch

def main():
    folder = "rnb"
    print("Preparing training data...")
    X, y = prepare.process_folder(folder)
    if X is None or y is None:
        print("No training data found. Please add MIDI files to the 'rnb' folder.")
        return

    torch.save((X, y), "training_data.pt")
    print(f"Saved training data with {len(X)} sequences.")

    print("Training model...")
    train.train()
    print("Training complete. Now run test_output.py to generate outputs for your chosen file.")

if __name__ == "__main__":
    main()
