import sys
import output

if len(sys.argv) < 2:
    print("Usage: python test_output.py <midi_file_path>")
    sys.exit(1)

midi_file = sys.argv[1]
output_file = "output_with_chords.mid"

output.overlay_chords_on_midi(midi_file, output_file)
print(f"Generated output saved to {output_file}")
print("Listen and let me know if you want to improve the model!")
