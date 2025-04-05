import tkinter as tk
from tkinter import filedialog, messagebox
import json
import numpy as np
import os

class PianoTuner:
    def __init__(self, master):
        self.master = master
        self.master.title("Piano Tuner")
        self.current_note_index = 0
        self.notes_to_tune = []
        self.stretch_curve = {}

        # UI Elements
        self.label = tk.Label(master, text="Load a stretch curve file to begin.", font=("Arial", 14))
        self.label.pack(pady=10)

        self.load_button = tk.Button(master, text="Load Stretch Curve", command=self.load_stretch_curve)
        self.load_button.pack(pady=5)

        self.next_button = tk.Button(master, text="Next Note", command=self.next_note, state=tk.DISABLED)
        self.next_button.pack(pady=10)

        self.previous_button = tk.Button(master, text="Previous Note", command=self.previous_note, state=tk.DISABLED)
        self.previous_button.pack(pady=5)

        self.status = tk.Label(master, text="", font=("Arial", 12))
        self.status.pack(pady=10)

    def load_stretch_curve(self):
        filepath = filedialog.askopenfilename(
            title="Select Stretch Curve File",
            filetypes=(("JSON Files", "*.json"), ("All Files", "*.*"))
        )
        if not filepath:
            return

        try:
            with open(filepath, "r") as f:
                self.stretch_curve = json.load(f)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {e}")
            return

        # Sort notes: Start from middle C, go up, then down
        middle_c_index = NOTES.index("C4")
        self.notes_to_tune = (
            NOTES[middle_c_index:] + NOTES[:middle_c_index][::-1]
        )
        self.notes_to_tune = [note for note in self.notes_to_tune if note in self.stretch_curve]

        self.current_note_index = 0
        self.label.config(text=f"Loaded stretch curve: {os.path.basename(filepath)}")
        self.status.config(text="Press 'Next Note' to start tuning.")
        self.next_button.config(state=tk.NORMAL)
        self.previous_button.config(state=tk.DISABLED)  # Disable "Previous Note" initially

    def next_note(self):
        if self.current_note_index >= len(self.notes_to_tune):
            self.label.config(text="Tuning complete!")
            self.status.config(text="All notes have been tuned.")
            self.next_button.config(state=tk.DISABLED)
            return

        note = self.notes_to_tune[self.current_note_index]
        partials = self.stretch_curve[note]
        self.label.config(text=f"Tune {note} (Fundamental: {note_frequencies[note]} Hz)")
        self.status.config(text=f"Partials: {np.round(partials, 2)}")
        self.current_note_index += 1

        # Enable "Previous Note" if not at the first note
        if self.current_note_index > 0:
            self.previous_button.config(state=tk.NORMAL)

    def previous_note(self):
        if self.current_note_index <= 0:
            self.previous_button.config(state=tk.DISABLED)
            return

        self.current_note_index -= 1
        note = self.notes_to_tune[self.current_note_index]
        partials = self.stretch_curve[note]
        self.label.config(text=f"Tune {note} (Fundamental: {note_frequencies[note]} Hz)")
        self.status.config(text=f"Partials: {np.round(partials, 2)}")

        # Disable "Previous Note" if back at the first note
        if self.current_note_index == 0:
            self.previous_button.config(state=tk.DISABLED)

# Note frequencies for reference
note_frequencies = {
    "C2": 65.41, "C3": 130.81, "C4": 261.63, "C5": 523.25, "C6": 1046.50, "C7": 2093.00,
    "E2": 82.41, "E3": 164.81, "E4": 329.63, "E5": 659.26, "E6": 1318.51, "E7": 2637.02,
    "F2": 87.31, "F3": 174.61, "F4": 349.23, "F5": 698.46, "F6": 1396.91, "F7": 2793.83,
    "A2": 110.00, "A3": 220.00, "A4": 440.00, "A5": 880.00, "A6": 1760.00, "A7": 3520.00
}

# Notes in order for tuning
NOTES = [
    "C2", "C3", "C4", "C5", "C6", "C7",
    "E2", "E3", "E4", "E5", "E6", "E7",
    "F2", "F3", "F4", "F5", "F6", "F7",
    "A2", "A3", "A4", "A5", "A6", "A7"
]

if __name__ == "__main__":
    root = tk.Tk()
    app = PianoTuner(root)
    root.mainloop()