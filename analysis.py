import tkinter as tk
from tkinter import messagebox, filedialog
import sounddevice as sd
import numpy as np
import librosa
import json
import threading
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.interpolate import UnivariateSpline

SAMPLE_RATE = 44100
DURATION = 3  # seconds
AMPLITUDE_THRESHOLD = 0.001  # lowered threshold for softer note detection

NOTES_TO_SAMPLE = ["C2", "C3", "C4", "C5", "C6", "C7", "E2", "E3", "E4", "E5", "E6", "E7", "F2", "F3", "F4", "F5", "F6", "F7", "A2", "A3", "A4", "A5", "A6", "A7"]
note_frequencies = {
    "C2": 65.41, "C3": 130.81, "C4": 261.63, "C5": 523.25, "C6": 1046.50, "C7": 2093.00,
    "E2": 82.41, "E3": 164.81, "E4": 329.63, "E5": 659.26, "E6": 1318.51, "E7": 2637.02,
    "F2": 87.31, "F3": 174.61, "F4": 349.23, "F5": 698.46, "F6": 1396.91, "F7": 2793.83,
    "A2": 110.00, "A3": 220.00, "A4": 440.00, "A5": 880.00, "A6": 1760.00, "A7": 3520.00
}

recorded_partials = {}


def record_note():
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    return audio.flatten()


def extract_partials(audio, target_freq):
    spectrum = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
    freqs = np.fft.rfftfreq(len(audio), 1 / SAMPLE_RATE)

    partials = []
    for n in range(1, 7):
        expected = target_freq * n
        idx = np.argmin(np.abs(freqs - expected))
        partial_freq = freqs[idx]
        partials.append(partial_freq)
    return partials


class PianoStretchAnalyzer:
    def __init__(self, master):
        self.master = master
        self.master.title("Piano Stretch Analyzer")
        self.current_note_index = 0

        self.label = tk.Label(master, text="Press Start to begin analysis", font=("Arial", 14))
        self.label.pack(pady=10)

        self.folder_button = tk.Button(master, text="Choose Save Folder", command=self.choose_folder)
        self.folder_button.pack(pady=5)

        self.name_entry = tk.Entry(master, width=30)
        self.name_entry.insert(0, "MyPiano")
        self.name_entry.pack(pady=5)

        self.start_button = tk.Button(master, text="Start Analysis", command=self.start_analysis)
        self.start_button.pack(pady=10)

        self.status = tk.Label(master, text="", font=("Arial", 12))
        self.status.pack(pady=10)

        self.figure, self.ax = plt.subplots(figsize=(6, 3))
        self.canvas = FigureCanvasTkAgg(self.figure, master)
        self.canvas.get_tk_widget().pack()

        self.save_folder = os.getcwd()

    def choose_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.save_folder = folder
            self.status.config(text=f"Save folder: {self.save_folder}")

    def start_analysis(self):
        self.start_button.config(state=tk.DISABLED)
        threading.Thread(target=self.record_next).start()

    def record_next(self):
        if self.current_note_index >= len(NOTES_TO_SAMPLE):
            self.finish_analysis()
            return

        note = NOTES_TO_SAMPLE[self.current_note_index]
        self.label.config(text=f"Play {note} and wait...")
        self.status.config(text="Recording...")

        audio = record_note()
        rms = np.sqrt(np.mean(audio ** 2))

        if rms < AMPLITUDE_THRESHOLD:
            self.status.config(text=f"Too quiet. Please play {note} louder.")
            self.master.after(3000, self.record_next)
            return

        target_freq = note_frequencies[note]
        partials = extract_partials(audio, target_freq)
        recorded_partials[note] = partials

        self.status.config(text=f"Captured {note} partials: {np.round(partials, 2)}")
        self.current_note_index += 1
        self.master.after(3000, self.record_next)

    def finish_analysis(self):
        self.label.config(text="Analysis complete.")
        piano_name = self.name_entry.get().strip()
        if not piano_name:
            piano_name = "MyPiano"

        filepath = os.path.join(self.save_folder, f"{piano_name}_stretch_curve.json")
        with open(filepath, "w") as f:
            json.dump(recorded_partials, f, indent=2)

        self.status.config(text=f"Stretch curve saved to {filepath}")
        self.plot_stretch_curve()
        messagebox.showinfo("Done", f"Stretch curve saved as {filepath}")

    def plot_stretch_curve(self):
        notes = list(recorded_partials.keys())
        fundamentals = [recorded_partials[n][0] for n in notes]
        nominal = [note_frequencies[n] for n in notes]
        stretch = [1200 * np.log2(f / n) for f, n in zip(fundamentals, nominal)]

        # Convert notes to numeric x for regression
        x = np.array([note_frequencies[n] for n in notes])
        y = np.array(stretch)

        # Use a smoothing spline
        spline = UnivariateSpline(x, y, s=5)  # Adjust s for smoothness
        x_dense = np.linspace(min(x), max(x), 500)
        y_smooth = spline(x_dense)

        self.ax.clear()
        self.ax.plot(x, y, 'o', label="Measured")
        self.ax.plot(x_dense, y_smooth, '-', label="Smoothed")
        self.ax.set_title("Stretch Curve (in cents)")
        self.ax.set_ylabel("Deviation (cents)")
        self.ax.set_xlabel("Frequency (Hz)")
        self.ax.grid(True)
        self.ax.legend()
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = PianoStretchAnalyzer(root)
    root.mainloop()
