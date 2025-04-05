import tkinter as tk
from tkinter import filedialog, messagebox
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
AMPLITUDE_THRESHOLD = 0.0015  # lowered threshold for softer note detection

NOTES_TO_SAMPLE = ["C2", "C3", "C4", "C5", "C6", "C7", "E2", "E3", "E4", "E5", "E6", "E7", "F2", "F3", "F4", "F5", "F6", "F7", "A2", "A3", "A4", "A5", "A6", "A7"]
note_frequencies = {
    "C2": 65.41, "C3": 130.81, "C4": 261.63, "C5": 523.25, "C6": 1046.50, "C7": 2093.00,
    "E2": 82.41, "E3": 164.81, "E4": 329.63, "E5": 659.26, "E6": 1318.51, "E7": 2637.02,
    "F2": 87.31, "F3": 174.61, "F4": 349.23, "F5": 698.46, "F6": 1396.91, "F7": 2793.83,
    "A2": 110.00, "A3": 220.00, "A4": 440.00, "A5": 880.00, "A6": 1760.00, "A7": 3520.00
}

recorded_partials = {}


class MainMenu:
    def __init__(self, master):
        self.master = master
        self.master.title("Piano Tuner")
        self.label = tk.Label(master, text="Welcome to Piano Tuner", font=("Arial", 16))
        self.label.pack(pady=20)

        self.analyze_button = tk.Button(master, text="Analyze", command=self.open_analyzer, font=("Arial", 14))
        self.analyze_button.pack(pady=10)

        self.tune_button = tk.Button(master, text="Tune", command=self.open_tuner, font=("Arial", 14))
        self.tune_button.pack(pady=10)

    def open_analyzer(self):
        self.clear_window()
        PianoStretchAnalyzer(self.master, self.return_to_menu)

    def open_tuner(self):
        self.clear_window()
        PianoTuner(self.master, self.return_to_menu)

    def clear_window(self):
        for widget in self.master.winfo_children():
            widget.destroy()

    def return_to_menu(self):
        self.clear_window()
        MainMenu(self.master)


class PianoStretchAnalyzer:
    def __init__(self, master, return_callback):
        self.master = master
        self.return_callback = return_callback
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

        self.back_button = tk.Button(master, text="Back to Menu", command=self.return_callback)
        self.back_button.pack(pady=10)

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

        audio = self.record_note()
        rms = np.sqrt(np.mean(audio ** 2))

        if rms < AMPLITUDE_THRESHOLD:
            self.status.config(text=f"Too quiet. Please play {note} louder.")
            self.master.after(3000, self.record_next)
            return

        target_freq = note_frequencies[note]
        partials = self.extract_partials(audio, target_freq)
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

        x = np.array([note_frequencies[n] for n in notes])
        y = np.array(stretch)

        x_filtered, y_filtered = remove_outliers(x, y)

        spline = UnivariateSpline(x_filtered, y_filtered, s=5)
        x_dense = np.linspace(min(x_filtered), max(x_filtered), 500)
        y_smooth = spline(x_dense)

        self.ax.clear()
        self.ax.plot(x_filtered, y_filtered, 'o', label="Measured")
        self.ax.plot(x_dense, y_smooth, '-', label="Smoothed")
        self.ax.set_title("Stretch Curve (in cents)")
        self.ax.set_ylabel("Deviation (cents)")
        self.ax.set_xlabel("Frequency (Hz)")
        self.ax.grid(True)
        self.ax.legend()
        self.canvas.draw()

    def record_note(self):
        audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
        sd.wait()
        return audio.flatten()

    def extract_partials(self, audio, target_freq):
        spectrum = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
        freqs = np.fft.rfftfreq(len(audio), 1 / SAMPLE_RATE)

        partials = []
        for n in range(1, 7):
            expected = target_freq * n
            idx = np.argmin(np.abs(freqs - expected))
            partial_freq = freqs[idx]
            partials.append(partial_freq)
        return partials


class PianoTuner:
    def __init__(self, master, return_callback):
        self.master = master
        self.return_callback = return_callback
        self.current_note_index = 0
        self.notes_to_tune = []
        self.stretch_curve = {}

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

        self.back_button = tk.Button(master, text="Back to Menu", command=self.return_callback)
        self.back_button.pack(pady=10)

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

        middle_c_index = NOTES.index("C4")
        self.notes_to_tune = (
            NOTES[middle_c_index:] + NOTES[:middle_c_index][::-1]
        )
        self.notes_to_tune = [note for note in self.notes_to_tune if note in self.stretch_curve]

        self.current_note_index = 0
        self.label.config(text=f"Loaded stretch curve: {os.path.basename(filepath)}")
        self.status.config(text="Press 'Next Note' to start tuning.")
        self.next_button.config(state=tk.NORMAL)
        self.previous_button.config(state=tk.DISABLED)

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

        if self.current_note_index == 0:
            self.previous_button.config(state=tk.DISABLED)


def remove_outliers(x, y):
    q1, q3 = np.percentile(y, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    mask = (y >= lower_bound) & (y <= upper_bound)
    return x[mask], y[mask]


if __name__ == "__main__":
    root = tk.Tk()
    MainMenu(root)
    root.mainloop()