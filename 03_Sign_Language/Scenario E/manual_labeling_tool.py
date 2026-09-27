import csv
import glob
import os
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import Button, SpanSelector
from scipy.interpolate import interp1d

# Try interactive backend.
try:
    matplotlib.use("TkAgg")
except Exception:
    pass


# ================= CONFIGURATION =================
DATA_DIR = "Sentence proceed"
OUTPUT_FILE = "manual_annotations.csv"
TARGET_SAMPLES = 10
TARGET_LENGTH = 2000
SENTENCE_ROW_PREFIX = "__SENTENCE__::"

SENTENCE_CONFIG = {
    "155": ["Book", "Me"],
    "156": ["Book", "You"],
    "157": ["Book", "Where"],
    "158": ["Better", "Now"],
    "159": ["Better", "Sick"],
    "160": ["My", "Father", "Know"],
    "161": ["I", "Like", "You"],
    "162": ["I", "Like", "Catsup"],
    "163": ["I", "Like", "Yellow"],
    "164": ["Father", "Sick"],
    "165": ["Wife", "Sick"],
    "166": ["Phone", "Where"],
    "167": ["Goodbye", "You"],
    "168": ["Lesson", "You"],
    "169": ["Why", "Lesson"],
    "170": ["Why", "Lesson"],
    "171": ["You", "Have", "Book"],
}


def resize_signal(signal, target_len):
    if len(signal) < 2:
        return np.zeros(target_len)
    x_old = np.linspace(0, 1, len(signal))
    x_new = np.linspace(0, 1, target_len)
    f = interp1d(x_old, signal, kind="linear")
    return f(x_new)


def normalize_preview(clip):
    if np.std(clip) < 1e-6:
        return clip
    return (clip - np.mean(clip)) / np.std(clip)


class LabelingApp:
    def __init__(self, data_dir, output_file):
        self.data_dir = data_dir
        self.output_file = output_file
        self.pairs = self.load_files()
        self.ids = [fid for fid in sorted(SENTENCE_CONFIG.keys()) if fid in self.pairs]

        if not self.ids:
            raise RuntimeError(f"No valid files found in: {self.data_dir}")

        # Runtime state
        self.current_id_idx = 0
        self.current_group_idx = 0
        self.current_word_idx = 0
        self.current_data = None
        self.view_start = 0
        self.view_width = 15000
        self.selection = None
        self.group_word_ranges = []

        self.ensure_csv_ready()
        self.init_plot()
        self.load_current_file()

    def ensure_csv_ready(self):
        if not os.path.exists(self.output_file):
            with open(self.output_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["file_id", "word_label", "start_idx", "end_idx"])
            return

        # If file exists but is empty, create header.
        if os.path.getsize(self.output_file) == 0:
            with open(self.output_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["file_id", "word_label", "start_idx", "end_idx"])

    def append_annotation(self, fid, label, start_idx, end_idx):
        with open(self.output_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([fid, label, int(start_idx), int(end_idx)])

    def load_files(self):
        files = glob.glob(os.path.join(self.data_dir, "scope_*_*.csv"))
        pairs = {}
        pattern = re.compile(r"scope_(\d+)_(\d+)")
        for fpath in files:
            m = pattern.search(os.path.basename(fpath))
            if not m:
                continue
            fid, ch = m.group(1), m.group(2)
            pairs.setdefault(fid, {})[ch] = fpath
        return pairs

    def init_plot(self):
        self.fig = plt.figure(figsize=(16, 9))
        self.ax_main = plt.axes([0.05, 0.3, 0.9, 0.6])
        self.ax_prev = plt.axes([0.75, 0.05, 0.2, 0.2])
        self.ax_prev.set_title("Standardized (2000 pts)", fontsize=9)
        self.ax_prev.set_xticks([])
        self.ax_prev.set_yticks([])

        btn_y = 0.1
        btn_h = 0.08
        btn_w = 0.1

        self.btn_prev_id = Button(plt.axes([0.05, btn_y, btn_w, btn_h]), "<< Prev ID")
        self.btn_prev_id.on_clicked(self.prev_id)

        self.btn_next_id = Button(plt.axes([0.16, btn_y, btn_w, btn_h]), "Next ID >>")
        self.btn_next_id.on_clicked(self.next_id)

        self.btn_scroll = Button(plt.axes([0.27, btn_y, btn_w, btn_h]), "Scroll View ->")
        self.btn_scroll.on_clicked(self.scroll_view)

        self.btn_confirm = Button(
            plt.axes([0.45, btn_y, 0.15, btn_h]),
            "CONFIRM SELECTION",
            color="lightgreen",
            hovercolor="green",
        )
        self.btn_confirm.on_clicked(self.confirm_selection)

        self.btn_skip = Button(plt.axes([0.61, btn_y, btn_w, btn_h]), "Skip Word")
        self.btn_skip.on_clicked(self.skip_word)

        self.span = SpanSelector(
            self.ax_main,
            self.on_select,
            "horizontal",
            useblit=True,
            props=dict(alpha=0.3, facecolor="red"),
            interactive=True,
            drag_from_anywhere=True,
        )

    def load_current_file(self):
        fid = self.ids[self.current_id_idx]
        self.group_word_ranges = []

        try:
            if "1" not in self.pairs[fid]:
                raise FileNotFoundError(f"Missing channel 1 for ID {fid}")
            df = pd.read_csv(self.pairs[fid]["1"], header=None, usecols=[1], dtype=np.float32)
            raw = df.values.flatten()
            self.current_data = raw - np.mean(raw)
            self.view_start = 0
            self.update_plot()
        except Exception as e:
            print(f"Error loading {fid}: {e}")

    def update_plot(self):
        fid = self.ids[self.current_id_idx]
        words = SENTENCE_CONFIG[fid]
        curr_word = words[self.current_word_idx]

        self.ax_main.clear()

        view_end = min(self.view_start + self.view_width, len(self.current_data))
        x_axis = np.arange(self.view_start, view_end)
        y_data = self.current_data[self.view_start:view_end]

        self.ax_main.plot(x_axis, y_data, color="#1f77b4", linewidth=1)
        self.ax_main.grid(True, alpha=0.3)
        self.ax_main.set_xlim(self.view_start, view_end)
        self.ax_main.axhline(0, color="gray", linestyle="--", alpha=0.5)

        sentence_str = "[" + ", ".join(words) + "]"
        target_str = f">>> {curr_word.upper()} <<<"
        title_txt = (
            f"ID: {fid} | Sentence: {sentence_str}\n"
            f"Progress: Group {self.current_group_idx + 1}/{TARGET_SAMPLES} | "
            f"Word {self.current_word_idx + 1}/{len(words)}\n"
            f"ACTION: Select the range for {target_str}"
        )
        self.ax_main.set_title(title_txt, fontsize=14, fontweight="bold", color="darkred")

        self.selection = None
        self.ax_prev.clear()
        self.ax_prev.set_title("Preview (2000 pts)")
        self.fig.canvas.draw_idle()

    def on_select(self, xmin, xmax):
        start, end = int(xmin), int(xmax)
        if end - start < 5:
            return

        self.selection = (start, end)

        clip = self.current_data[start:end]
        resized = resize_signal(clip, TARGET_LENGTH)
        normalized = normalize_preview(resized)

        self.ax_prev.clear()
        self.ax_prev.plot(normalized, color="orange", linewidth=1.5)
        self.ax_prev.set_title(f"Input for CNN: {len(clip)}->{TARGET_LENGTH}", fontsize=10, color="green")
        self.ax_prev.grid(True, alpha=0.2)
        self.fig.canvas.draw_idle()

    def confirm_selection(self, _event):
        if self.selection is None:
            print("No selection to confirm.")
            return

        fid = self.ids[self.current_id_idx]
        words = SENTENCE_CONFIG[fid]
        curr_word = words[self.current_word_idx]
        s, e = self.selection

        self.append_annotation(fid, curr_word, s, e)
        self.group_word_ranges.append((curr_word, s, e))
        print(f"Saved word: ID={fid}, label={curr_word}, range=[{s}:{e}]")
        self.advance_step()

    def skip_word(self, _event):
        print("Skipped current word.")
        self.advance_step()

    def save_sentence_annotation_for_group(self, fid, words):
        if len(self.group_word_ranges) != len(words):
            print("Group incomplete: sentence-level annotation not saved for this group.")
            return

        starts = [s for _, s, _ in self.group_word_ranges]
        ends = [e for _, _, e in self.group_word_ranges]
        sent_start, sent_end = min(starts), max(ends)
        sentence_label = SENTENCE_ROW_PREFIX + "|".join(words)
        self.append_annotation(fid, sentence_label, sent_start, sent_end)
        print(f"Saved sentence: ID={fid}, label={sentence_label}, range=[{sent_start}:{sent_end}]")

    def advance_step(self):
        fid = self.ids[self.current_id_idx]
        words = SENTENCE_CONFIG[fid]

        self.current_word_idx += 1

        if self.current_word_idx >= len(words):
            self.save_sentence_annotation_for_group(fid, words)
            self.group_word_ranges = []

            self.current_word_idx = 0
            self.current_group_idx += 1
            print(f"--- Group {self.current_group_idx} Finished ---")

            self.view_start += 3500

            if self.current_group_idx >= TARGET_SAMPLES:
                print(f"ID {fid}: all groups collected. Switching to next ID.")
                self.next_id(None)
                return

        self.update_plot()

    def scroll_view(self, _event):
        self.view_start += int(self.view_width * 0.7)
        if self.view_start >= len(self.current_data):
            self.view_start = 0
        self.update_plot()

    def next_id(self, _event):
        if self.current_id_idx < len(self.ids) - 1:
            self.current_id_idx += 1
            self.current_group_idx = 0
            self.current_word_idx = 0
            self.load_current_file()
            return

        self.ax_main.clear()
        self.ax_main.text(
            0.5,
            0.5,
            "MISSION ACCOMPLISHED!\nAll Files Annotated.",
            ha="center",
            va="center",
            fontsize=24,
            color="green",
            fontweight="bold",
        )
        self.fig.canvas.draw_idle()

    def prev_id(self, _event):
        if self.current_id_idx > 0:
            self.current_id_idx -= 1
            self.current_group_idx = 0
            self.current_word_idx = 0
            self.load_current_file()


if __name__ == "__main__":
    app = LabelingApp(DATA_DIR, OUTPUT_FILE)
    plt.show()
