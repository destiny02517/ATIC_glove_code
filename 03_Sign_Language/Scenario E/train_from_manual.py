import os

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.interpolate import interp1d
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

# ================= 1. Config =================
CSV_PATH = "manual_annotations.csv"
DATA_DIR = "Sentence proceed"
PRETRAINED_MODEL_PATH = os.path.join("CNN", "FusionSense.keras")
FINE_TUNED_SAVE_PATH = "CNN/FusionSense_FineTuned.keras"

TARGET_LENGTH = 2000
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 5e-5

SENTENCE_ROW_PREFIX = "__SENTENCE__::"
SENTENCE_CONTEXT_RATIO = 0.15
SENTENCE_CONTEXT_PAD_MAX = 250
NUM_CLASSES = 46

# Must match the original 46-class definition.
LABEL_MAP = {
    "Book": 11,
    "Me": 2,
    "You": 1,
    "Where": 6,
    "Better": 24,
    "Now": 37,
    "Sick": 22,
    "My": 4,
    "Father": 3,
    "Know": 14,
    "I": 2,
    "Like": 12,
    "Catsup": 32,
    "Yellow": 27,
    "Wife": 36,
    "Phone": 13,
    "Goodbye": 33,
    "Lesson": 10,
    "Why": 5,
    "Have": 16,
}


def resize_signal(signal, target_len=TARGET_LENGTH):
    if len(signal) < 2:
        return np.zeros(target_len, dtype=np.float32)
    x_old = np.linspace(0, 1, len(signal))
    x_new = np.linspace(0, 1, target_len)
    f = interp1d(x_old, signal, kind="linear")
    return f(x_new).astype(np.float32)


def normalize_signal(clip):
    clip = np.asarray(clip, dtype=np.float32)
    std = np.std(clip)
    if std < 1e-6:
        return clip - np.mean(clip)
    return (clip - np.mean(clip)) / std


def is_sentence_row(label):
    return str(label).startswith(SENTENCE_ROW_PREFIX)


def parse_sentence_words(label):
    label = str(label)
    if not is_sentence_row(label):
        return []
    body = label[len(SENTENCE_ROW_PREFIX) :].strip()
    if not body:
        return []
    return [w.strip() for w in body.split("|") if w.strip()]


def read_annotations(csv_path):
    print(f"Reading annotations: {csv_path}")
    try:
        df = pd.read_csv(
            csv_path,
            header=None,
            names=["file_id", "word_label", "start_idx", "end_idx"],
            encoding="utf-8-sig",
        )
    except Exception as e:
        print(f"Failed to read CSV: {e}")
        return None

    if df.empty:
        print("Annotation CSV is empty.")
        return None

    df["file_id"] = pd.to_numeric(df["file_id"], errors="coerce")
    df["start_idx"] = pd.to_numeric(df["start_idx"], errors="coerce")
    df["end_idx"] = pd.to_numeric(df["end_idx"], errors="coerce")
    df["word_label"] = df["word_label"].astype(str).str.strip()
    df = df.dropna(subset=["file_id", "start_idx", "end_idx", "word_label"]).copy()

    if df.empty:
        print("No valid annotation rows after cleanup.")
        return None

    df["file_id"] = df["file_id"].astype(int).astype(str)
    df["start_idx"] = df["start_idx"].astype(int)
    df["end_idx"] = df["end_idx"].astype(int)
    df = df[df["end_idx"] > df["start_idx"]].copy()
    return df.reset_index(drop=True)


def load_dual_signal(data_dir, fid, cache):
    if fid in cache:
        return cache[fid]

    f_teng = os.path.join(data_dir, f"scope_{fid}_1.csv")
    f_pmut = os.path.join(data_dir, f"scope_{fid}_2.csv")
    if not os.path.exists(f_teng) or not os.path.exists(f_pmut):
        cache[fid] = None
        return None

    try:
        sig_t = pd.read_csv(f_teng, header=None, usecols=[1]).values.flatten()
        sig_p = pd.read_csv(f_pmut, header=None, usecols=[1]).values.flatten()
    except Exception:
        cache[fid] = None
        return None

    min_len = min(len(sig_t), len(sig_p))
    if min_len < 10:
        cache[fid] = None
        return None

    sig_t = np.asarray(sig_t[:min_len], dtype=np.float32)
    sig_p = np.asarray(sig_p[:min_len], dtype=np.float32)
    cache[fid] = (sig_t, sig_p)
    return cache[fid]


def append_sample(sig_t, sig_p, start_idx, end_idx, label_text, label_map, x_t, x_p, y):
    if label_text not in label_map:
        return False

    start_idx = max(0, int(start_idx))
    end_idx = min(len(sig_t), int(end_idx))
    if end_idx - start_idx < 5:
        return False

    clip_t = sig_t[start_idx:end_idx]
    clip_p = sig_p[start_idx:end_idx]
    if len(clip_t) < 5 or len(clip_p) < 5:
        return False

    sample_t = normalize_signal(resize_signal(clip_t, TARGET_LENGTH))
    sample_p = normalize_signal(resize_signal(clip_p, TARGET_LENGTH))
    x_t.append(sample_t)
    x_p.append(sample_p)
    y.append(label_map[label_text])
    return True


def load_manual_data(csv_path, data_dir, label_map):
    df = read_annotations(csv_path)
    if df is None:
        return None, None

    word_df = df[~df["word_label"].apply(is_sentence_row)].copy()
    sentence_df = df[df["word_label"].apply(is_sentence_row)].copy()

    x_teng = []
    x_pmut = []
    labels = []
    signal_cache = {}

    # 1) Word-level samples.
    word_success = 0
    for _, row in word_df.iterrows():
        fid = row["file_id"]
        sig_pair = load_dual_signal(data_dir, fid, signal_cache)
        if sig_pair is None:
            continue
        sig_t, sig_p = sig_pair
        ok = append_sample(
            sig_t,
            sig_p,
            row["start_idx"],
            row["end_idx"],
            row["word_label"],
            label_map,
            x_teng,
            x_pmut,
            labels,
        )
        if ok:
            word_success += 1

    # 2) Sentence-context augmented word samples.
    sentence_aug_success = 0
    if not sentence_df.empty:
        for _, srow in sentence_df.iterrows():
            fid = srow["file_id"]
            sig_pair = load_dual_signal(data_dir, fid, signal_cache)
            if sig_pair is None:
                continue
            sig_t, sig_p = sig_pair

            sent_start = max(0, int(srow["start_idx"]))
            sent_end = min(len(sig_t), int(srow["end_idx"]))
            if sent_end - sent_start < 10:
                continue

            words_in_sentence = parse_sentence_words(srow["word_label"])
            inside_words = word_df[
                (word_df["file_id"] == fid)
                & (word_df["start_idx"] >= sent_start)
                & (word_df["end_idx"] <= sent_end)
            ].sort_values("start_idx")

            if inside_words.empty:
                # Fallback: weakly split sentence by expected word order.
                n_words = len(words_in_sentence)
                if n_words <= 0:
                    continue
                total_len = sent_end - sent_start
                step = max(total_len // n_words, 1)
                for i, w in enumerate(words_in_sentence):
                    if w not in label_map:
                        continue
                    ws = sent_start + i * step
                    we = sent_start + (i + 1) * step if i < n_words - 1 else sent_end
                    if append_sample(sig_t, sig_p, ws, we, w, label_map, x_teng, x_pmut, labels):
                        sentence_aug_success += 1
                continue

            sentence_len = sent_end - sent_start
            pad = int(min(SENTENCE_CONTEXT_PAD_MAX, sentence_len * SENTENCE_CONTEXT_RATIO))
            for i, (_, wrow) in enumerate(inside_words.iterrows()):
                label_text = str(wrow["word_label"])
                if i < len(words_in_sentence) and words_in_sentence[i] in label_map:
                    label_text = words_in_sentence[i]

                ws = max(sent_start, int(wrow["start_idx"]) - pad)
                we = min(sent_end, int(wrow["end_idx"]) + pad)
                if append_sample(sig_t, sig_p, ws, we, label_text, label_map, x_teng, x_pmut, labels):
                    sentence_aug_success += 1

    total = len(labels)
    print(f"Word samples loaded: {word_success}")
    print(f"Sentence-context augmented samples loaded: {sentence_aug_success}")
    print(f"Total samples: {total}")

    if total == 0:
        return None, None

    x_teng = np.asarray(x_teng, dtype=np.float32).reshape(-1, TARGET_LENGTH, 1)
    x_pmut = np.asarray(x_pmut, dtype=np.float32).reshape(-1, TARGET_LENGTH, 1)
    y = tf.keras.utils.to_categorical(np.asarray(labels), num_classes=NUM_CLASSES)
    return [x_teng, x_pmut], y


def main():
    if not os.path.exists(PRETRAINED_MODEL_PATH):
        print(f"Error: pretrained model not found: {PRETRAINED_MODEL_PATH}")
        return

    print("Loading pretrained model...")
    model = load_model(PRETRAINED_MODEL_PATH)

    x_data, y_data = load_manual_data(CSV_PATH, DATA_DIR, LABEL_MAP)
    if x_data is None or y_data is None:
        print("No valid training data. Stop.")
        return

    print(f"Loaded {len(y_data)} samples for fine-tuning.")

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("Start fine-tuning...")
    model.fit(
        x_data,
        y_data,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        shuffle=True,
    )

    os.makedirs("CNN", exist_ok=True)
    model.save(FINE_TUNED_SAVE_PATH)
    print(f"Done. Saved model to: {FINE_TUNED_SAVE_PATH}")


if __name__ == "__main__":
    main()
