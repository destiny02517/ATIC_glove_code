import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from nnom import generate_model  # type: ignore
from scipy.interpolate import interp1d
from tensorflow.keras import callbacks, layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model, to_categorical


DATA_DIR = "Sentence proceed"
CSV_PATH = "manual_annotations.csv"
OUTPUT_ROOT = "sentence_edge_output"
QUANTIZED_SIGNAL_DIR = os.path.join(OUTPUT_ROOT, "quantized_full_signals")
QUANTIZED_CLIP_DIR = os.path.join(OUTPUT_ROOT, "quantized_sentence_clips")
MODEL_DIR = os.path.join(OUTPUT_ROOT, "models")
REPORT_DIR = os.path.join(OUTPUT_ROOT, "reports")
TEST_DATASET_DIR = os.path.join(OUTPUT_ROOT, "mcu_test_dataset")
TEST_HEADER_DIR = os.path.join(TEST_DATASET_DIR, "headers")
MCU_INFERENCE_DIR = os.path.join(TEST_DATASET_DIR, "nnom_input")

SENTENCE_ROW_PREFIX = "__SENTENCE__::"
TARGET_LEN = 2000
RANDOM_SEED = 42
BATCH_SIZE = 16
EPOCHS = 150
LEARNING_RATE = 1e-4
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2
TEST_SAMPLES_PER_CLASS = 5
MIN_CLIP_LEN = 20

SENTENCE_CLASS_CONFIG = {
    "155": {"category": "S01", "words": ["Book", "Me"], "text": "This is my book."},
    "156": {"category": "S02", "words": ["Book", "You"], "text": "Is this your book?"},
    "157": {"category": "S03", "words": ["Book", "Where"], "text": "Where is the book?"},
    "166": {"category": "S04", "words": ["Phone", "Where"], "text": "Where is the phone?"},
    "167": {"category": "S05", "words": ["Goodbye", "You"], "text": "Goodbye to you."},
    "168": {"category": "S06", "words": ["Lesson", "You"], "text": "You have a lesson."},
    "170": {"category": "S07", "words": ["Why", "Lesson"], "text": "Why do we have a lesson?"},
}

CLASS_IDS = list(SENTENCE_CLASS_CONFIG.keys())
CLASS_NAMES = [SENTENCE_CLASS_CONFIG[fid]["category"] for fid in CLASS_IDS]
ID_TO_CLASS_IDX = {fid: idx for idx, fid in enumerate(CLASS_IDS)}
NUM_CLASSES = len(CLASS_IDS)


def ensure_dirs():
    for path in [
        OUTPUT_ROOT,
        QUANTIZED_SIGNAL_DIR,
        QUANTIZED_CLIP_DIR,
        MODEL_DIR,
        REPORT_DIR,
        TEST_DATASET_DIR,
        TEST_HEADER_DIR,
        MCU_INFERENCE_DIR,
    ]:
        os.makedirs(path, exist_ok=True)


def parse_sentence_words(label):
    label = str(label)
    if not label.startswith(SENTENCE_ROW_PREFIX):
        return []
    body = label[len(SENTENCE_ROW_PREFIX) :].strip()
    if not body:
        return []
    return [word.strip() for word in body.split("|") if word.strip()]


def read_sentence_annotations(csv_path):
    df = pd.read_csv(
        csv_path,
        header=None,
        names=["file_id", "word_label", "start_idx", "end_idx"],
        encoding="utf-8-sig",
    )
    df["file_id"] = pd.to_numeric(df["file_id"], errors="coerce")
    df["start_idx"] = pd.to_numeric(df["start_idx"], errors="coerce")
    df["end_idx"] = pd.to_numeric(df["end_idx"], errors="coerce")
    df["word_label"] = df["word_label"].astype(str).str.strip()
    df = df.dropna(subset=["file_id", "start_idx", "end_idx", "word_label"]).copy()
    df["file_id"] = df["file_id"].astype(int).astype(str)
    df["start_idx"] = df["start_idx"].astype(int)
    df["end_idx"] = df["end_idx"].astype(int)
    df = df[df["end_idx"] > df["start_idx"]].copy()
    df = df[df["word_label"].str.startswith(SENTENCE_ROW_PREFIX)].copy()
    df = df[df["file_id"].isin(CLASS_IDS)].copy()

    keep_rows = []
    for _, row in df.iterrows():
        fid = row["file_id"]
        words = parse_sentence_words(row["word_label"])
        if words == SENTENCE_CLASS_CONFIG[fid]["words"]:
            keep_rows.append(row)

    if not keep_rows:
        return pd.DataFrame(columns=df.columns)
    return pd.DataFrame(keep_rows).sort_values(["file_id", "start_idx"]).reset_index(drop=True)


def load_dual_signals(file_id):
    f_teng = os.path.join(DATA_DIR, f"scope_{file_id}_1.csv")
    f_pmut = os.path.join(DATA_DIR, f"scope_{file_id}_2.csv")
    if not os.path.exists(f_teng) or not os.path.exists(f_pmut):
        return None, None

    sig_t = pd.read_csv(f_teng, header=None, usecols=[1]).values.flatten().astype(np.float32)
    sig_p = pd.read_csv(f_pmut, header=None, usecols=[1]).values.flatten().astype(np.float32)
    min_len = min(len(sig_t), len(sig_p))
    if min_len < MIN_CLIP_LEN:
        return None, None
    return sig_t[:min_len], sig_p[:min_len]


def compute_channel_quant_params(signal_map):
    teng_vals = []
    pmut_vals = []
    for sig_t, sig_p in signal_map.values():
        teng_vals.append(sig_t)
        pmut_vals.append(sig_p)

    teng_all = np.concatenate(teng_vals).astype(np.float32)
    pmut_all = np.concatenate(pmut_vals).astype(np.float32)

    return {
        "teng": build_quant_params(teng_all),
        "pmut": build_quant_params(pmut_all),
    }


def build_quant_params(values):
    v_min = float(np.min(values))
    v_max = float(np.max(values))
    if abs(v_max - v_min) < 1e-12:
        scale = 1.0
        zero_point = 0
    else:
        scale = float((v_max - v_min) / 255.0)
        zero_point = int(np.clip(np.round(-128.0 - v_min / scale), -128, 127))
    return {"min": v_min, "max": v_max, "scale": scale, "zero_point": zero_point}


def quantize_signal(values, params):
    scale = float(params["scale"])
    zero_point = int(params["zero_point"])
    if scale <= 0:
        return np.zeros_like(values, dtype=np.int8)
    q = np.round(values / scale + zero_point)
    return np.clip(q, -128, 127).astype(np.int8)


def normalize_signal(signal):
    signal = np.asarray(signal, dtype=np.float32)
    if signal.size == 0:
        return signal
    std = float(np.std(signal))
    if std < 1e-8:
        return signal - float(np.mean(signal))
    return (signal - float(np.mean(signal))) / std


def normalize_by_absmax(signal):
    signal = np.asarray(signal, dtype=np.float32)
    max_abs = float(np.max(np.abs(signal)))
    if max_abs < 1e-8:
        return None
    return signal / max_abs


def resize_signal(signal, target_len):
    signal = np.asarray(signal, dtype=np.float32)
    if len(signal) < 2:
        return np.zeros(target_len, dtype=np.float32)
    f = interp1d(np.linspace(0, 1, len(signal)), signal, kind="linear")
    return f(np.linspace(0, 1, target_len)).astype(np.float32)


def build_input_from_quantized(clip_t, clip_p, target_len=TARGET_LEN):
    feat_t = resize_signal(clip_t.astype(np.float32), target_len) / 127.0
    feat_p = resize_signal(clip_p.astype(np.float32), target_len) / 127.0
    return np.stack([feat_t, feat_p], axis=-1).astype(np.float32)


def to_mcu_input_int8(sample):
    sample = np.asarray(sample, dtype=np.float32)
    return np.clip(np.round(sample * 127.0), -128, 127).astype(np.int8)


def flatten_hwc_sample(sample_int8):
    sample_int8 = np.asarray(sample_int8, dtype=np.int8)
    return sample_int8.reshape(-1)


def save_quantized_full_signals(signal_map, quant_params):
    manifest_rows = []
    for fid, (sig_t, sig_p) in signal_map.items():
        q_t = quantize_signal(sig_t, quant_params["teng"])
        q_p = quantize_signal(sig_p, quant_params["pmut"])

        fid_dir = os.path.join(QUANTIZED_SIGNAL_DIR, f"ID_{fid}")
        os.makedirs(fid_dir, exist_ok=True)

        path_t = os.path.join(fid_dir, f"scope_{fid}_1_quantized.csv")
        path_p = os.path.join(fid_dir, f"scope_{fid}_2_quantized.csv")
        pd.DataFrame({"value": q_t.astype(np.int16)}).to_csv(path_t, index=False)
        pd.DataFrame({"value": q_p.astype(np.int16)}).to_csv(path_p, index=False)

        manifest_rows.append(
            {
                "file_id": fid,
                "category": SENTENCE_CLASS_CONFIG[fid]["category"],
                "teng_path": path_t,
                "pmut_path": path_p,
                "num_samples": int(len(q_t)),
            }
        )

    pd.DataFrame(manifest_rows).to_csv(
        os.path.join(QUANTIZED_SIGNAL_DIR, "full_signal_manifest.csv"), index=False
    )


def build_detection_template(segments):
    valid_segments = [seg for seg in segments if len(seg) >= 2]
    if not valid_segments:
        return None, 0

    lengths = [len(seg) for seg in valid_segments]
    target_len = max(int(np.median(lengths)), MIN_CLIP_LEN)
    resized = [normalize_signal(resize_signal(seg, target_len)) for seg in valid_segments]
    template = normalize_signal(np.mean(np.stack(resized, axis=0), axis=0))
    return template.astype(np.float32), target_len


def adaptive_threshold(correlation):
    correlation = np.asarray(correlation, dtype=np.float32)
    return max(float(np.mean(correlation) + 1.5 * np.std(correlation)), 0.15)


def detect_global_cycles(sig_t, sig_p, template_t, template_p, template_len):
    norm_sig_t = normalize_signal(sig_t)
    norm_sig_p = normalize_signal(sig_p)
    corr_t = normalize_by_absmax(np.correlate(norm_sig_t, template_t, mode="valid"))
    corr_p = normalize_by_absmax(np.correlate(norm_sig_p, template_p, mode="valid"))
    if corr_t is None and corr_p is None:
        return np.array([], dtype=np.int64), {"thresh": 0.0, "min_distance": 0}
    if corr_t is None:
        corr_t = np.zeros_like(corr_p)
    if corr_p is None:
        corr_p = np.zeros_like(corr_t)

    correlation = 0.60 * corr_t + 0.40 * corr_p
    base_thresh = adaptive_threshold(correlation)
    min_distance = max(int(template_len * 0.70), 100)

    best_peaks = np.array([], dtype=np.int64)
    best_thresh = base_thresh
    for factor in [1.00, 0.90, 0.80, 0.70, 0.60, 0.50]:
        trial_thresh = max(base_thresh * factor, 0.09)
        peaks = []
        last_peak = -10**18
        for i in range(1, len(correlation) - 1):
            if correlation[i] < trial_thresh:
                continue
            if correlation[i] < correlation[i - 1] or correlation[i] < correlation[i + 1]:
                continue
            if i - last_peak < min_distance:
                if len(peaks) > 0 and correlation[i] > correlation[peaks[-1]]:
                    peaks[-1] = i
                    last_peak = i
                continue
            peaks.append(i)
            last_peak = i
        peaks = np.asarray(peaks, dtype=np.int64)
        if len(peaks) > len(best_peaks):
            best_peaks = peaks
            best_thresh = trial_thresh

    peaks = np.sort(best_peaks)
    if len(peaks) > 0:
        peak_scores = correlation[peaks]
        quality_floor = max(float(np.median(peak_scores) * 0.75), best_thresh * 0.6)
        peaks = peaks[peak_scores >= quality_floor]
    return np.asarray(np.sort(peaks), dtype=np.int64), {"thresh": best_thresh, "min_distance": min_distance}


def build_detection_references(sentence_df, signal_map):
    refs = {}
    for fid in CLASS_IDS:
        class_rows = sentence_df[sentence_df["file_id"] == fid].sort_values("start_idx")
        if class_rows.empty:
            continue

        sig_t, sig_p = signal_map[fid]
        clips_t = []
        clips_p = []
        for _, row in class_rows.iterrows():
            start_idx = max(0, int(row["start_idx"]))
            end_idx = min(len(sig_t), int(row["end_idx"]))
            if end_idx - start_idx < MIN_CLIP_LEN:
                continue
            clips_t.append(sig_t[start_idx:end_idx])
            clips_p.append(sig_p[start_idx:end_idx])

        template_t, len_t = build_detection_template(clips_t)
        template_p, len_p = build_detection_template(clips_p)
        if template_t is None or template_p is None:
            continue

        refs[fid] = {
            "template_t": template_t,
            "template_p": template_p,
            "template_len": max(int(np.median([len_t, len_p])), MIN_CLIP_LEN),
            "n_ref": len(clips_t),
        }
    return refs


def save_detection_reference_report(detection_refs):
    rows = []
    for fid in CLASS_IDS:
        if fid not in detection_refs:
            continue
        ref = detection_refs[fid]
        rows.append(
            {
                "file_id": fid,
                "category": SENTENCE_CLASS_CONFIG[fid]["category"],
                "reference_cycles": int(ref["n_ref"]),
                "template_len": int(ref["template_len"]),
            }
        )
    pd.DataFrame(rows).to_csv(
        os.path.join(REPORT_DIR, "detection_reference_summary.csv"), index=False
    )


def save_quantized_sentence_clips(signal_map, detection_refs, quant_params):
    clip_records = []
    clip_counter_by_id = {fid: 0 for fid in CLASS_IDS}

    for fid in CLASS_IDS:
        if fid not in detection_refs:
            continue
        sig_t, sig_p = signal_map[fid]
        det_ref = detection_refs[fid]
        peaks, det_info = detect_global_cycles(
            sig_t,
            sig_p,
            det_ref["template_t"],
            det_ref["template_p"],
            det_ref["template_len"],
        )
        print(
            f"ID {fid} detected cycles={len(peaks)} "
            f"(ref={det_ref['n_ref']}, thresh={det_info['thresh']:.3f}, "
            f"min_distance={det_info['min_distance']})"
        )

        for peak in peaks:
            start_idx = max(0, int(peak))
            end_idx = min(len(sig_t), start_idx + int(det_ref["template_len"]))
            if end_idx - start_idx < MIN_CLIP_LEN:
                continue

            clip_counter_by_id[fid] += 1
            clip_idx = clip_counter_by_id[fid]
            category = SENTENCE_CLASS_CONFIG[fid]["category"]
            clip_dir = os.path.join(QUANTIZED_CLIP_DIR, category)
            os.makedirs(clip_dir, exist_ok=True)

            q_t = quantize_signal(sig_t[start_idx:end_idx], quant_params["teng"])
            q_p = quantize_signal(sig_p[start_idx:end_idx], quant_params["pmut"])
            clip_name = f"{category}_ID_{fid}_clip_{clip_idx:03d}.csv"
            clip_path = os.path.join(clip_dir, clip_name)
            pd.DataFrame({"teng": q_t.astype(np.int16), "pmut": q_p.astype(np.int16)}).to_csv(
                clip_path, index=False
            )

            clip_records.append(
                {
                    "file_id": fid,
                    "category": category,
                    "class_idx": ID_TO_CLASS_IDX[fid],
                    "clip_index": clip_idx,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "clip_len": int(end_idx - start_idx),
                    "clip_path": clip_path,
                }
            )

    clip_df = pd.DataFrame(clip_records)
    clip_df.to_csv(os.path.join(QUANTIZED_CLIP_DIR, "clip_manifest.csv"), index=False)
    return clip_df


def split_indices_by_class(labels, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, dtype=np.int64)
    train_idx = []
    val_idx = []
    test_idx = []

    for cls_idx in range(NUM_CLASSES):
        idx = np.where(labels == cls_idx)[0]
        if len(idx) == 0:
            continue
        idx = rng.permutation(idx)

        n_total = len(idx)
        if n_total == 1:
            train_idx.extend(idx.tolist())
            continue
        if n_total == 2:
            train_idx.append(int(idx[0]))
            test_idx.append(int(idx[1]))
            continue

        n_test = max(1, int(round(n_total * TEST_RATIO)))
        n_test = min(n_test, max(1, n_total - 2))
        if n_total >= TEST_SAMPLES_PER_CLASS + 2:
            n_test = max(n_test, min(TEST_SAMPLES_PER_CLASS, n_total - 2))

        remaining = n_total - n_test
        n_val = max(1, int(round(n_total * VAL_RATIO)))
        n_val = min(n_val, max(1, remaining - 1))
        n_train = n_total - n_test - n_val

        if n_train < 1:
            deficit = 1 - n_train
            n_train = 1
            if n_val > deficit:
                n_val -= deficit
            else:
                n_test = max(1, n_test - (deficit - max(0, n_val - 1)))
                n_val = max(1, n_val)
            n_train = n_total - n_test - n_val

        train_idx.extend(idx[:n_train].tolist())
        val_idx.extend(idx[n_train : n_train + n_val].tolist())
        test_idx.extend(idx[n_train + n_val : n_train + n_val + n_test].tolist())

    return (
        np.asarray(sorted(train_idx), dtype=np.int64),
        np.asarray(sorted(val_idx), dtype=np.int64),
        np.asarray(sorted(test_idx), dtype=np.int64),
    )


def prepare_dataset(clip_df):
    features = []
    labels = []
    paths = []
    for _, row in clip_df.iterrows():
        clip = pd.read_csv(row["clip_path"])
        clip_t = clip["teng"].to_numpy(dtype=np.int16).astype(np.int8)
        clip_p = clip["pmut"].to_numpy(dtype=np.int16).astype(np.int8)
        features.append(build_input_from_quantized(clip_t, clip_p, target_len=TARGET_LEN))
        labels.append(int(row["class_idx"]))
        paths.append(row["clip_path"])

    x = np.asarray(features, dtype=np.float32)
    y_idx = np.asarray(labels, dtype=np.int64)
    y = to_categorical(y_idx, num_classes=NUM_CLASSES)
    return x, y_idx, y, paths


def save_split_manifest(clip_df, train_idx, val_idx, test_idx):
    split_labels = np.full(len(clip_df), "unused", dtype=object)
    split_labels[train_idx] = "train"
    split_labels[val_idx] = "val"
    split_labels[test_idx] = "test"

    split_df = clip_df.copy()
    split_df["split"] = split_labels
    split_df.to_csv(os.path.join(QUANTIZED_CLIP_DIR, "clip_manifest_with_split.csv"), index=False)
    return split_df


def print_split_summary(y_idx, train_idx, val_idx, test_idx):
    train_labels = y_idx[train_idx]
    val_labels = y_idx[val_idx]
    test_labels = y_idx[test_idx]

    print("Per-class split summary")
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        train_count = int(np.sum(train_labels == cls_idx))
        val_count = int(np.sum(val_labels == cls_idx))
        test_count = int(np.sum(test_labels == cls_idx))
        total_count = int(np.sum(y_idx == cls_idx))
        print(
            f"  {cls_name}: train={train_count}, val={val_count}, "
            f"test={test_count}, total={total_count}"
        )


def build_sentence_model(input_shape=(TARGET_LEN, 2), num_classes=NUM_CLASSES):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(3, kernel_size=2, padding="same")(inputs)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(pool_size=2, strides=2, padding="same")(x)
    x = layers.Conv1D(6, kernel_size=2, padding="same")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(pool_size=2, strides=2, padding="same")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Flatten()(x)
    x = layers.Activation("relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs=inputs, outputs=outputs, name="whole_sentence_edge_cnn")


def plot_history(history, save_path):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train Loss", color="#0b6e4f")
    plt.plot(history.history["val_loss"], label="Validation Loss", color="#b80c09")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(False)

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Train Accuracy", color="#0b6e4f")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy", color="#b80c09")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def compute_confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def plot_confusion_matrix(cm, class_names, accuracy, save_path):
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm * 100.0, row_sum, out=np.zeros_like(cm, dtype=np.float32), where=row_sum > 0)

    plt.figure(figsize=(9, 7))
    plt.imshow(cm_pct, cmap="Greens", vmin=0, vmax=100)
    plt.colorbar(label="Row-wise Accuracy (%)")
    plt.xticks(np.arange(len(class_names)), class_names)
    plt.yticks(np.arange(len(class_names)), class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Whole Sentence Confusion Matrix\nAccuracy: {accuracy:.2f}%")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm_pct[i, j] >= 50 else "black"
            plt.text(
                j,
                i,
                f"{cm[i, j]}\n{cm_pct[i, j]:.1f}%",
                ha="center",
                va="center",
                color=color,
                fontsize=10,
                fontweight="bold",
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_prediction_reports(model, x_test, y_test_idx, test_paths):
    probs = model.predict(x_test, verbose=0)
    preds = np.argmax(probs, axis=1)
    accuracy = float(np.mean(preds == y_test_idx) * 100.0) if len(y_test_idx) > 0 else 0.0

    cm = compute_confusion_matrix(y_test_idx, preds, NUM_CLASSES)
    plot_confusion_matrix(
        cm,
        CLASS_NAMES,
        accuracy,
        os.path.join(REPORT_DIR, "whole_sentence_confusion_matrix.png"),
    )

    prediction_one_hot = np.zeros((len(preds), NUM_CLASSES), dtype=np.int32)
    prediction_one_hot[np.arange(len(preds)), preds] = 1
    np.savetxt(
        os.path.join(REPORT_DIR, "accuracy_fusion.txt"),
        prediction_one_hot,
        fmt="%d",
        delimiter=" ",
    )

    rows = []
    for path, true_idx, pred_idx, prob in zip(test_paths, y_test_idx, preds, probs):
        rows.append(
            {
                "clip_path": path,
                "true_idx": int(true_idx),
                "true_name": CLASS_NAMES[int(true_idx)],
                "pred_idx": int(pred_idx),
                "pred_name": CLASS_NAMES[int(pred_idx)],
                "pred_confidence": float(prob[int(pred_idx)]),
                "correct": int(int(true_idx) == int(pred_idx)),
            }
        )
    pd.DataFrame(rows).to_csv(os.path.join(REPORT_DIR, "test_predictions.csv"), index=False)
    return accuracy, preds, cm


def export_test_dataset_folder(x_test, y_test_idx, test_paths):
    rows = []
    for i, (sample, true_idx, source_path) in enumerate(zip(x_test, y_test_idx, test_paths)):
        sample_int8 = to_mcu_input_int8(sample)
        sample_2d = sample_int8.reshape(TARGET_LEN, 2)
        category = CLASS_NAMES[int(true_idx)]
        file_name = f"test_{i:03d}_{category}.csv"
        save_path = os.path.join(TEST_DATASET_DIR, file_name)

        pd.DataFrame(
            {
                "teng": sample_2d[:, 0].astype(np.int16),
                "pmut": sample_2d[:, 1].astype(np.int16),
            }
        ).to_csv(save_path, index=False)

        rows.append(
            {
                "test_index": i,
                "file_name": file_name,
                "true_idx": int(true_idx),
                "true_name": category,
                "source_clip_path": source_path,
            }
        )

    manifest_path = os.path.join(TEST_DATASET_DIR, "test_dataset_manifest.csv")
    pd.DataFrame(rows).to_csv(manifest_path, index=False)

    label_map_path = os.path.join(TEST_DATASET_DIR, "label_map.csv")
    pd.DataFrame(
        [{"class_idx": idx, "class_name": name} for idx, name in enumerate(CLASS_NAMES)]
    ).to_csv(label_map_path, index=False)
    return manifest_path


def export_mcu_test_headers_by_class(x_test, y_test_idx):
    header_paths = []

    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_samples = x_test[y_test_idx == class_idx]
        if len(class_samples) == 0:
            continue

        class_samples_int8 = to_mcu_input_int8(class_samples)
        class_samples_int8 = class_samples_int8.reshape(len(class_samples_int8), TARGET_LEN, 2)

        teng_data = class_samples_int8[:, :, 0]
        pmut_data = class_samples_int8[:, :, 1]
        sample_count = len(class_samples_int8)
        array_prefix = class_name.lower()
        header_path = os.path.join(TEST_HEADER_DIR, f"{array_prefix}_test_data.h")

        with open(header_path, "w", encoding="utf-8") as f:
            guard = f"__{array_prefix.upper()}_TEST_DATA_H__"
            f.write("#ifndef " + guard + "\n")
            f.write("#define " + guard + "\n\n")
            f.write("#include <stdint.h>\n\n")
            f.write(f"#define {array_prefix.upper()}_TEST_SAMPLE_COUNT {sample_count}\n")
            f.write(f"#define {array_prefix.upper()}_TEST_SAMPLE_LEN {TARGET_LEN}\n\n")

            f.write(
                f"static const int8_t {array_prefix}_teng[{sample_count}][{TARGET_LEN}] = {{\n"
            )
            for sample_idx, sample in enumerate(teng_data):
                suffix = "," if sample_idx < sample_count - 1 else ""
                f.write("    {" + ", ".join(map(str, sample.tolist())) + "}" + suffix + "\n")
            f.write("};\n\n")

            f.write(
                f"static const int8_t {array_prefix}_pmut[{sample_count}][{TARGET_LEN}] = {{\n"
            )
            for sample_idx, sample in enumerate(pmut_data):
                suffix = "," if sample_idx < sample_count - 1 else ""
                f.write("    {" + ", ".join(map(str, sample.tolist())) + "}" + suffix + "\n")
            f.write("};\n\n")
            f.write("#endif\n")

        header_paths.append(
            {
                "class_idx": class_idx,
                "class_name": class_name,
                "header_path": header_path,
                "sample_count": sample_count,
            }
        )

    manifest_path = os.path.join(TEST_HEADER_DIR, "header_manifest.csv")
    pd.DataFrame(header_paths).to_csv(manifest_path, index=False)
    return manifest_path


def save_embedded_test_header(x_test, y_test_idx, num_verify=5):
    num_verify = min(num_verify, len(x_test))
    path = os.path.join(REPORT_DIR, "embedded_test_data.h")
    with open(path, "w", encoding="utf-8") as f:
        f.write("#ifndef __WHOLE_SENTENCE_EMBEDDED_TEST_DATA_H\n")
        f.write("#define __WHOLE_SENTENCE_EMBEDDED_TEST_DATA_H\n\n")
        f.write(f"#define TEST_SAMPLE_COUNT {num_verify}\n")
        f.write(f"#define TEST_SAMPLE_LEN {TARGET_LEN}\n")
        f.write("#define TEST_SAMPLE_CH 2\n\n")
        for i in range(num_verify):
            data = to_mcu_input_int8(x_test[i])
            flat = flatten_hwc_sample(data)
            f.write(f"// Sample Index: {i}, True Label: {int(y_test_idx[i])}\n")
            f.write(f"int8_t test_sample_{i}[{flat.size}] = {{")
            f.write(", ".join(map(str, flat.tolist())))
            f.write("};\n\n")
        f.write("#endif\n")
    return path


def export_mcu_inference_dataset(x_test, y_test_idx, test_paths, preds):
    rows = []
    class_groups = {
        class_name: {"inputs": [], "labels": [], "preds": [], "test_indices": []}
        for class_name in CLASS_NAMES
    }
    for i, (sample, true_idx, source_path, pred_idx) in enumerate(
        zip(x_test, y_test_idx, test_paths, preds)
    ):
        sample_int8 = to_mcu_input_int8(sample)
        flat_input = flatten_hwc_sample(sample_int8)
        category = CLASS_NAMES[int(true_idx)]
        file_stem = f"test_{i:03d}_{category}"
        csv_path = os.path.join(MCU_INFERENCE_DIR, file_stem + "_input_flat.csv")

        pd.DataFrame({"value": flat_input.astype(np.int16)}).to_csv(csv_path, index=False)
        class_groups[category]["inputs"].append(flat_input)
        class_groups[category]["labels"].append(int(true_idx))
        class_groups[category]["preds"].append(int(pred_idx))
        class_groups[category]["test_indices"].append(i)
        rows.append(
            {
                "test_index": i,
                "file_name": os.path.basename(csv_path),
                "true_idx": int(true_idx),
                "true_name": CLASS_NAMES[int(true_idx)],
                "pred_idx": int(pred_idx),
                "pred_name": CLASS_NAMES[int(pred_idx)],
                "input_len": int(flat_input.size),
                "source_clip_path": source_path,
                "csv_path": csv_path,
            }
        )

    manifest_path = os.path.join(MCU_INFERENCE_DIR, "nnom_input_manifest.csv")
    pd.DataFrame(rows).to_csv(manifest_path, index=False)

    header_path = os.path.join(MCU_INFERENCE_DIR, "whole_sentence_mcu_test_inputs.h")
    with open(header_path, "w", encoding="utf-8") as f:
        f.write("#ifndef __WHOLE_SENTENCE_MCU_TEST_INPUTS_H__\n")
        f.write("#define __WHOLE_SENTENCE_MCU_TEST_INPUTS_H__\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"#define WHOLE_SENTENCE_TEST_SAMPLE_COUNT {len(rows)}\n")
        f.write(f"#define WHOLE_SENTENCE_TEST_SAMPLE_LEN {TARGET_LEN}\n")
        f.write("#define WHOLE_SENTENCE_TEST_SAMPLE_CH 2\n")
        f.write(f"#define WHOLE_SENTENCE_TEST_INPUT_SIZE {TARGET_LEN * 2}\n\n")

        for class_idx, class_name in enumerate(CLASS_NAMES):
            group = class_groups[class_name]
            array_prefix = class_name.lower()
            sample_count = len(group["inputs"])
            f.write(f"// Class {class_name} (class_idx={class_idx})\n")
            f.write(
                f"// Sentence: {SENTENCE_CLASS_CONFIG[CLASS_IDS[class_idx]]['text']}\n"
            )
            f.write(f"#define {array_prefix.upper()}_TEST_SAMPLE_COUNT {sample_count}\n")
            f.write(f"#define {array_prefix.upper()}_TEST_CLASS_IDX {class_idx}\n\n")

            f.write(
                f"static const int8_t {array_prefix}_test_inputs"
                f"[{sample_count}][{TARGET_LEN * 2}] = {{\n"
            )
            for sample_idx, flat_input in enumerate(group["inputs"]):
                suffix = "," if sample_idx < sample_count - 1 else ""
                f.write("    {" + ", ".join(map(str, flat_input.tolist())) + "}" + suffix + "\n")
            f.write("};\n\n")

            f.write(f"static const uint16_t {array_prefix}_test_indices[{sample_count}] = {{")
            f.write(", ".join(str(idx) for idx in group["test_indices"]))
            f.write("};\n\n")

            f.write(f"static const uint8_t {array_prefix}_test_preds[{sample_count}] = {{")
            f.write(", ".join(str(pred) for pred in group["preds"]))
            f.write("};\n\n")

        f.write("#endif\n")
    return manifest_path, header_path


def export_quant_metadata(quant_params):
    with open(os.path.join(OUTPUT_ROOT, "quantization_params.json"), "w", encoding="utf-8") as f:
        json.dump(quant_params, f, indent=2)


def export_model_to_nnom(model, x_test):
    weights_path = os.path.join(MODEL_DIR, "whole_sentence_weights.h")
    calibration_samples = x_test[: min(len(x_test), 100)]
    if len(calibration_samples) == 0:
        print("Skip NNoM export: no calibration samples available.")
        return None
    generate_model(model, calibration_samples, format="hwc", name=weights_path)
    return weights_path


def main():
    ensure_dirs()

    sentence_df = read_sentence_annotations(CSV_PATH)
    if sentence_df.empty:
        print("No valid sentence-level annotations found.")
        return

    signal_map = {}
    for fid in CLASS_IDS:
        sig_t, sig_p = load_dual_signals(fid)
        if sig_t is None or sig_p is None:
            print(f"Missing or invalid signal for ID {fid}, stop.")
            return
        signal_map[fid] = (sig_t, sig_p)

    quant_params = compute_channel_quant_params(signal_map)
    export_quant_metadata(quant_params)
    save_quantized_full_signals(signal_map, quant_params)

    detection_refs = build_detection_references(sentence_df, signal_map)
    missing_refs = [fid for fid in CLASS_IDS if fid not in detection_refs]
    if missing_refs:
        print(f"Missing valid sentence templates for IDs: {', '.join(missing_refs)}")
        return
    save_detection_reference_report(detection_refs)

    clip_df = save_quantized_sentence_clips(signal_map, detection_refs, quant_params)
    if clip_df.empty:
        print("No valid quantized sentence clips were generated.")
        return

    x_data, y_idx, y_data, clip_paths = prepare_dataset(clip_df)
    train_idx, val_idx, test_idx = split_indices_by_class(y_idx, seed=RANDOM_SEED)
    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        print("Dataset split failed. Need more sentence clips per class.")
        return
    save_split_manifest(clip_df, train_idx, val_idx, test_idx)

    x_train = x_data[train_idx]
    y_train = y_data[train_idx]
    x_val = x_data[val_idx]
    y_val = y_data[val_idx]
    x_test = x_data[test_idx]
    y_test = y_data[test_idx]
    y_test_idx = y_idx[test_idx]
    test_paths = [clip_paths[i] for i in test_idx]

    print(f"Dataset ready: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    print_split_summary(y_idx, train_idx, val_idx, test_idx)

    physical_devices = tf.config.list_physical_devices("GPU")
    print("Num GPUs Available:", len(physical_devices))
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    model = build_sentence_model()
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model_png = os.path.join(MODEL_DIR, "whole_sentence_model.png")
    plot_model(model, to_file=model_png, show_shapes=True)

    best_model_path = os.path.join(MODEL_DIR, "whole_sentence_edge_best.keras")
    callback_list = [
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=20, verbose=1),
        callbacks.ModelCheckpoint(
            best_model_path, monitor="val_accuracy", save_best_only=True, mode="max", verbose=1
        ),
    ]

    print("Training ----------------------")
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=callback_list,
        verbose=1,
    )

    plot_history(history, os.path.join(REPORT_DIR, "loss_accuracy_curve.png"))

    best_model = tf.keras.models.load_model(best_model_path)
    test_loss, test_accuracy = best_model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss: {test_loss:.6f}")
    print(f"Test accuracy: {test_accuracy:.6f}")

    accuracy_pct, preds, _ = save_prediction_reports(best_model, x_test, y_test_idx, test_paths)
    test_manifest_path = export_test_dataset_folder(x_test, y_test_idx, test_paths)
    header_manifest_path = export_mcu_test_headers_by_class(x_test, y_test_idx)
    header_path = save_embedded_test_header(x_test, y_test_idx, num_verify=5)
    mcu_manifest_path, mcu_header_path = export_mcu_inference_dataset(
        x_test, y_test_idx, test_paths, preds
    )
    weights_path = export_model_to_nnom(best_model, x_test)

    print(f"Quantized full signals saved to: {QUANTIZED_SIGNAL_DIR}")
    print(f"Quantized sentence clips saved to: {QUANTIZED_CLIP_DIR}")
    print(f"MCU test dataset saved to: {TEST_DATASET_DIR}")
    print(f"MCU test manifest saved to: {test_manifest_path}")
    print(f"MCU test headers saved to: {TEST_HEADER_DIR}")
    print(f"MCU header manifest saved to: {header_manifest_path}")
    print(f"Embedded test header saved to: {header_path}")
    print(f"MCU NNoM input manifest saved to: {mcu_manifest_path}")
    print(f"MCU NNoM input header saved to: {mcu_header_path}")
    if weights_path is not None:
        print(f"NNoM weights saved to: {weights_path}")
    print(f"Final test accuracy: {accuracy_pct:.2f}%")


if __name__ == "__main__":
    main()
