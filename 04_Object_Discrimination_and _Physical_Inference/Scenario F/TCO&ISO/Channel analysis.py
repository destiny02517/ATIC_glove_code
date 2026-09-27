import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from sklearn.manifold import TSNE
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Model, load_model


# ================== Config ==================
FUSION_MODEL_PATH = "CNN/FusionSense.keras"
BE_MODEL_PATH = "CNN/BE_MODEL.keras"  # Optional. If missing, script uses BE-proxy automatically.

BE_DATA_PATH = "locat"
UE_DATA_PATH = "sense"
OUTPUT_DIR = "DualModel_Compare_Results"

TRAIN_SAMPLE_NUMBER = 70
TEST_SAMPLE_NUMBER = 30
TSNE_PERPLEXITY = 30
TSNE_RANDOM_STATE = 42


def list_numeric_csv_ids(folder):
    ids = []
    for name in os.listdir(folder):
        if not name.lower().endswith(".csv"):
            continue
        stem = os.path.splitext(name)[0]
        if stem.isdigit():
            ids.append(int(stem))
    return sorted(ids)


def try_resolve_be_model_path():
    if os.path.exists(BE_MODEL_PATH):
        return BE_MODEL_PATH

    cnn_dir = os.path.dirname(BE_MODEL_PATH) or "CNN"
    if not os.path.isdir(cnn_dir):
        return None

    all_keras = [f for f in os.listdir(cnn_dir) if f.lower().endswith(".keras")]
    fusion_name = os.path.basename(FUSION_MODEL_PATH).lower()
    candidates = [f for f in all_keras if f.lower() != fusion_name and "fusion" not in f.lower()]

    if len(candidates) == 1:
        picked = os.path.join(cnn_dir, candidates[0])
        print(f"[Info] BE model auto-selected: {picked}")
        return picked

    return None


def infer_input_len(model, input_index=0):
    shape = model.input_shape
    if isinstance(shape, list):
        return int(shape[input_index][1])
    return int(shape[1])


def infer_output_classes(model):
    shape = model.output_shape
    if isinstance(shape, list):
        shape = shape[0]
    return int(shape[-1])


def load_split(folder, class_ids, sample_length, start_col, n_cols):
    chunks = []
    for cid in class_ids:
        p = os.path.join(folder, f"{cid}.csv")
        arr = pd.read_csv(p, header=None).values
        if arr.shape[0] < sample_length:
            raise ValueError(f"{p} has {arr.shape[0]} rows, need at least {sample_length}.")
        if arr.shape[1] < start_col + n_cols:
            raise ValueError(f"{p} has {arr.shape[1]} cols, need at least {start_col + n_cols}.")
        block = arr[:sample_length, start_col:start_col + n_cols]
        chunks.append(block)

    merged = np.concatenate(chunks, axis=1)
    return merged.T.reshape(-1, sample_length, 1)


def zeros_like_second_input(model, batch_size):
    if not isinstance(model.input_shape, list) or len(model.input_shape) < 2:
        return None
    second_shape = model.input_shape[1]
    length = int(second_shape[1])
    channels = int(second_shape[2]) if len(second_shape) > 2 and second_shape[2] is not None else 1
    return np.zeros((batch_size, length, channels), dtype=np.float32)


def predict_labels(model, be_data, ue_data=None, use_fusion_inputs=False):
    if isinstance(model.input_shape, list):
        if len(model.input_shape) == 1:
            probs = model.predict(be_data, verbose=0)
        elif len(model.input_shape) == 2:
            if use_fusion_inputs:
                if ue_data is None:
                    raise ValueError("Fusion-style prediction requires UE input.")
                probs = model.predict([be_data, ue_data], verbose=0)
            else:
                second = zeros_like_second_input(model, be_data.shape[0])
                probs = model.predict([be_data, second], verbose=0)
        else:
            raise ValueError("Unsupported model: more than 2 inputs.")
    else:
        probs = model.predict(be_data, verbose=0)
    return np.argmax(probs, axis=1)


def choose_feature_layer(model, preferred_names):
    for name in preferred_names:
        try:
            return model.get_layer(name)
        except ValueError:
            continue
    if len(model.layers) >= 2:
        return model.layers[-2]
    return model.layers[-1]


def extract_features(model, be_data, ue_data=None, use_fusion_inputs=False, preferred_names=None):
    preferred_names = preferred_names or []
    layer = choose_feature_layer(model, preferred_names)
    extractor = Model(inputs=model.input, outputs=layer.output)

    if isinstance(model.input_shape, list):
        if len(model.input_shape) == 1:
            feat = extractor.predict(be_data, verbose=0)
        elif len(model.input_shape) == 2:
            if use_fusion_inputs:
                if ue_data is None:
                    raise ValueError("Fusion-style feature extraction requires UE input.")
                feat = extractor.predict([be_data, ue_data], verbose=0)
            else:
                second = zeros_like_second_input(model, be_data.shape[0])
                feat = extractor.predict([be_data, second], verbose=0)
        else:
            raise ValueError("Unsupported model: more than 2 inputs.")
    else:
        feat = extractor.predict(be_data, verbose=0)

    return feat.reshape(feat.shape[0], -1)


def compute_tsne(features):
    tsne = TSNE(
        n_components=2,
        perplexity=TSNE_PERPLEXITY,
        random_state=TSNE_RANDOM_STATE,
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(features)


def fit_proxy_classifier(train_features, train_labels):
    clf = make_pipeline(
        StandardScaler(with_mean=True),
        LogisticRegression(
            max_iter=5000,
            multi_class="multinomial",
            solver="lbfgs",
        ),
    )
    clf.fit(train_features, train_labels)
    return clf


def extract_fusion_branch_features(fusion_model, be_data, ue_data, branch):
    if branch == "be":
        preferred = ["activation_5", "activation_4", "dense"]
        in_be = be_data
        in_ue = np.zeros_like(ue_data)
    elif branch == "ue":
        preferred = ["activation_11", "activation_10", "dense_1"]
        in_be = np.zeros_like(be_data)
        in_ue = ue_data
    else:
        raise ValueError("branch must be 'be' or 'ue'.")

    layer = choose_feature_layer(fusion_model, preferred)
    extractor = Model(inputs=fusion_model.input, outputs=layer.output)
    feat = extractor.predict([in_be, in_ue], verbose=0)
    return feat.reshape(feat.shape[0], -1)


def draw_compact_ellipse(ax, x, y, color, cover_quantile=0.60, scale=0.85, alpha=0.14):
    if x.size < 3:
        return

    pos = np.column_stack((x, y))
    center = np.mean(pos, axis=0)
    cov = np.cov(x, y) + np.eye(2) * 1e-12

    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = np.maximum(vals[order], 1e-12)
    vecs = vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    aligned = (pos - center) @ vecs
    rad2 = np.sum((aligned ** 2) / vals, axis=1)
    r = np.sqrt(np.quantile(rad2, cover_quantile)) * scale

    width, height = 2 * r * np.sqrt(vals)
    ell = Ellipse(
        xy=center,
        width=width,
        height=height,
        angle=theta,
        facecolor=color,
        edgecolor="white",
        linestyle="--",
        linewidth=1.0,
        alpha=alpha,
        zorder=1,
    )
    ax.add_patch(ell)


def plot_tsne_correct_error(embedding, y_true, y_pred, class_ids, title, save_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.cm.get_cmap("tab20", max(20, len(class_ids)))

    for c in np.unique(y_true):
        m = (y_true == c)
        pts = embedding[m]
        color = cmap(int(c) % 20)
        draw_compact_ellipse(ax, pts[:, 0], pts[:, 1], color=color)
        ax.scatter(pts[:, 0], pts[:, 1], c=[color], s=16, alpha=0.50, edgecolors="none", zorder=2)

    wrong = (y_true != y_pred)
    if np.any(wrong):
        ax.scatter(
            embedding[wrong, 0], embedding[wrong, 1],
            c="black", marker="x", s=34, linewidths=1.0, alpha=0.90, zorder=4
        )

    acc = np.mean(y_true == y_pred) * 100.0
    wrong_n = int(np.sum(wrong))
    total_n = int(len(y_true))
    ax.set_title(f"{title}\nAccuracy={acc:.2f}%  Errors={wrong_n}/{total_n}", fontsize=15, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=7, label="Samples"),
        Line2D([0], [0], marker="x", color="black", linestyle="None", markersize=8, label="Misclassified"),
    ]
    ax.legend(handles=handles, loc="best", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_tsne_clusters(embedding, y_true, class_ids, title, save_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.cm.get_cmap("tab20", max(20, len(class_ids)))

    for c in np.unique(y_true):
        m = (y_true == c)
        pts = embedding[m]
        color = cmap(int(c) % 20)
        draw_compact_ellipse(ax, pts[:, 0], pts[:, 1], color=color)
        ax.scatter(pts[:, 0], pts[:, 1], c=[color], s=18, alpha=0.62, edgecolors="none", zorder=2)

    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def per_class_accuracy(y_true, y_pred, n_class):
    out = np.full(n_class, np.nan, dtype=float)
    for c in range(n_class):
        m = (y_true == c)
        if np.any(m):
            out[c] = np.mean(y_pred[m] == c)
    return out


def plot_per_class_bars(be_acc_pct, fusion_acc_pct, class_ids, be_label, save_path):
    x = np.arange(len(class_ids))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(12, len(class_ids) * 0.5), 5.5))
    ax.bar(x - width / 2, be_acc_pct, width=width, label=be_label, color="#4C78A8", alpha=0.90)
    ax.bar(x + width / 2, fusion_acc_pct, width=width, label="Fusion model", color="#F58518", alpha=0.90)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Per-class Accuracy (%)")
    ax.set_xlabel("Class ID")
    ax.set_title("Per-class Accuracy Comparison", fontsize=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in class_ids], fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_gain_bar(gain_pct, class_ids, summary_text, save_path):
    x = np.arange(len(class_ids))
    colors = np.where(gain_pct >= 0, "#4CAF50", "#E74C3C")
    fig, ax = plt.subplots(figsize=(max(12, len(class_ids) * 0.5), 5.5))
    ax.bar(x, gain_pct, color=colors, alpha=0.90, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_ylabel("Gain (%)")
    ax.set_xlabel("Class ID")
    ax.set_title("Per-class Accuracy Gain: Fusion - BE (%)", fontsize=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in class_ids], fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.text(
        0.01, 0.98, summary_text,
        transform=ax.transAxes, ha="left", va="top", fontsize=10,
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none")
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_metrics_csv(csv_path, class_ids, be_acc_pct, fusion_acc_pct, gain_pct, counts):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "samples", "be_acc_pct", "fusion_acc_pct", "gain_pct"])
        for cid, n, a, b, g in zip(class_ids, counts, be_acc_pct, fusion_acc_pct, gain_pct):
            w.writerow([cid, int(n), float(a), float(b), float(g)])


def save_metrics_csv_named(csv_path, class_ids, base_label, base_acc_pct, fusion_acc_pct, gain_pct, counts):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "samples", f"{base_label}_acc_pct", "fusion_acc_pct", "gain_pct"])
        for cid, n, a, b, g in zip(class_ids, counts, base_acc_pct, fusion_acc_pct, gain_pct):
            w.writerow([cid, int(n), float(a), float(b), float(g)])


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Working directory:", os.getcwd())

    if not os.path.exists(FUSION_MODEL_PATH):
        raise FileNotFoundError(f"Fusion model not found: {FUSION_MODEL_PATH}")
    fusion_model = load_model(FUSION_MODEL_PATH)
    print("Fusion model:", FUSION_MODEL_PATH)

    be_model_path = try_resolve_be_model_path()
    be_model = None
    be_mode = "proxy"
    if be_model_path is not None:
        be_model = load_model(be_model_path)
        be_mode = "model"
        print("BE model:", be_model_path)
    else:
        print("[Info] BE model not found. Using BE-proxy: Fusion BE-branch features + LogisticRegression")

    be_ids = list_numeric_csv_ids(BE_DATA_PATH)
    ue_ids = list_numeric_csv_ids(UE_DATA_PATH)
    common_ids = sorted(set(be_ids).intersection(ue_ids))
    if not common_ids:
        raise ValueError("No matching numeric CSV ids between locat and sense.")

    n_fusion_cls = infer_output_classes(fusion_model)
    n_cls = min(len(common_ids), n_fusion_cls)
    if be_mode == "model":
        n_be_cls = infer_output_classes(be_model)
        n_cls = min(n_cls, n_be_cls)
    class_ids = common_ids[:n_cls]
    print(f"Using {n_cls} classes:", class_ids)

    be_len_for_fusion = infer_input_len(fusion_model, 0)
    ue_len_for_fusion = infer_input_len(fusion_model, 1) if isinstance(fusion_model.input_shape, list) and len(fusion_model.input_shape) > 1 else be_len_for_fusion

    be_x_train_fusion = load_split(BE_DATA_PATH, class_ids, be_len_for_fusion, 0, TRAIN_SAMPLE_NUMBER)
    ue_x_train_fusion = load_split(UE_DATA_PATH, class_ids, ue_len_for_fusion, 0, TRAIN_SAMPLE_NUMBER)
    be_x_test_fusion = load_split(BE_DATA_PATH, class_ids, be_len_for_fusion, TRAIN_SAMPLE_NUMBER, TEST_SAMPLE_NUMBER)
    ue_x_test_fusion = load_split(UE_DATA_PATH, class_ids, ue_len_for_fusion, TRAIN_SAMPLE_NUMBER, TEST_SAMPLE_NUMBER)

    y_train = np.repeat(np.arange(n_cls), TRAIN_SAMPLE_NUMBER)
    y_true = np.repeat(np.arange(n_cls), TEST_SAMPLE_NUMBER)
    class_counts = np.full(n_cls, TEST_SAMPLE_NUMBER, dtype=int)

    y_pred_fusion = predict_labels(
        fusion_model,
        be_x_test_fusion,
        ue_data=ue_x_test_fusion,
        use_fusion_inputs=True,
    )

    if be_mode == "model":
        be_len_for_be_model = infer_input_len(be_model, 0)
        be_x_test_be_model = load_split(BE_DATA_PATH, class_ids, be_len_for_be_model, TRAIN_SAMPLE_NUMBER, TEST_SAMPLE_NUMBER)
        y_pred_be = predict_labels(
            be_model,
            be_x_test_be_model,
            ue_data=None,
            use_fusion_inputs=False,
        )
        be_feat_test = extract_features(
            be_model,
            be_x_test_be_model,
            ue_data=None,
            use_fusion_inputs=False,
            preferred_names=["activation_5", "activation_4", "dense"],
        )
        be_label = "BE model"
        be_summary_name = "BE model"
    else:
        be_feat_train = extract_fusion_branch_features(
            fusion_model, be_x_train_fusion, ue_x_train_fusion, branch="be"
        )
        be_feat_test = extract_fusion_branch_features(
            fusion_model, be_x_test_fusion, ue_x_test_fusion, branch="be"
        )
        be_proxy_clf = fit_proxy_classifier(be_feat_train, y_train)
        y_pred_be = be_proxy_clf.predict(be_feat_test)
        be_label = "BE proxy"
        be_summary_name = "BE proxy (Fusion BE-branch + LR)"

    # UE-only proxy: Fusion UE-branch features + LR
    ue_feat_train = extract_fusion_branch_features(
        fusion_model, be_x_train_fusion, ue_x_train_fusion, branch="ue"
    )
    ue_feat_test = extract_fusion_branch_features(
        fusion_model, be_x_test_fusion, ue_x_test_fusion, branch="ue"
    )
    ue_proxy_clf = fit_proxy_classifier(ue_feat_train, y_train)
    y_pred_ue = ue_proxy_clf.predict(ue_feat_test)
    ue_label = "UE proxy"

    be_acc = np.mean(y_pred_be == y_true) * 100.0
    ue_acc = np.mean(y_pred_ue == y_true) * 100.0
    fusion_acc = np.mean(y_pred_fusion == y_true) * 100.0
    print(f"{be_label} overall acc : {be_acc:.2f}%")
    print(f"{ue_label} overall acc : {ue_acc:.2f}%")
    print(f"Fusion overall acc   : {fusion_acc:.2f}%")

    be_pc = per_class_accuracy(y_true, y_pred_be, n_cls) * 100.0
    ue_pc = per_class_accuracy(y_true, y_pred_ue, n_cls) * 100.0
    fusion_pc = per_class_accuracy(y_true, y_pred_fusion, n_cls) * 100.0
    gain_be_pc = fusion_pc - be_pc
    gain_ue_pc = fusion_pc - ue_pc
    mean_gain_be = float(np.nanmean(gain_be_pc))
    mean_gain_ue = float(np.nanmean(gain_ue_pc))

    be_embed = compute_tsne(be_feat_test)
    plot_tsne_correct_error(
        be_embed, y_true, y_pred_be, class_ids,
        f"{be_label} t-SNE with Error Marks",
        os.path.join(OUTPUT_DIR, "Fig1_BE_tsne_error.png"),
    )
    plot_tsne_clusters(
        be_embed, y_true, class_ids,
        f"{be_label} t-SNE Clusters",
        os.path.join(OUTPUT_DIR, "Fig1b_BE_tsne_clusters.png"),
    )

    ue_embed = compute_tsne(ue_feat_test)
    plot_tsne_correct_error(
        ue_embed, y_true, y_pred_ue, class_ids,
        f"{ue_label} t-SNE with Error Marks",
        os.path.join(OUTPUT_DIR, "Fig2_UE_tsne_error.png"),
    )
    plot_tsne_clusters(
        ue_embed, y_true, class_ids,
        f"{ue_label} t-SNE Clusters",
        os.path.join(OUTPUT_DIR, "Fig2b_UE_tsne_clusters.png"),
    )

    fusion_feat = extract_features(
        fusion_model,
        be_x_test_fusion,
        ue_data=ue_x_test_fusion,
        use_fusion_inputs=True,
        preferred_names=["activation_13", "activation_12", "dense_2"],
    )
    fusion_embed = compute_tsne(fusion_feat)
    plot_tsne_correct_error(
        fusion_embed, y_true, y_pred_fusion, class_ids,
        "Fusion model t-SNE with Error Marks",
        os.path.join(OUTPUT_DIR, "Fig3_Fusion_tsne_error.png"),
    )
    plot_tsne_clusters(
        fusion_embed, y_true, class_ids,
        "Fusion model t-SNE Clusters",
        os.path.join(OUTPUT_DIR, "Fig3b_Fusion_tsne_clusters.png"),
    )

    plot_per_class_bars(
        be_pc, fusion_pc, class_ids, be_label,
        os.path.join(OUTPUT_DIR, "Fig4_PerClassAcc_BE_vs_Fusion.png"),
    )
    plot_per_class_bars(
        ue_pc, fusion_pc, class_ids, ue_label,
        os.path.join(OUTPUT_DIR, "Fig5_PerClassAcc_UE_vs_Fusion.png"),
    )

    summary_be = (
        f"Macro {be_label}={np.nanmean(be_pc):.2f}% | "
        f"Macro Fusion={np.nanmean(fusion_pc):.2f}% | "
        f"Mean Gain={mean_gain_be:.2f}%"
    )
    plot_gain_bar(
        gain_be_pc, class_ids, summary_be,
        os.path.join(OUTPUT_DIR, "Fig6_PerClassGain_FusionMinusBE.png"),
    )

    summary_ue = (
        f"Macro {ue_label}={np.nanmean(ue_pc):.2f}% | "
        f"Macro Fusion={np.nanmean(fusion_pc):.2f}% | "
        f"Mean Gain={mean_gain_ue:.2f}%"
    )
    plot_gain_bar(
        gain_ue_pc, class_ids, summary_ue,
        os.path.join(OUTPUT_DIR, "Fig7_PerClassGain_FusionMinusUE.png"),
    )

    save_metrics_csv(
        os.path.join(OUTPUT_DIR, "per_class_metrics_be_vs_fusion.csv"),
        class_ids, be_pc, fusion_pc, gain_be_pc, class_counts
    )
    save_metrics_csv_named(
        os.path.join(OUTPUT_DIR, "per_class_metrics_ue_vs_fusion.csv"),
        class_ids, "ue", ue_pc, fusion_pc, gain_ue_pc, class_counts
    )

    with open(os.path.join(OUTPUT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"Fusion model path: {FUSION_MODEL_PATH}\n")
        if be_mode == "model":
            f.write(f"BE model path: {be_model_path}\n")
        else:
            f.write("BE model path: <none>\n")
            f.write("BE mode: proxy (Fusion BE-branch features + LogisticRegression)\n")
        f.write("UE mode: proxy (Fusion UE-branch features + LogisticRegression)\n")
        f.write(f"Classes used: {class_ids}\n")
        f.write(f"Overall {be_label} accuracy: {be_acc:.4f}%\n")
        f.write(f"Overall {ue_label} accuracy: {ue_acc:.4f}%\n")
        f.write(f"Overall Fusion accuracy: {fusion_acc:.4f}%\n")
        f.write(f"Macro {be_label} accuracy: {np.nanmean(be_pc):.4f}%\n")
        f.write(f"Macro {ue_label} accuracy: {np.nanmean(ue_pc):.4f}%\n")
        f.write(f"Macro Fusion accuracy: {np.nanmean(fusion_pc):.4f}%\n")
        f.write(f"Mean per-class gain (Fusion-BE): {mean_gain_be:.4f}%\n")
        f.write(f"Mean per-class gain (Fusion-UE): {mean_gain_ue:.4f}%\n")
        f.write(f"BE summary name: {be_summary_name}\n")
        f.write("UE summary name: UE proxy (Fusion UE-branch + LR)\n")

    print(f"Saved all results to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
