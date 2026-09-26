from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import griddata
from scipy.optimize import nnls
from scipy.stats import f_oneway, kruskal, mannwhitneyu, pearsonr


CLASS_NODES = {
    1: [1],
    2: [2],
    3: [3],
    4: [4],
    5: [5],
    6: [6],
    7: [7],
    8: [8],
    9: [9],
    10: [10],
    11: [11],
    12: [12],
    13: [13],
    14: [14],
    15: [15],
    16: [16],
    17: [17],
    18: [18],
    19: [5, 6],
    20: [13, 14, 15, 16],
    21: [7, 9],
    22: [17, 12],
    23: [14, 15],
    24: [13, 16],
    25: [3, 5, 7],
    26: [7, 8],
    27: [1, 2],
    28: [3, 5, 7, 9],
    29: [14, 15, 18],
    30: [11, 12, 13, 14, 15, 16, 17, 18],
}

NODE_LOC = {
    1: "Thumb",
    2: "Thumb",
    3: "Index",
    4: "Index",
    5: "Middle",
    6: "Middle",
    7: "Ring",
    8: "Ring",
    9: "Pinky",
    10: "Pinky",
    11: "Palm",
    12: "Palm",
    13: "Palm",
    14: "Palm",
    15: "Palm",
    16: "Palm",
    17: "Palm",
    18: "Palm",
}

NODE_COORDS = {
    1: (-4.0, 4.6),
    2: (-3.3, 3.4),
    3: (-1.9, 6.2),
    4: (-1.9, 4.5),
    5: (0.0, 6.5),
    6: (0.0, 4.7),
    7: (1.9, 6.1),
    8: (1.9, 4.4),
    9: (3.7, 5.2),
    10: (3.7, 3.7),
    11: (-2.9, 1.0),
    12: (-1.3, 1.4),
    13: (-0.5, 2.5),
    14: (0.5, 2.8),
    15: (1.4, 2.8),
    16: (2.4, 2.3),
    17: (-0.2, 0.9),
    18: (1.6, 0.9),
}

VB_REF_COORD = (0.35, 0.15)

BASE_DIR = Path(__file__).resolve().parent
TOTAL_WAVE_DIR = BASE_DIR / "Total wave"
OUTPUT_DIR = BASE_DIR / "Relationship_Results"
MODEL_PATH = BASE_DIR / "FusionSense.keras"
NUM_CLASSES = 30

NC_FONT = "Arial"
NC_TITLE_SIZE = 18
NC_LABEL_SIZE = 16
NC_TICK_SIZE = 14
NC_ANNOTATION_SIZE = 14


def read_average_wave(class_id, channel):
    file_path = TOTAL_WAVE_DIR / f"Class{class_id}" / f"Average_V{channel}_{class_id}.csv"
    data = pd.read_csv(file_path)
    if data.shape[1] < 2:
        raise ValueError(f"{file_path} should contain time and waveform columns.")
    values = pd.to_numeric(data.iloc[:, 1], errors="coerce").dropna()
    return values.to_numpy(dtype=float)


def read_cycles(class_id, channel):
    file_path = TOTAL_WAVE_DIR / f"Class{class_id}" / f"Cycles_V{channel}_{class_id}.csv"
    data = pd.read_csv(file_path)
    numeric = data.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(axis=0, how="all").dropna(axis=1, how="all")
    return numeric.to_numpy(dtype=float)


def load_class_data(class_id):
    avg_v1 = read_average_wave(class_id, 1)
    avg_v2 = read_average_wave(class_id, 2)
    cyc_v1 = read_cycles(class_id, 1)
    cyc_v2 = read_cycles(class_id, 2)

    mean_signal = np.concatenate([avg_v1, avg_v2])
    cycle_energy = np.concatenate(
        [
            np.sum(cyc_v1**2, axis=0),
            np.sum(cyc_v2**2, axis=0),
        ]
    )
    cycle_std = np.concatenate(
        [
            np.std(cyc_v1, axis=0),
            np.std(cyc_v2, axis=0),
        ]
    )

    return {
        "class_id": class_id,
        "nodes": CLASS_NODES[class_id],
        "avg_v1": avg_v1,
        "avg_v2": avg_v2,
        "signal": mean_signal,
        "signal_norm": zscore(mean_signal),
        "energy_mean": float(np.mean(cycle_energy)),
        "energy_std": float(np.mean(cycle_std)),
    }


def zscore(arr):
    arr = np.asarray(arr, dtype=float)
    std = arr.std()
    if std == 0:
        return np.zeros_like(arr)
    return (arr - arr.mean()) / std


def safe_corr(a, b):
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return 0.0
    corr, _ = pearsonr(a, b)
    if np.isnan(corr):
        return 0.0
    return float(corr)


def jaccard_similarity(left, right):
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    if not union:
        return 0.0
    return len(left_set & right_set) / len(union)


def location_similarity(nodes_a, nodes_b):
    loc_a = {NODE_LOC[n] for n in nodes_a}
    loc_b = {NODE_LOC[n] for n in nodes_b}
    return jaccard_similarity(loc_a, loc_b)


def energy_similarity(energy_a, energy_b):
    denom = max(abs(energy_a), abs(energy_b), 1e-9)
    return max(0.0, 1.0 - abs(energy_a - energy_b) / denom)


def build_similarity_table(class_data):
    rows = []
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))

    for i in range(1, NUM_CLASSES + 1):
        data_i = class_data[i]
        for j in range(1, NUM_CLASSES + 1):
            data_j = class_data[j]
            wave_corr = safe_corr(data_i["signal_norm"], data_j["signal_norm"])
            wave_score = (wave_corr + 1.0) / 2.0
            node_score = jaccard_similarity(data_i["nodes"], data_j["nodes"])
            loc_score = location_similarity(data_i["nodes"], data_j["nodes"])
            energy_score = energy_similarity(data_i["energy_mean"], data_j["energy_mean"])

            combined_score = (
                0.50 * wave_score
                + 0.20 * node_score
                + 0.10 * loc_score
                + 0.20 * energy_score
            )
            matrix[i - 1, j - 1] = combined_score

            if i < j:
                rows.append(
                    {
                        "Class_A": i,
                        "Class_B": j,
                        "WaveCorr": wave_corr,
                        "NodeJaccard": node_score,
                        "LocationJaccard": loc_score,
                        "EnergySimilarity": energy_score,
                        "CombinedSimilarity": combined_score,
                    }
                )

    return pd.DataFrame(rows), matrix


def additive_fit_report(class_data):
    base_signals = {
        class_id: class_data[class_id]["signal_norm"]
        for class_id in range(1, 19)
    }
    reports = []

    for class_id in range(1, NUM_CLASSES + 1):
        nodes = CLASS_NODES[class_id]
        if len(nodes) <= 1:
            continue

        components = [node for node in nodes if node in base_signals]
        if not components:
            continue

        design_matrix = np.column_stack([base_signals[node] for node in components])
        target = class_data[class_id]["signal_norm"]
        coeffs, _ = nnls(design_matrix, target)
        reconstruction = design_matrix @ coeffs
        residual = target - reconstruction
        denom = np.sum((target - target.mean()) ** 2)
        additive_r2 = 0.0 if denom == 0 else max(0.0, 1.0 - np.sum(residual**2) / denom)

        coeff_sum = coeffs.sum()
        normalized_coeffs = coeffs / coeff_sum if coeff_sum > 0 else coeffs
        dominant = components[int(np.argmax(coeffs))] if coeffs.size else None

        reports.append(
            {
                "Class": class_id,
                "Components": ",".join(str(v) for v in components),
                "AdditiveR2": additive_r2,
                "DominantComponent": dominant,
                "Weights": ",".join(f"{v:.4f}" for v in normalized_coeffs),
            }
        )

    return pd.DataFrame(reports).sort_values(by="AdditiveR2", ascending=False)


def class_position(class_id):
    points = np.array([NODE_COORDS[node] for node in CLASS_NODES[class_id]], dtype=float)
    center = points.mean(axis=0)
    if len(CLASS_NODES[class_id]) > 1:
        direction = center / (np.linalg.norm(center) + 1e-9)
        center = center + direction * 0.35
    return center


def top_similarity_edges(similarity_df, limit=45):
    filtered = similarity_df[similarity_df["CombinedSimilarity"] >= 0.63].copy()
    if filtered.empty:
        filtered = similarity_df.nlargest(limit, "CombinedSimilarity").copy()
    else:
        filtered = filtered.nlargest(limit, "CombinedSimilarity")
    return filtered


def plot_similarity_heatmap(matrix):
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(
        matrix,
        cmap="mako",
        vmin=0,
        vmax=1,
        square=True,
        xticklabels=range(1, NUM_CLASSES + 1),
        yticklabels=range(1, NUM_CLASSES + 1),
        cbar_kws={
            "shrink": 1,  # 🔴 控制长度（高度）
            "aspect": 20  # 🔴 控制宽度（越大越细）
        }
    )

    highlight_pairs = set()
    for class_id, nodes in CLASS_NODES.items():
        for node in nodes:
            if 1 <= node <= 18 and class_id != node:
                highlight_pairs.add((class_id, node))
                highlight_pairs.add((node, class_id))

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            class_i = i + 1
            class_j = j + 1
            should_annotate = (i == j) or ((class_i, class_j) in highlight_pairs)
            if not should_annotate:
                continue

            value = matrix[i, j]
            text_color = "white" if value < 0.55 else "#101010"
            ax.text(
                j + 0.5,
                i + 0.5,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=10,
                color=text_color,
            )

    plt.title(
        "30-Class Similarity Matrix",
        fontsize=18,
        fontweight='bold'
    )
    plt.xlabel("Class", fontsize=14, fontweight='bold')
    plt.ylabel("Class", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Class_Similarity_Heatmap.svg", format="svg")
    plt.close()


def plot_relationship_graph(similarity_df, additive_df):
    fig, ax = plt.subplots(figsize=(15, 12))
    positions = {class_id: class_position(class_id) for class_id in range(1, NUM_CLASSES + 1)}

    for _, row in top_similarity_edges(similarity_df).iterrows():
        a = int(row["Class_A"])
        b = int(row["Class_B"])
        x0, y0 = positions[a]
        x1, y1 = positions[b]
        ax.plot(
            [x0, x1],
            [y0, y1],
            color="#2a6f97",
            alpha=0.15 + 0.55 * row["CombinedSimilarity"],
            linewidth=0.6 + 2.8 * row["CombinedSimilarity"],
            zorder=1,
        )

    for _, row in additive_df.iterrows():
        if row["AdditiveR2"] < 0.65:
            continue
        target = int(row["Class"])
        components = [int(v) for v in row["Components"].split(",") if v]
        weights = [float(v) for v in row["Weights"].split(",") if v]
        for component, weight in zip(components, weights):
            x0, y0 = positions[target]
            x1, y1 = positions[component]
            ax.plot(
                [x0, x1],
                [y0, y1],
                color="#e07a5f",
                alpha=0.45,
                linewidth=1.0 + 4.0 * weight,
                linestyle="--",
                zorder=2,
            )

    for class_id in range(1, NUM_CLASSES + 1):
        x, y = positions[class_id]
        node_count = len(CLASS_NODES[class_id])
        if class_id <= 18:
            color = "#457b9d"
        else:
            color = "#d1495b"
        ax.scatter(
            x,
            y,
            s=180 + 75 * node_count,
            color=color,
            edgecolor="white",
            linewidth=1.2,
            zorder=3,
        )
        ax.text(x, y, str(class_id), ha="center", va="center", fontsize=9, color="white", zorder=4)

    for node_id, (x, y) in NODE_COORDS.items():
        ax.text(x, y - 0.35, f"N{node_id}", fontsize=7, ha="center", color="#4a4a4a")

    ax.set_title("30-Class Relationship Graph\nBlue: overall similarity, Orange dashed: additive links")
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Class_Relationship_Graph.svg", format="svg")
    plt.close()


def plot_additive_scores(additive_df):
    if additive_df.empty:
        return

    plt.figure(figsize=(7.2, 2.9))
    show_df = additive_df.sort_values(by="AdditiveR2", ascending=False).head(8).copy()
    ax = sns.barplot(
        data=show_df,
        x="AdditiveR2",
        y="Class",
        hue="Class",
        orient="h",
        palette="rocket",
        dodge=False,
        legend=False,
    )
    plt.xlim(0, 1)
    plt.title(
        "Top 8 Composite Additive Fits",
        fontsize=NC_TITLE_SIZE,
        fontname=NC_FONT,
        pad=4,
    )
    plt.xlabel("Non-negative additive fit (R²)", fontsize=NC_LABEL_SIZE, fontname=NC_FONT)
    plt.ylabel("Composite class", fontsize=NC_LABEL_SIZE, fontname=NC_FONT)
    ax.tick_params(axis="both", labelsize=NC_TICK_SIZE, length=2.5, width=0.6)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontname(NC_FONT)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.grid(False)

    for _, row in show_df.iterrows():
        class_id = int(row["Class"])
        components = [int(v) for v in row["Components"].split(",") if v]
        terms = [str(component) for component in components]
        label = format_additive_label(class_id, terms)
        y = list(show_df["Class"]).index(class_id)
        ax.text(
            0.985,
            y,
            label,
            va="center",
            ha="right",
            fontsize=NC_ANNOTATION_SIZE,
            fontname=NC_FONT,
            color="#303030",
        )

    plt.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.18)
    plt.savefig(OUTPUT_DIR / "Composite_Additive_Fit.svg", format="svg")
    plt.close()


def format_additive_label(class_id, terms, max_terms_per_line=4):
    if len(terms) <= max_terms_per_line:
        return f"{class_id}: [{', '.join(terms)}]"

    chunks = []
    for idx in range(0, len(terms), max_terms_per_line):
        chunks.append(", ".join(terms[idx : idx + max_terms_per_line]))

    lines = [f"{class_id}: [{chunks[0]},"]
    lines.extend(chunks[1:])
    lines[-1] = lines[-1] + "]"
    return "\n".join(lines)


def reference_point_vb():
    return VB_REF_COORD


def amplitude_distance_report(class_data):
    ref_x, ref_y = reference_point_vb()
    rows = []

    for class_id in range(1, 19):
        x, y = NODE_COORDS[class_id]
        avg_v1 = class_data[class_id]["avg_v1"]
        avg_v2 = class_data[class_id]["avg_v2"]
        amp_v1 = float(np.ptp(avg_v1))
        amp_v2 = float(np.ptp(avg_v2))
        amp_combined = float(np.ptp(np.concatenate([avg_v1, avg_v2])))
        rows.append(
            {
                "Class": class_id,
                "Node": class_id,
                "Region": NODE_LOC[class_id],
                "X": x,
                "Y": y,
                "RefX": ref_x,
                "RefY": ref_y,
                "DistanceToVb": float(np.hypot(x - ref_x, y - ref_y)),
                "Amplitude_V1": amp_v1,
                "Amplitude_V2": amp_v2,
                "Amplitude_Combined": amp_combined,
                "Peak_V1": float(np.max(avg_v1)),
                "Peak_V2": float(np.max(avg_v2)),
            }
        )

    df = pd.DataFrame(rows).sort_values(by="DistanceToVb")
    df.to_csv(
        OUTPUT_DIR / "Class1_18_Amplitude_Distance_to_Vb.csv",
        index=False,
        encoding="utf_8_sig",
    )
    return df


def plot_amplitude_distance(df):
    curve_df = df.copy().sort_values(by="DistanceToVb").reset_index(drop=True)
    curve_df["PositionIndex"] = np.arange(1, len(curve_df) + 1)
    curve_df.to_csv(
        OUTPUT_DIR / "Class1_18_Amplitude_Distance_to_Vb_Curve.csv",
        index=False,
        encoding="utf_8_sig",
    )

    plt.figure(figsize=(6.4, 3.0))
    ax = plt.gca()
    ax.plot(
        curve_df["PositionIndex"],
        curve_df["Amplitude_Combined"],
        color="#1f77b4",
        marker="o",
        markersize=4.5,
        linewidth=1.2,
    )

    for _, row in curve_df.iterrows():
        ax.text(
            row["PositionIndex"],
            row["Amplitude_Combined"] + 0.015,
            str(int(row["Class"])),
            ha="center",
            va="bottom",
            fontsize=6.5,
            fontname=NC_FONT,
            color="#303030",
        )

    corr = safe_corr(curve_df["DistanceToVb"].to_numpy(), curve_df["Amplitude_Combined"].to_numpy())
    ax.set_title(
        f"Distance Relationship to Vb (Classes 1-18, r = {corr:.3f})",
        fontsize=NC_TITLE_SIZE,
        fontname=NC_FONT,
        pad=4,
    )
    ax.set_xlabel("Distance rank from Vb", fontsize=NC_LABEL_SIZE, fontname=NC_FONT)
    ax.set_ylabel("Combined output amplitude", fontsize=NC_LABEL_SIZE, fontname=NC_FONT)
    ax.tick_params(axis="both", labelsize=NC_TICK_SIZE, length=2.5, width=0.6)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontname(NC_FONT)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    ax.set_xticks(curve_df["PositionIndex"])
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Class1_18_Amplitude_vs_Distance_to_Vb_Curve.svg", format="svg")
    plt.close()


def plot_spatial_amplitude_map(df):
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.2), sharex=True, sharey=True)
    specs = [
        ("Amplitude_V1", "Channel V1"),
        ("Amplitude_V2", "Channel V2"),
        ("Amplitude_Combined", "Combined"),
    ]

    x_vals = np.array([v[0] for v in NODE_COORDS.values()], dtype=float)
    y_vals = np.array([v[1] for v in NODE_COORDS.values()], dtype=float)
    x_pad = 0.8
    y_pad = 0.8
    grid_x, grid_y = np.mgrid[
        (x_vals.min() - 0.3) : (x_vals.max() + 0.3) : 180j,
        (y_vals.min() - 0.3) : (y_vals.max() + 0.3) : 180j,
    ]

    for ax, (col, title) in zip(axes, specs):
        z_grid = griddata(
            points=df[["X", "Y"]].to_numpy(),
            values=df[col].to_numpy(),
            xi=(grid_x, grid_y),
            method="cubic",
        )
        if np.isnan(z_grid).any():
            nearest_grid = griddata(
                points=df[["X", "Y"]].to_numpy(),
                values=df[col].to_numpy(),
                xi=(grid_x, grid_y),
                method="nearest",
            )
            z_grid = np.where(np.isnan(z_grid), nearest_grid, z_grid)

        contour = ax.contourf(
            grid_x,
            grid_y,
            z_grid,
            levels=12,
            cmap="viridis",
            alpha=0.95,
        )
        ax.contour(
            grid_x,
            grid_y,
            z_grid,
            levels=12,
            colors="white",
            linewidths=0.25,
            alpha=0.5,
        )
        ax.scatter(
            df["X"],
            df["Y"],
            c=df[col],
            cmap="viridis",
            s=150,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        for _, row in df.iterrows():
            ax.text(
                row["X"],
                row["Y"],
                str(int(row["Class"])),
                ha="center",
                va="center",
                fontsize=7,
                fontname=NC_FONT,
                color="white",
            )
        ax.scatter(
            [VB_REF_COORD[0]],
            [VB_REF_COORD[1]],
            marker="X",
            s=90,
            color="#d62828",
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )
        ax.text(
            VB_REF_COORD[0] + 0.12,
            VB_REF_COORD[1] - 0.12,
            "Vb",
            fontsize=7,
            fontname=NC_FONT,
            color="#d62828",
        )
        ax.set_title(title, fontsize=NC_TITLE_SIZE, fontname=NC_FONT, pad=4)
        ax.set_aspect("equal")
        ax.set_xlim(x_vals.min() - x_pad, x_vals.max() + x_pad)
        ax.set_ylim(y_vals.min() - y_pad, y_vals.max() + y_pad)
        ax.tick_params(axis="both", labelsize=NC_TICK_SIZE, length=2.5, width=0.6)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontname(NC_FONT)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.6)
        ax.spines["bottom"].set_linewidth(0.6)
        cbar = fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.03)
        cbar.ax.tick_params(labelsize=6, length=2)
        cbar.set_label("Amplitude", fontsize=7, fontname=NC_FONT)

    axes[0].set_xlabel("X", fontsize=NC_LABEL_SIZE, fontname=NC_FONT)
    axes[1].set_xlabel("X", fontsize=NC_LABEL_SIZE, fontname=NC_FONT)
    axes[2].set_xlabel("X", fontsize=NC_LABEL_SIZE, fontname=NC_FONT)
    axes[0].set_ylabel("Y", fontsize=NC_LABEL_SIZE, fontname=NC_FONT)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Class1_18_Spatial_Amplitude_Contour_Map.svg", format="svg")
    plt.close()


def regional_amplitude_stats(df):
    grouped = (
        df.groupby("Region", sort=False)
        .agg(
            Count=("Class", "count"),
            Mean_DistanceToVb=("DistanceToVb", "mean"),
            Mean_Amplitude_V1=("Amplitude_V1", "mean"),
            Mean_Amplitude_V2=("Amplitude_V2", "mean"),
            Mean_Amplitude_Combined=("Amplitude_Combined", "mean"),
            Std_Amplitude_Combined=("Amplitude_Combined", "std"),
            Max_Amplitude_Combined=("Amplitude_Combined", "max"),
            Min_Amplitude_Combined=("Amplitude_Combined", "min"),
        )
        .reset_index()
    )
    grouped["Std_Amplitude_Combined"] = grouped["Std_Amplitude_Combined"].fillna(0.0)
    grouped.to_csv(
        OUTPUT_DIR / "Class1_18_Regional_Amplitude_Stats.csv",
        index=False,
        encoding="utf_8_sig",
    )
    return grouped


def plot_regional_amplitude_stats(stats_df):
    order = ["Thumb", "Index", "Middle", "Ring", "Pinky", "Palm"]
    plot_df = stats_df.copy()
    plot_df["Region"] = pd.Categorical(plot_df["Region"], categories=order, ordered=True)
    plot_df = plot_df.sort_values("Region")

    plt.figure(figsize=(5.2, 3.0))
    ax = sns.barplot(
        data=plot_df,
        x="Region",
        y="Mean_Amplitude_Combined",
        hue="Region",
        palette=["#5f0f40", "#9a031e", "#fb8b24", "#0f4c5c", "#335c67", "#7a9e7e"],
        dodge=False,
        legend=False,
    )
    ax.errorbar(
        x=np.arange(len(plot_df)),
        y=plot_df["Mean_Amplitude_Combined"],
        yerr=plot_df["Std_Amplitude_Combined"],
        fmt="none",
        ecolor="#303030",
        elinewidth=0.8,
        capsize=2,
    )
    for idx, row in enumerate(plot_df.itertuples(index=False)):
        ax.text(
            idx,
            row.Mean_Amplitude_Combined + row.Std_Amplitude_Combined + 0.02,
            f"n={int(row.Count)}",
            ha="center",
            va="bottom",
            fontsize=6.5,
            fontname=NC_FONT,
            color="#303030",
        )

    ax.set_title("Regional Mean Combined Amplitude", fontsize=NC_TITLE_SIZE, fontname=NC_FONT, pad=4)
    ax.set_xlabel("Region", fontsize=NC_LABEL_SIZE, fontname=NC_FONT)
    ax.set_ylabel("Mean combined amplitude", fontsize=NC_LABEL_SIZE, fontname=NC_FONT)
    ax.tick_params(axis="both", labelsize=NC_TICK_SIZE, length=2.5, width=0.6)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontname(NC_FONT)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Class1_18_Regional_Amplitude_Stats.svg", format="svg")
    plt.close()


def regional_significance_tests(df):
    rows = []
    palm = df.loc[df["Region"] == "Palm", "Amplitude_Combined"].to_numpy()
    non_palm = df.loc[df["Region"] != "Palm", "Amplitude_Combined"].to_numpy()
    if len(palm) > 0 and len(non_palm) > 0:
        u_stat, u_p = mannwhitneyu(palm, non_palm, alternative="two-sided")
        rows.append(
            {
                "Test": "Palm vs non-Palm",
                "Metric": "Amplitude_Combined",
                "Statistic": float(u_stat),
                "PValue": float(u_p),
                "GroupSummary": f"Palm(n={len(palm)}) vs non-Palm(n={len(non_palm)})",
            }
        )

    groups = []
    labels = []
    for region, group in df.groupby("Region"):
        vals = group["Amplitude_Combined"].to_numpy()
        if len(vals) > 0:
            groups.append(vals)
            labels.append(region)

    if len(groups) >= 2:
        kw_stat, kw_p = kruskal(*groups)
        rows.append(
            {
                "Test": "Kruskal-Wallis across regions",
                "Metric": "Amplitude_Combined",
                "Statistic": float(kw_stat),
                "PValue": float(kw_p),
                "GroupSummary": ", ".join(f"{label}(n={len(vals)})" for label, vals in zip(labels, groups)),
            }
        )
        if all(len(vals) >= 2 for vals in groups):
            anova_stat, anova_p = f_oneway(*groups)
            rows.append(
                {
                    "Test": "One-way ANOVA across regions",
                    "Metric": "Amplitude_Combined",
                    "Statistic": float(anova_stat),
                    "PValue": float(anova_p),
                    "GroupSummary": ", ".join(f"{label}(n={len(vals)})" for label, vals in zip(labels, groups)),
                }
            )

    result = pd.DataFrame(rows)
    result.to_csv(
        OUTPUT_DIR / "Class1_18_Regional_Significance_Tests.csv",
        index=False,
        encoding="utf_8_sig",
    )
    return result


def build_model_channel_ratio_report():
    try:
        import os

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        from tensorflow import keras
    except Exception as exc:
        raise RuntimeError(f"TensorFlow/Keras is not available: {exc}") from exc

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    model = keras.models.load_model(MODEL_PATH, compile=False)
    rows = []

    for class_id in range(1, 19):
        cycles_v1 = read_cycles(class_id, 1)
        cycles_v2 = read_cycles(class_id, 2)
        sample_count = min(cycles_v1.shape[1], cycles_v2.shape[1])
        x1 = cycles_v1[:, :sample_count].T[..., np.newaxis].astype(np.float32)
        x2 = cycles_v2[:, :sample_count].T[..., np.newaxis].astype(np.float32)
        zeros_1 = np.zeros_like(x1)
        zeros_2 = np.zeros_like(x2)

        pred_both = model.predict([x1, x2], verbose=0)
        pred_v1 = model.predict([x1, zeros_2], verbose=0)
        pred_v2 = model.predict([zeros_1, x2], verbose=0)

        true_idx = class_id - 1
        prob_both = pred_both[:, true_idx]
        prob_v1 = pred_v1[:, true_idx]
        prob_v2 = pred_v2[:, true_idx]
        ratio_v1 = prob_v1 / (prob_v1 + prob_v2 + 1e-9)
        ratio_v2 = prob_v2 / (prob_v1 + prob_v2 + 1e-9)

        for cycle_idx in range(sample_count):
            rows.append(
                {
                    "Class": class_id,
                    "CycleIndex": cycle_idx + 1,
                    "Region": NODE_LOC[class_id],
                    "TrueProb_Both": float(prob_both[cycle_idx]),
                    "TrueProb_V1Only": float(prob_v1[cycle_idx]),
                    "TrueProb_V2Only": float(prob_v2[cycle_idx]),
                    "V1_Ratio": float(ratio_v1[cycle_idx]),
                    "V2_Ratio": float(ratio_v2[cycle_idx]),
                    "PredictedClass_Both": int(np.argmax(pred_both[cycle_idx]) + 1),
                }
            )

    return pd.DataFrame(rows)


def summarize_model_channel_ratio(df):
    summary = (
        df.groupby(["Class", "Region"], sort=True)
        .agg(
            Count=("CycleIndex", "count"),
            Mean_TrueProb_Both=("TrueProb_Both", "mean"),
            Mean_TrueProb_V1Only=("TrueProb_V1Only", "mean"),
            Mean_TrueProb_V2Only=("TrueProb_V2Only", "mean"),
            Mean_V1_Ratio=("V1_Ratio", "mean"),
            Median_V1_Ratio=("V1_Ratio", "median"),
            Std_V1_Ratio=("V1_Ratio", "std"),
        )
        .reset_index()
    )
    summary["Std_V1_Ratio"] = summary["Std_V1_Ratio"].fillna(0.0)
    return summary


def plot_model_channel_ratio_violin(df):
    plt.figure(figsize=(8.8, 3.2))
    ax = sns.violinplot(
        data=df,
        x="Class",
        y="V1_Ratio",
        inner=None,
        color="#efb4d7",
        linewidth=0.5,
        cut=0,
        density_norm="width",
    )
    sns.pointplot(
        data=df,
        x="Class",
        y="V1_Ratio",
        estimator=np.median,
        errorbar=None,
        color="#2a9d8f",
        markers="_",
        linestyle="none",
        ax=ax,
    )

    ax.axhline(0.5, color="#7a7a7a", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_title("Model Channel Ratio Analysis for Classes 1-18", fontsize=NC_TITLE_SIZE, fontname=NC_FONT, pad=4)
    ax.set_xlabel("Class", fontsize=NC_LABEL_SIZE, fontname=NC_FONT)
    ax.set_ylabel("V1 / (V1 + V2) on true-class output", fontsize=NC_LABEL_SIZE, fontname=NC_FONT)
    ax.tick_params(axis="both", labelsize=NC_TICK_SIZE, length=2.5, width=0.6)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontname(NC_FONT)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Class1_18_Model_Channel_Ratio_Violin.svg", format="svg")
    plt.close()


def model_channel_ratio_significance(df):
    rows = []
    class_groups = [grp["V1_Ratio"].to_numpy() for _, grp in df.groupby("Class")]
    if len(class_groups) >= 2:
        kw_stat, kw_p = kruskal(*class_groups)
        rows.append(
            {
                "Test": "Kruskal-Wallis across classes",
                "Metric": "V1_Ratio",
                "Statistic": float(kw_stat),
                "PValue": float(kw_p),
            }
        )

    for region_name, grp in df.groupby("Region"):
        v1_ratio = grp["V1_Ratio"].to_numpy()
        rows.append(
            {
                "Test": f"{region_name} mean ratio",
                "Metric": "V1_Ratio",
                "Statistic": float(np.mean(v1_ratio)),
                "PValue": np.nan,
            }
        )

    return pd.DataFrame(rows)


def plot_model_channel_ratio_class_grid(df):
    melt_df = df.melt(
        id_vars=["Class", "Region"],
        value_vars=["V1_Ratio", "V2_Ratio"],
        var_name="Channel",
        value_name="Ratio",
    )
    melt_df["Channel"] = melt_df["Channel"].map({"V1_Ratio": "V1", "V2_Ratio": "V2"})

    fig, axes = plt.subplots(6, 3, figsize=(7.2, 10.8), sharey=True)
    axes = axes.flatten()
    palette = {"V1": "#4ea8de", "V2": "#f4a261"}

    for idx, class_id in enumerate(range(1, 19)):
        ax = axes[idx]
        class_df = melt_df[melt_df["Class"] == class_id]
        sns.violinplot(
            data=class_df,
            x="Channel",
            y="Ratio",
            hue="Channel",
            density_norm="width",
            inner=None,
            cut=0,
            linewidth=0.5,
            palette=palette,
            dodge=False,
            legend=False,
            ax=ax,
        )
        sns.pointplot(
            data=class_df,
            x="Channel",
            y="Ratio",
            estimator=np.median,
            errorbar=None,
            hue="Channel",
            dodge=False,
            markers="_",
            palette={"V1": "#1d3557", "V2": "#9c6644"},
            linestyle="none",
            legend=False,
            ax=ax,
        )
        ax.axhline(0.5, color="#b0b0b0", linestyle="--", linewidth=0.6, alpha=0.8)
        ax.set_title(f"C{class_id}", fontsize=NC_TITLE_SIZE, fontname=NC_FONT, pad=2)
        ax.set_xlabel("")
        ax.set_ylabel("Ratio" if idx % 3 == 0 else "", fontsize=NC_LABEL_SIZE, fontname=NC_FONT)
        ax.set_ylim(0, 1.02)
        ax.tick_params(axis="both", labelsize=NC_TICK_SIZE, length=2, width=0.6)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontname(NC_FONT)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.6)
        ax.spines["bottom"].set_linewidth(0.6)

    plt.suptitle("Class 1-18 Channel Ratio Distributions", fontsize=NC_TITLE_SIZE, fontname=NC_FONT, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    plt.savefig(OUTPUT_DIR / "Class1_18_Channel_Ratio_18Violin.svg", format="svg")
    plt.close()


def plot_model_channel_ratio_scatter(df):
    summary_rows = []
    for class_id in range(1, 19):
        avg_v1 = read_average_wave(class_id, 1)
        avg_v2 = read_average_wave(class_id, 2)
        amp_v1 = float(np.ptp(avg_v1))
        amp_v2 = float(np.ptp(avg_v2))
        ratio = amp_v1 / (amp_v1 + amp_v2 + 1e-9)
        summary_rows.append(
            {
                "Class": class_id,
                "Region": NODE_LOC[class_id],
                "Amplitude_V1": amp_v1,
                "Amplitude_V2": amp_v2,
                "Amplitude_Ratio_V1": ratio,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        OUTPUT_DIR / "Class1_18_Channel_Amplitude_Ratio.csv",
        index=False,
        encoding="utf_8_sig",
    )

    palette = {
        "Thumb": "#6d597a",
        "Index": "#b56576",
        "Middle": "#e56b6f",
        "Ring": "#eaac8b",
        "Pinky": "#355070",
        "Palm": "#2a9d8f",
    }

    plt.figure(figsize=(6.144, 2.048))
    ax = sns.scatterplot(
        data=summary,
        x="Class",
        y="Amplitude_Ratio_V1",
        hue="Region",
        palette=palette,
        s=64,
        edgecolor="white",
        linewidth=0.7,
    )
    ax.plot(
        summary["Class"],
        summary["Amplitude_Ratio_V1"],
        color="#8d99ae",
        linewidth=0.9,
        alpha=0.8,
        zorder=1,
    )

    for _, row in summary.iterrows():
        ax.text(
            row["Class"],
            row["Amplitude_Ratio_V1"] + 0.025,
            str(int(row["Class"])),
            ha="center",
            fontsize=6.5,
            fontname=NC_FONT,
            color="#303030",
        )

    ax.axhline(0.5, linestyle="--", color="#9a9a9a", linewidth=0.8, alpha=0.8)
    ax.set_xlim(0.5, 18.5)
    ax.set_ylim(0.2, 0.8)
    ax.set_title("Class 1-18 Channel Amplitude Ratio", fontsize=NC_TITLE_SIZE, fontname=NC_FONT, pad=4)
    ax.set_xlabel("Class", fontsize=NC_LABEL_SIZE, fontname=NC_FONT)
    ax.set_ylabel("Amp_V1 / (Amp_V1 + Amp_V2)", fontsize=NC_LABEL_SIZE, fontname=NC_FONT)
    ax.set_xticks(range(1, 19))
    ax.tick_params(axis="both", labelsize=NC_TICK_SIZE, length=2.5, width=0.6)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontname(NC_FONT)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.legend(
        title="Region",
        frameon=False,
        fontsize=6.2,
        title_fontsize=6.5,
        loc="upper right",
        ncol=3,
        columnspacing=0.8,
        handletextpad=0.3,
        borderaxespad=0.2,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Class1_18_Channel_Ratio_Scatter.svg", format="svg")
    plt.close()


def export_top8_composite(additive_df):
    top8 = additive_df.sort_values(by="AdditiveR2", ascending=False).head(8).copy()
    top8.insert(0, "Order", range(1, len(top8) + 1))
    top8["Relation"] = top8.apply(
        lambda r: f"{int(r['Class'])}: [{', '.join(r['Components'].split(','))}]",
        axis=1,
    )
    out = top8[["Order", "Class", "AdditiveR2", "Relation"]]
    out.to_csv(OUTPUT_DIR / "Top8_Composite_For_Origin.csv", index=False, encoding="utf_8_sig")
    return out


def export_r2_19_22(additive_df):
    out = additive_df[additive_df["Class"].isin([19, 20, 21, 22])][["Class", "Components", "AdditiveR2"]].copy()
    out["Relation"] = out.apply(
        lambda r: f"{int(r['Class'])}: [{', '.join(str(v) for v in str(r['Components']).split(','))}]",
        axis=1,
    )
    out = out[["Class", "Components", "Relation", "AdditiveR2"]].sort_values("Class")
    out.to_csv(OUTPUT_DIR / "Class19_22_R2.csv", index=False, encoding="utf_8_sig")
    return out


def build_summary_tables(similarity_df, additive_df):
    strongest_pairs = similarity_df.sort_values(by="CombinedSimilarity", ascending=False).head(20)
    strongest_pairs.to_csv(OUTPUT_DIR / "Top_Similar_Class_Pairs.csv", index=False, encoding="utf_8_sig")

    if not additive_df.empty:
        additive_df.to_csv(OUTPUT_DIR / "Composite_Additive_Report.csv", index=False, encoding="utf_8_sig")


def ensure_inputs():
    missing = []
    for class_id in range(1, NUM_CLASSES + 1):
        expected = [
            TOTAL_WAVE_DIR / f"Class{class_id}" / f"Average_V1_{class_id}.csv",
            TOTAL_WAVE_DIR / f"Class{class_id}" / f"Average_V2_{class_id}.csv",
            TOTAL_WAVE_DIR / f"Class{class_id}" / f"Cycles_V1_{class_id}.csv",
            TOTAL_WAVE_DIR / f"Class{class_id}" / f"Cycles_V2_{class_id}.csv",
        ]
        missing.extend(str(path) for path in expected if not path.exists())
    if missing:
        raise FileNotFoundError("Missing input files:\n" + "\n".join(missing[:12]))


def main():
    sns.set_theme(style="white")
    plt.rcParams["font.sans-serif"] = [NC_FONT]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = NC_LABEL_SIZE
    plt.rcParams["axes.titlesize"] = NC_TITLE_SIZE
    plt.rcParams["axes.labelsize"] = NC_LABEL_SIZE
    plt.rcParams["xtick.labelsize"] = NC_TICK_SIZE
    plt.rcParams["ytick.labelsize"] = NC_TICK_SIZE

    OUTPUT_DIR.mkdir(exist_ok=True)
    ensure_inputs()

    class_data = {class_id: load_class_data(class_id) for class_id in range(1, NUM_CLASSES + 1)}
    similarity_df, similarity_matrix = build_similarity_table(class_data)
    additive_df = additive_fit_report(class_data)
    model_ratio_df = build_model_channel_ratio_report()
    plot_model_channel_ratio_scatter(model_ratio_df)
    top8_df = export_top8_composite(additive_df)
    r2_19_22_df = export_r2_19_22(additive_df)
    similarity_matrix_df = pd.DataFrame(
        similarity_matrix,
        index=[f"Class{i}" for i in range(1, NUM_CLASSES + 1)],
        columns=[f"Class{i}" for i in range(1, NUM_CLASSES + 1)],
    )
    similarity_matrix_df.to_csv(
        OUTPUT_DIR / "Class_Similarity_Matrix.csv",
        encoding="utf_8_sig",
    )
    plot_similarity_heatmap(similarity_matrix)

    print("Generated outputs:")
    print(f"- {OUTPUT_DIR / 'Class1_18_Channel_Ratio_Scatter.svg'}")
    print(f"- {OUTPUT_DIR / 'Class1_18_Channel_Amplitude_Ratio.csv'}")
    print(f"- {OUTPUT_DIR / 'Class19_22_R2.csv'}")
    print(f"- {OUTPUT_DIR / 'Top8_Composite_For_Origin.csv'}")
    print(f"- {OUTPUT_DIR / 'Class_Similarity_Matrix.csv'}")
    print(f"- {OUTPUT_DIR / 'Class_Similarity_Heatmap.svg'}")

    print("\nClass 19-22 R2:")
    for _, row in r2_19_22_df.iterrows():
        print(f"Class {int(row['Class'])}: R2={row['AdditiveR2']:.3f} | {row['Relation']}")

    print("\nTop 8 composite:")
    for _, row in top8_df.iterrows():
        print(f"{int(row['Order'])}. Class {int(row['Class'])}: R2={row['AdditiveR2']:.3f} | {row['Relation']}")


if __name__ == "__main__":
    main()
