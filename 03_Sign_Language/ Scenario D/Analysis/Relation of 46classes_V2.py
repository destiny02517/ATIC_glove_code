import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Circle
from matplotlib.colors import LinearSegmentedColormap

# ==========================================
# 1. 基础配置与数据处理
# ==========================================
base_dir = r"Wave"
output_dir = "Correlation_Analysis_Results"
if not os.path.exists(output_dir): os.makedirs(output_dir)

start_idx, end_idx, num_points = 108, 153, 2000

LABELS_MAP = {
    108: "you", 109: "me", 110: "father", 111: "my", 112: "why",
    113: "where", 114: "how", 115: "what", 116: "who", 117: "lesson",
    118: "book", 119: "like", 120: "phone", 121: "know", 122: "want",
    123: "have", 124: "train", 125: "clear", 126: "must", 127: "tell",
    128: "many", 129: "sick", 130: "good", 131: "better", 132: "disbelieve",
    133: "red", 134: "yellow", 135: "white", 136: "pay", 137: "tv",
    138: "disk", 139: "catsup", 140: "goodbye", 141: "give me", 142: "light flash",
    143: "wife", 144: "now", 145: "old", 146: "see", 147: "hawk",
    148: "a", 149: "b", 150: "L", 151: "I", 152: "13", 153: "100"
}

GROUPS = {
    "Group A (Pronoun/Kin)": [108, 109, 110, 111, 143],
    "Group B (Wh-Q/Time)": [112, 113, 114, 115, 116, 144],
    "Group C (Object/Noun)": [117, 118, 120, 124, 137, 138, 139, 147],
    "Group D (Verb/Action)": [119, 121, 122, 123, 127, 132, 136, 146, 126],
    "Group E (Adj/Color)": [125, 128, 129, 130, 131, 133, 134, 135, 145],
    "Group F (Letter/Num)": [148, 149, 150, 151, 152, 153],
    "Group G (Phrase)": [140, 141, 142]
}

GROUP_COLORS = {
    "Group A (Pronoun/Kin)": "#F1C40F", "Group B (Wh-Q/Time)": "#F39C12",
    "Group C (Object/Noun)": "#E67E22", "Group D (Verb/Action)": "#E74C3C",
    "Group E (Adj/Color)": "#9B59B6", "Group F (Letter/Num)": "#2ECC71",
    "Group G (Phrase)": "#16A085"
}


def load_and_process_data():
    all_features = []
    for i in range(start_idx, end_idx + 1):
        try:
            folder_path = os.path.join(base_dir, f"scope_{i}_export_csv")
            d1 = pd.read_csv(os.path.join(folder_path, f"mean_curve_ch1_{i}.csv"), skiprows=1, header=None).iloc[:,
                 1].values[:num_points]
            d2 = pd.read_csv(os.path.join(folder_path, f"mean_curve_ch2_{i}.csv"), skiprows=1, header=None).iloc[:,
                 1].values[:num_points]
            all_features.append(np.concatenate([d1, d2]))
        except:
            all_features.append(np.zeros(num_points * 2))
    return np.array(all_features)


features = load_and_process_data()
corr_matrix = np.corrcoef(features)

# ==========================================
# [新增功能] 提取并打印高相关性 (>0.95) 组合
# ==========================================
print("\n" + "=" * 50)
print("Finding pairs with Correlation > 0.95 ...")
print("=" * 50)

high_corr_list = []
n_dim = corr_matrix.shape[0]

# 遍历矩阵的上三角部分 (不包含对角线)
for i in range(n_dim):
    for j in range(i + 1, n_dim):
        r_val = corr_matrix[i, j]

        # 核心判断逻辑
        if r_val > 0.95:
            # 将矩阵索引 i, j 还原为真实的 ID (start_idx + i)
            id_1 = start_idx + i
            id_2 = start_idx + j

            # 获取对应的单词标签
            word_1 = LABELS_MAP.get(id_1, str(id_1))
            word_2 = LABELS_MAP.get(id_2, str(id_2))

            # 打印到控制台
            print(f"[{word_1}] <---> [{word_2}] : {r_val:.5f}")

            # 存入列表以便保存
            high_corr_list.append({
                "ID_1": id_1,
                "Word_1": word_1,
                "ID_2": id_2,
                "Word_2": word_2,
                "Correlation": r_val
            })

# 如果找到了，保存为 CSV
if high_corr_list:
    df_high_corr = pd.DataFrame(high_corr_list)
    # 按相关性从高到低排序
    df_high_corr = df_high_corr.sort_values(by="Correlation", ascending=False)

    csv_path = os.path.join(output_dir, "High_Correlation_Pairs_GT_0.95.csv")
    df_high_corr.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n[Success] Found {len(high_corr_list)} pairs.")
    print(f"List saved to: {csv_path}")
else:
    print("\n[Result] No pairs found with correlation > 0.95")

print("=" * 50 + "\n")

# ==========================================
# 2. 绘图核心逻辑 (合并坐标系)
# ==========================================
# 修改字体配置
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

fig, ax = plt.subplots(figsize=(14, 12))

colors = ["#096897", "#ffffff", "#ff5e5e"]
custom_cmap = LinearSegmentedColormap.from_list("custom_red_blue", colors)

# 绘制热图
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, ax=ax, mask=mask,
            cmap=custom_cmap,
            center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.5, linecolor='white', annot=True, fmt=".1f",
            annot_kws={"size": 5},
            cbar_kws={"shrink": 0.3, "label": "Pearson Correlation (r)", "location": "left", "pad": 0.05})
labels = [LABELS_MAP.get(i, str(i)) for i in range(start_idx, end_idx + 1)]
n = len(labels)
ax.set_xticklabels(labels, rotation=90, fontsize=8)
ax.set_yticklabels(labels, rotation=0, fontsize=8)

# 绘制对角线彩色圆点
for i in range(n):
    sid = start_idx + i
    color = next((GROUP_COLORS[gn] for gn, sids in GROUPS.items() if sid in sids), 'gray')
    ax.add_patch(Circle((i + 0.5, i + 0.5), radius=0.3, color=color, ec='white', zorder=10))

# ==========================================
# 3. 改进逻辑：斜率对齐排列
# ==========================================
group_names = list(GROUPS.keys())
n_groups = len(group_names)

offset_base = 10.0

for i, name in enumerate(group_names):
    color = GROUP_COLORS[name]

    avg_idx = np.mean([sid - start_idx for sid in GROUPS[name]])
    ly = avg_idx

    if "Group F" in name:
        ly -= 5.0

    lx = ly + offset_base + 2.0

    ax.scatter(lx, ly, s=50, color=color, edgecolors='white', zorder=11, clip_on=False)

    ax.text(lx + 0.8, ly, name, va='center', ha='left',
            fontsize=18, fontweight='bold', color=color, clip_on=False)

    for sid in GROUPS[name]:
        idx = sid - start_idx
        con = ConnectionPatch(xyA=(idx + 0.5, idx + 0.5), xyB=(lx, ly),
                              coordsA=ax.transData, coordsB=ax.transData,
                              color=color, alpha=1, linewidth=1.5,
                              connectionstyle="arc3,rad=-0.15")
        ax.add_artist(con)

# ==========================================
# 4. 最终修饰
# ==========================================
ax.set_xlim(0, n + 10)
ax.set_ylim(n, 0)
ax.set_title("Tight Mantel Style Analysis: Slope Aligned", fontsize=18, pad=5)

# save_path = os.path.join(output_dir, "Mantel_Slope_Aligned.svg")
plt.savefig(save_path, format='svg', bbox_inches='tight', dpi=300)
plt.show()