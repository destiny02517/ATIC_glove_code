import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from matplotlib import rcParams

# —— 1. NC Publication Standard Setup ——
config = {
    "font.family": 'serif',
    "font.serif": ['Times New Roman'],
    "font.size": 12,
    "axes.unicode_minus": False,
    "mathtext.fontset": 'stix'
}
rcParams.update(config)

# —— 2. Parameter Configuration ——
categories = ["PLA", "TPU50", "TPU70"]
int8_min, int8_max = -128, 127
gap_size = 1000  # 类别间的视觉间隔点数
num_cols_to_avg = 20  # 每个 CSV 文件抽取 20 列计算均值特征
script_dir = os.path.dirname(os.path.abspath(__file__))

all_orig_feat = []  # 存储拼接后的原始特征波形
all_recon_feat = []  # 存储拼接后的量化还原特征波形
cat_ticks = []

print("🚀 Extracting characteristic waveforms from each CSV (20 columns per file)...")

# —— 3. Data Processing Loop ——
for cat in categories:
    cat_path = os.path.join(script_dir, cat)
    orig_sub = os.path.join(cat_path, "locat")

    # 自动定位量化文件夹
    try:
        quant_sub_name = [d for d in os.listdir(cat_path) if cat in d and "float to int" in d][0]
        quant_sub = os.path.join(cat_path, quant_sub_name)
    except Exception:
        print(f"⚠️ Skip {cat}: Quantization folder not found.")
        continue

    # 按编号排序读取所有 CSV
    files = sorted([f for f in os.listdir(orig_sub) if f.endswith('.csv')],
                   key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else x)

    start_idx = len(all_orig_feat)

    for file_name in files:
        f_orig_path = os.path.join(orig_sub, file_name)
        f_quant_path = os.path.join(quant_sub, file_name)

        if not os.path.exists(f_quant_path): continue

        try:
            # 读取完整 CSV (行=采样点, 列=周期)
            df_o = pd.read_csv(f_orig_path, header=None)
            df_q = pd.read_csv(f_quant_path, header=None)

            # 随机抽取 20 列（如果不足 20 列则取全部）
            total_cols = df_o.shape[1]
            selected = random.sample(range(total_cols), min(num_cols_to_avg, total_cols))

            # 提取数据并转为 NumPy
            data_o = df_o.iloc[:, selected].values
            data_q = df_q.iloc[:, selected].values

            # 【动态校准】：计算该文件的真实最值，消除如 PLA 出现的 Offset 偏置
            c_min, c_max = np.min(data_o), np.max(data_o)

            # 反量化还原
            data_r = (data_q.astype(float) - int8_min) / (int8_max - int8_min) * (c_max - c_min) + c_min

            # —— 核心步骤：计算该文件的特征波形（20列取平均） ——
            feat_o = np.mean(data_o, axis=1)
            feat_r = np.mean(data_r, axis=1)

            all_orig_feat.extend(feat_o)
            all_recon_feat.extend(feat_r)

        except Exception as e:
            print(f"Error in {cat}/{file_name}: {e}")

    # 记录类别标注位置
    end_idx = len(all_orig_feat)
    cat_ticks.append(((start_idx + end_idx) / 2, cat))

    # 加入 NaN 间隔，防止绘图连线
    all_orig_feat.extend([np.nan] * gap_size)
    all_recon_feat.extend([np.nan] * gap_size)

# —— 4. Professional Visualization ——
plt.figure(figsize=(18, 6))

# 绘图：原始 vs 还原
plt.plot(all_orig_feat, label="Original Feature (20-col Mean)", color='#1f77b4', linewidth=1.0, alpha=0.8)
plt.plot(all_recon_feat, label="Reconstructed Feature (Int8)", color='#d62728', linestyle='--', linewidth=0.8,
         alpha=0.9)

# 计算全局 RMSE (排除 NaN)
arr_o = np.array(all_orig_feat)
arr_r = np.array(all_recon_feat)
mask = ~np.isnan(arr_o)
global_rmse = np.sqrt(np.mean((arr_o[mask] - arr_r[mask]) ** 2))

# 类别文字标注
y_min, y_max = plt.gca().get_ylim()
for pos, name in cat_ticks:
    plt.text(pos, y_max - (y_max - y_min) * 0.05, name, fontsize=14, fontweight='bold', ha='center')

# 图表修饰
plt.title(f"Statistical Feature Waveform Comparison (RMSE: {global_rmse:.6f})", fontsize=16, pad=20)
plt.xlabel("Concatenated Characteristic Cycles (20-column Average per Sample)", fontsize=14)
plt.ylabel("Normalized Amplitude", fontsize=14)
plt.legend(loc='upper right', frameon=True, fontsize=11)
plt.grid(True, linestyle=':', alpha=0.4)

# 保存高分辨率图片
plt.tight_layout()
save_path = os.path.join(script_dir, "NC_Enhanced_Feature_Comparison.png")
plt.savefig(save_path, dpi=600)
plt.show()

print(f"✨ Feature extraction complete! Figure saved to: {save_path}")