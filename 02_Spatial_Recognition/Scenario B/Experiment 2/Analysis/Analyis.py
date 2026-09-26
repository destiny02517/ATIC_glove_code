import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
# ==========================================
# 1. 物理结构定义 (根据手掌布局建模)
# ==========================================
CLASS_NODES = {
    1: [1], 2: [2], 3: [3], 4: [4], 5: [5], 6: [6], 7: [7], 8: [8], 9: [9], 10: [10],
    11: [11], 12: [12], 13: [13], 14: [14], 15: [15], 16: [16], 17: [17], 18: [18],
    19: [5, 6], 20: [13, 14, 15, 16], 21: [7, 9], 22: [17, 12], 23: [14, 15], 24: [13, 16],
    25: [3, 5, 7], 26: [7, 8], 27: [1, 2], 28: [3, 5, 7, 9],
    29: [14, 15, 18], 30: [11, 12, 13, 14, 15, 16, 17, 18]
}

NODE_LOC = {
    1: "Thumb", 2: "Thumb", 3: "Index", 4: "Index", 5: "Middle", 6: "Middle",
    7: "Ring", 8: "Ring", 9: "Pinky", 10: "Pinky",
    11: "Palm", 12: "Palm", 13: "Palm", 14: "Palm", 15: "Palm", 16: "Palm", 17: "Palm", 18: "Palm"
}

NODE_DIST = {
    1: 1, 3: 1, 5: 1, 7: 1, 9: 1,
    2: 2, 4: 2, 6: 2, 8: 2, 10: 2,
    13: 3, 14: 3, 15: 3, 16: 3,
    11: 4, 12: 4, 17: 4, 18: 4
}


def get_class_info(c_id):
    nodes = CLASS_NODES.get(c_id, [])
    locs = list(set([NODE_LOC.get(n) for n in nodes if n in NODE_LOC]))
    dists = list(set([NODE_DIST.get(n) for n in nodes if n in NODE_DIST]))
    return nodes, locs, dists


def diagnose_reason(c1, c2, corr):
    nodes1, locs1, dist1 = get_class_info(c1)
    nodes2, locs2, dist2 = get_class_info(c2)
    if set(nodes1).issubset(set(nodes2)) or set(nodes2).issubset(set(nodes1)):
        return "Activation Overlap"
    if any(l in locs2 for l in locs1):
        return "Spatial Coupling"
    if set(dist1) == set(dist2) and corr > 0.95:
        return "Structural Symmetry"
    if corr > 0.98:
        return "Signal Similarity"
    return "Unexplained Variability"


# ==========================================
# 2. 绘图功能
# ==========================================
def plot_diagnosis_distribution(df, overall_error, total_samples):
    # 统计每种误分类机制对应的总误判样本数和总误差率贡献
    samples_per_class = total_samples // 30

    summary_feature = df.groupby('Diagnosis', as_index=False).agg(
        Error_Count=('Rate(%)', lambda x: int(round(np.sum(x) * samples_per_class / 100)))
    )

    # 转为相对于总样本数的误判概率
    summary_feature['Occurrence_Probability'] = summary_feature['Error_Count'] / total_samples

    # 按概率从大到小排序
    summary_feature = summary_feature.sort_values(by='Occurrence_Probability', ascending=False)

    total_errors = summary_feature['Error_Count'].sum()

    plt.figure(figsize=(12, 8))
    ax = sns.barplot(
        data=summary_feature,
        x='Occurrence_Probability',
        y='Diagnosis',
        hue='Diagnosis',
        palette='viridis',
        legend=False
    )

    # 每个柱子右侧标百分比
    for container in ax.containers:
        labels = [f'{v.get_width() * 100:.2f}%' for v in container]
        ax.bar_label(container, labels=labels, padding=5, fontweight='bold', fontsize=12)

    # x轴改成百分比显示
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))

    plt.title(
        f'Contribution of Different Misclassification Mechanisms\n'
        f'Overall Error Rate: {overall_error:.2f}% | Total Errors: {total_errors}',
        fontsize=16,
        fontweight='bold',
        pad=20
    )

    plt.xlabel('Contribution to Overall Error Rate (%)', fontsize=14)
    plt.ylabel('Misclassification Mechanism', fontsize=14)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('Diagnosis_Distribution_Bar.svg', format='svg')
    plt.show()

# ==========================================
# 3. 主分析程序
# ==========================================
def run_analysis():
    txt_path = 'Accuracy/accuracy_fusion.txt'
    if not os.path.exists(txt_path):
        print(f"Error: {txt_path} not found.")
        return

    data = np.loadtxt(txt_path)
    num_classes = 30
    samples_per_class = len(data) // num_classes

    conf_matrix = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        start, end = i * samples_per_class, (i + 1) * samples_per_class
        preds = np.argmax(data[start:end], axis=1)
        for p in preds:
            conf_matrix[i, p] += 1
    conf_matrix_pc = (conf_matrix / samples_per_class) * 100

    # 计算整体准确率和误判率
    overall_acc = np.mean(np.diagonal(conf_matrix_pc))
    overall_error = 100 - overall_acc

    plt.rcParams['font.sans-serif'] = ['Arial']

    # 1. 完整混淆矩阵 (显示准确率)
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix_pc.T, annot=True, fmt=".0f", cmap='Blues',
                xticklabels=range(1, 31), yticklabels=range(1, 31))
    plt.title(f'Confusion Matrix (Overall Accuracy: {overall_acc:.2f}%)',
              fontsize=18, fontweight='bold')

    plt.xlabel('True Labels',
               fontsize=14, fontweight='bold')

    plt.ylabel('Predicted Labels',
               fontsize=14, fontweight='bold')
    plt.savefig('Confusion_Matrix_Full.svg', format='svg')

    # 2. 误判热力分布 (显示误判率)
    plt.figure(figsize=(12, 10))
    error_matrix = conf_matrix_pc.copy()
    np.fill_diagonal(error_matrix, 0)
    sns.heatmap(error_matrix.T, annot=True, fmt=".1f", cmap='YlOrRd',
                xticklabels=range(1, 31), yticklabels=range(1, 31))
    plt.title(f'Misclassification Heatmap (Overall Error Rate: {overall_error:.2f}%)', fontsize=18)
    plt.xlabel('True Labels', fontsize=14)
    plt.ylabel('Predicted Labels', fontsize=14)
    plt.savefig('Misclassification_Heatmap.svg', format='svg')

    # 3. 原因诊断
    analysis_list = []
    print(f"\n--- System Report ---")
    print(f"Overall Accuracy: {overall_acc:.2f}%")
    print(f"Overall Error Rate: {overall_error:.2f}%")

    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and conf_matrix_pc[i, j] > 0:
                true_c, pred_c = i + 1, j + 1
                rate = conf_matrix_pc[i, j]
                try:
                    w1_t = pd.read_csv(f'Wave/Wave1/{true_c}.csv', header=None).mean(axis=1).values
                    w2_t = pd.read_csv(f'Wave/Wave2/{true_c}.csv', header=None).mean(axis=1).values
                    sig_t = np.concatenate([w1_t, w2_t])
                    w1_p = pd.read_csv(f'Wave/Wave1/{pred_c}.csv', header=None).mean(axis=1).values
                    w2_p = pd.read_csv(f'Wave/Wave2/{pred_c}.csv', header=None).mean(axis=1).values
                    sig_p = np.concatenate([w1_p, w2_p])
                    corr, _ = pearsonr(sig_t, sig_p)
                except:
                    corr = 0
                reason = diagnose_reason(true_c, pred_c, corr)
                analysis_list.append([true_c, pred_c, rate, corr, reason])

    df = pd.DataFrame(analysis_list, columns=['True_Class', 'Pred_Class', 'Rate(%)', 'Correlation', 'Diagnosis'])
    total_samples = num_classes * samples_per_class
    plot_diagnosis_distribution(df, overall_error, total_samples)
    df = df.sort_values(by='Rate(%)', ascending=False)
    df['Rate(%)'] = df['Rate(%)'].round(2)
    df['Correlation'] = df['Correlation'].round(3)
    df.to_csv('Full_Error_Diagnosis.csv', index=False, encoding='utf_8_sig')

    print("\nSVG Graphics & CSV Reports generated successfully.")


if __name__ == "__main__":
    run_analysis()