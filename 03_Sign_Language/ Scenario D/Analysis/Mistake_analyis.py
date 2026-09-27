import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# ==========================================
# 0. 全局字体配置 (设置为 Arial)
# ==========================================
plt.rcParams['font.sans-serif'] = ['Arial']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False   # 解决保存图像是负号'-'显示为方块的问题

# ==========================================
# 1. 环境与路径配置
# ==========================================
MASTER_DIR = "Sign_Language_Error_Analysis"
if not os.path.exists(MASTER_DIR):
    os.makedirs(MASTER_DIR)


# ==========================================
# 2. 加载属性表
# ==========================================
def load_sign_properties(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: 找不到属性表 '{csv_path}'")
        return None
    df_prop = pd.read_csv(csv_path)
    df_prop['Sample number'] = df_prop['Sample number'].astype(str)
    return df_prop.set_index('Sample number').to_dict('index')


# ==========================================
# 3. 误差归因诊断逻辑
# ==========================================
def diagnose_sign_reason(t_real_id, p_real_id, prop_dict):
    p1 = prop_dict.get(str(t_real_id), {})
    p2 = prop_dict.get(str(p_real_id), {})
    if not p1 or not p2: return "Cross-Domain Variance"

    reasons = []
    # 语义归因
    if p1.get('Group') == p2.get('Group'):
        reasons.append("Same Semantic Group")
    # 物理归因
    if p1.get('Location') == p2.get('Location'):
        reasons.append("Location Overlap")
    if p1.get('Motion_Type') == p2.get('Motion_Type'):
        reasons.append("Similar Motion Type")
    elif p1.get('Trajectory') == p2.get('Trajectory'):
        reasons.append("Similar Trajectory")

    return " & ".join(reasons[:2]) if reasons else "Cross-Domain Variance"


# ==========================================
# 4. 自动化分析核心逻辑
# ==========================================
def run_full_analysis(txt_path, prop_file):
    if not os.path.exists(txt_path):
        print(f"Error: 找不到预测文件 '{txt_path}'")
        return

    # 1. 加载预测结果 (One-hot 格式) 并转换 ID
    print("正在加载预测数据并转换 ID...")
    pred_onehot = np.loadtxt(txt_path)
    y_pred = np.argmax(pred_onehot, axis=1) + 1

    # 手语测试集固定参数：46类，每类30个样本
    num_classes = 46
    samples_per_class = 30
    total_samples = num_classes * samples_per_class
    y_true = np.repeat(np.arange(1, num_classes + 1), samples_per_class)

    prop_dict = load_sign_properties(prop_file)
    if not prop_dict: return

    # 2. 计算混淆矩阵与误判率
    all_labels = [str(i) for i in range(1, num_classes + 1)]
    cm = confusion_matrix(y_true.astype(str), y_pred.astype(str), labels=all_labels)
    correct_samples = np.trace(cm)
    total_errors = total_samples - correct_samples
    overall_error_rate = (total_errors / total_samples) * 100

    # 3. 绘制热力图 (Misclassification_Heatmap)
    cm_pc = np.divide(cm.astype('float'), samples_per_class,
                      out=np.zeros_like(cm.astype('float')), where=samples_per_class != 0) * 100

    cm_error = cm_pc.copy()
    # 将对角线（正确率）替换为该类别的总误判率 (100% - 准确率)
    for i in range(num_classes):
        cm_error[i, i] = 100 - cm_pc[i, i]

    plt.figure(figsize=(16, 12))
    # 使用 YlOrRd 颜色表，深色表示误差率越高
    sns.heatmap(cm_error, annot=False, cmap='YlOrRd', xticklabels=all_labels, yticklabels=all_labels)

    # 标题同步修改为 Overall Error Rate
    plt.title(f'Sign Language Misclassification Rate Heatmap (%)\nOverall Error Rate: {overall_error_rate:.2f}%',
              fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Word ID (Error Hotspots)', fontsize=12)
    plt.ylabel('True Word ID', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(MASTER_DIR, 'Misclassification_Heatmap.svg'), format='svg')
    plt.show()

    # 4. 归因诊断并生成详细报告 (Detailed_Error_Report)
    print("正在进行误差归因诊断...")
    error_records = []
    for i in range(len(all_labels)):
        for j in range(len(all_labels)):
            if i != j and cm[i, j] > 0:
                reason = diagnose_sign_reason(all_labels[i], all_labels[j], prop_dict)
                error_records.append({
                    "True_ID": all_labels[i],
                    "Pred_ID": all_labels[j],
                    "Error_Count": int(cm[i, j]),
                    "Error_Rate(%)": round(cm_pc[i, j], 2),
                    "Diagnosis": reason
                })

    df_err = pd.DataFrame(error_records)
    df_err.to_csv(os.path.join(MASTER_DIR, 'Detailed_Error_Report.csv'), index=False, encoding='utf_8_sig')

    # 5. 绘制全局概率分布图 (Error_Reason_Distribution)
    if not df_err.empty:
        summary_feature = df_err.groupby('Diagnosis').agg({'Error_Count': 'sum'}).reset_index()
        summary_feature['Occurrence_Probability'] = summary_feature['Error_Count'] / total_samples
        summary_feature = summary_feature.sort_values(by='Occurrence_Probability', ascending=False)

        plt.figure(figsize=(12, 8))
        ax = sns.barplot(
            data=summary_feature,
            x='Occurrence_Probability',
            y='Diagnosis',
            hue='Diagnosis',
            palette='viridis',
            legend=False
        )

        for container in ax.containers:
            labels = [f'{v.get_width() * 100:.2f}%' for v in container]
            ax.bar_label(container, labels=labels, padding=5, fontweight='bold')

        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
        plt.title(f'Global Misclassification Causes Distribution\n'
                  f'Overall Error Rate: {overall_error_rate:.2f}% | Total Errors: {total_errors}',
                  fontsize=14, fontweight='bold', pad=20)

        plt.xlabel('Misclassification Probability (relative to total 1380 samples)', fontsize=12)
        plt.ylabel('Attributed Physical/Semantic Cause', fontsize=12)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()

        plt.savefig(os.path.join(MASTER_DIR, 'Error_Reason_Distribution.svg'), format='svg')
        plt.show()

    print(f"\n--- 分析完成 ---")
    print(f"详细报告已保存至: {MASTER_DIR}")


# ==========================================
# 5. 执行分析
# ==========================================
if __name__ == "__main__":
    PRED_FILE = 'accuracy_fusion.txt'
    PROP_FILE = 'Sign_Properties.csv'

    run_full_analysis(PRED_FILE, PROP_FILE)