"""
evaluate_channels_correct.py
────────────────────────────────────────────────────────────────────────────────
正确评估单通道 vs 双通道贡献的方法：

  ❌ 错误做法：用零填充另一路输入（会导致特征分布崩溃，准确率虚低）
  ✅ 正确做法：分别训练三个独立模型
      - Model A：仅 CH_BE 输入
      - Model B：仅 CH_UE 输入
      - Model C：CH_BE + CH_UE 融合（即已有的 FusionSense.keras）

本脚本会：
  1. 加载已有融合模型（不修改）
  2. 新建并训练 CH_BE-only 模型
  3. 新建并训练 CH_UE-only 模型
  4. 汇总三者测试准确率并可视化
────────────────────────────────────────────────────────────────────────────────
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model, Model
from keras.layers import (Input, Convolution1D, Activation, MaxPooling1D,
                          Flatten, Dense, Dropout, concatenate)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ════════════════════════════════════════════════════════════
#  参数配置（与原训练脚本完全一致）
# ════════════════════════════════════════════════════════════
Label = list(range(1, 47))

TrainSampleNumber = 70
TestSampleNumber  = 30
ValiSampleNumber  = 30
ClassNumber       = 46

CH_BE_SampleLength  = 2000
CH_BE_ChannelNumber = 1
CH_BE_Channel       = [0]
CH_BE_Path          = "locat/"

CH_UE_SampleLength  = 2000
CH_UE_ChannelNumber = 1
CH_UE_Channel       = [0]
CH_UE_Path          = "sense/"

FUSION_MODEL_PATH  = "CNN/FusionSense.keras"
CH_BE_MODEL_PATH    = "CNN/CH_BE_only.keras"
CH_UE_MODEL_PATH    = "CNN/CH_UE_only.keras"

RESULTS_DIR        = "Multi CH accuracy results"          # 所有输出文件统一存放到此文件夹

EPOCHS     = 100
BATCH_SIZE = 64
LR         = 1e-4

# ════════════════════════════════════════════════════════════
#  数据加载（与原脚本完全一致）
# ════════════════════════════════════════════════════════════
def sample(array1):
    return (
        array1[:, :TrainSampleNumber],
        array1[:, TrainSampleNumber : TrainSampleNumber + TestSampleNumber],
        array1[:, TrainSampleNumber + TestSampleNumber :]
    )

def Load_data(SampleLength, ChannelNumber, channel, path):
    trainCol, testCol, valiCol = [], [], []
    for i in range(ClassNumber):
        fpath = path + str(i + 1) + ".csv"
        print(f"  加载: {fpath}")
        Array = np.array(pd.read_csv(fpath, header=None))[:, 0:]
        at, ae, av = sample(Array)
        trainCol.append(at); testCol.append(ae); valiCol.append(av)

    def _merge(col):
        merged = col[0]
        for c in col[1:]:
            merged = np.concatenate((merged, c), axis=1)
        merged = merged.T.reshape(-1, SampleLength, ChannelNumber)
        cut = [item.T[channel].T for item in merged]
        return np.array(cut)

    return _merge(trainCol), _merge(testCol), _merge(valiCol)

# ════════════════════════════════════════════════════════════
#  构建单输入 CNN 模型（结构与原脚本中的子网络完全相同）
# ════════════════════════════════════════════════════════════
def build_single_cnn(sample_length, channel_number, name_prefix=""):
    """
    与原脚本 creat_CH_BE_cnn / creat_CH_UE_cnn 子网络结构完全一致，
    但在末尾追加分类头（Dense → softmax），构成完整的独立分类器。
    """
    inputs = Input(shape=(sample_length, channel_number), name=f"{name_prefix}_input")
    x = inputs

    for filters in [8, 16, 32, 64, 128]:
        x = Convolution1D(filters=filters, kernel_size=5, padding='same')(x)
        x = Activation("relu")(x)
        x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)

    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(3000)(x)
    x = Activation("relu")(x)

    # 独立分类头
    x = Dense(ClassNumber)(x)
    outputs = Activation('softmax')(x)

    model = Model(inputs, outputs, name=f"{name_prefix}_classifier")
    return model

# ════════════════════════════════════════════════════════════
#  混淆矩阵绘制
# ════════════════════════════════════════════════════════════
def cm_plot(original_label, predict_label, accuracy, title_suffix="", save_path=None):
    cm = confusion_matrix(original_label, predict_label)
    plt.figure(figsize=(14, 12))
    plt.matshow(cm.T, fignum=1, cmap=plt.cm.Greens)
    plt.colorbar()
    for x in range(len(cm)):
        for y in range(len(cm)):
            val = cm[x, y]
            pct = val / TestSampleNumber * 100
            plt.annotate(
                f"{pct:.1f}%\n{val}",
                xy=(x, y), ha='center', va='center',
                color='white' if pct / 100 > 0.5 else 'black',
                fontsize=5
            )
    ticks = list(range(ClassNumber))
    plt.xticks(ticks, Label, fontsize=7, rotation=90)
    plt.yticks(ticks, Label, fontsize=7)
    plt.ylabel('Predicted label', fontsize=13)
    plt.xlabel('True label', fontsize=13)
    plt.title(f'Accuracy: {accuracy*100:.3f}%  [{title_suffix}]', fontsize=13, pad=40)
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"  混淆矩阵已保存 → {save_path}")
    plt.close()

# ════════════════════════════════════════════════════════════
#  训练曲线绘制
# ════════════════════════════════════════════════════════════
def plot_history(history, title_prefix, save_prefix):
    epochs = range(1, len(history.history['accuracy']) + 1)

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history.history['accuracy'],     color='magenta', label='Train')
    plt.plot(epochs, history.history['val_accuracy'], color='cyan',    label='Val')
    plt.title(f'{title_prefix} - Accuracy')
    plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_accuracy.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history.history['loss'],     color='magenta', label='Train')
    plt.plot(epochs, history.history['val_loss'], color='cyan',    label='Val')
    plt.title(f'{title_prefix} - Loss')
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_loss.png", dpi=150)
    plt.close()
    print(f"  训练曲线已保存 → {save_prefix}_accuracy.png / _loss.png")

# ════════════════════════════════════════════════════════════
#  准确率汇总条形图
# ════════════════════════════════════════════════════════════
def plot_accuracy_bar(results: dict, save_path="accuracy_comparison_correct.png"):
    modes  = list(results.keys())
    accs   = [results[m]['accuracy'] * 100 for m in modes]
    colors = ['#4C72B0', '#DD8452', '#55A868']

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(modes, accs, color=colors, width=0.45,
                  edgecolor='black', linewidth=0.8)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{acc:.2f}%",
                ha='center', va='bottom', fontsize=13, fontweight='bold')

    ax.set_ylim(0, 115)
    ax.set_ylabel('Test Accuracy (%)', fontsize=13)
    ax.set_title('Single-channel vs Dual-channel Accuracy\n(Independently Trained)', fontsize=13)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\n准确率对比图已保存 → {save_path}")
    plt.close()

# ════════════════════════════════════════════════════════════
#  主流程
# ════════════════════════════════════════════════════════════
def main():
    # GPU 配置
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"GPU 可用: {len(gpus)} 块")
    else:
        print("未检测到 GPU，使用 CPU")

    os.makedirs("CNN", exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"结果文件将保存至 → {RESULTS_DIR}/\n")

    # ── 标签 ──
    def make_labels(n_per_class):
        raw = [j for j in range(ClassNumber) for _ in range(n_per_class)]
        return raw, to_categorical(np.array(raw), ClassNumber)

    trainLabel_raw, trainLabel = make_labels(TrainSampleNumber)
    testLabel_raw,  testLabel  = make_labels(TestSampleNumber)
    valiLabel_raw,  valiLabel  = make_labels(ValiSampleNumber)

    # ── 加载数据 ──
    print("\n[1/3] 加载 CH_BE 数据...")
    trainCH_BE, testCH_BE, valiCH_BE = Load_data(
        CH_BE_SampleLength, CH_BE_ChannelNumber, CH_BE_Channel, CH_BE_Path)

    print("\n[2/3] 加载 CH_UE 数据...")
    trainCH_UE, testCH_UE, valiCH_UE = Load_data(
        CH_UE_SampleLength, CH_UE_ChannelNumber, CH_UE_Channel, CH_UE_Path)

    results = {}

    # ════════════════════════════════════
    #  A. 仅 CH_BE 独立分类器
    # ════════════════════════════════════
    print("\n" + "="*60)
    print("  [Model A] 训练 CH_BE-only 独立分类器")
    print("="*60)

    if os.path.exists(CH_BE_MODEL_PATH):
        print(f"  检测到已有模型 {CH_BE_MODEL_PATH}，直接加载...")
        model_ch_be = load_model(CH_BE_MODEL_PATH)
    else:
        model_ch_be = build_single_cnn(CH_BE_SampleLength, CH_BE_ChannelNumber, name_prefix="CH_BE")
        model_ch_be.compile(
            optimizer=Adam(learning_rate=LR),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print(model_ch_be.summary())
        history_ch_be = model_ch_be.fit(
            trainCH_BE, trainLabel,
            epochs=EPOCHS, batch_size=BATCH_SIZE,
            validation_data=(valiCH_BE, valiLabel),
            callbacks=[
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=0.5, patience=30, verbose=1),
                tf.keras.callbacks.ModelCheckpoint(
                    CH_BE_MODEL_PATH, monitor='val_accuracy',
                    save_best_only=True, mode='max', verbose=1)
            ]
        )
        plot_history(history_ch_be, "CH_BE only", os.path.join(RESULTS_DIR, "ch_be_only"))
        model_ch_be = load_model(CH_BE_MODEL_PATH)   # 加载最优权重

    loss_t, acc_t = model_ch_be.evaluate(testCH_BE, testLabel, verbose=0)
    pred_t = np.argmax(model_ch_be.predict(testCH_BE, verbose=0), axis=1)
    print(f"\n  CH_BE-only  Loss: {loss_t:.6f}  Accuracy: {acc_t*100:.3f}%")
    print(classification_report(testLabel_raw, pred_t,
                                target_names=[str(lb) for lb in Label],
                                zero_division=0))
    cm_plot(testLabel_raw, pred_t, acc_t,
            title_suffix="CH_BE only", save_path=os.path.join(RESULTS_DIR, "cm_ch_be_only.png"))
    results["CH_BE only"] = {'loss': loss_t, 'accuracy': acc_t}

    # ════════════════════════════════════
    #  B. 仅 CH_UE 独立分类器
    # ════════════════════════════════════
    print("\n" + "="*60)
    print("  [Model B] 训练 CH_UE-only 独立分类器")
    print("="*60)

    if os.path.exists(CH_UE_MODEL_PATH):
        print(f"  检测到已有模型 {CH_UE_MODEL_PATH}，直接加载...")
        model_ch_ue = load_model(CH_UE_MODEL_PATH)
    else:
        model_ch_ue = build_single_cnn(CH_UE_SampleLength, CH_UE_ChannelNumber, name_prefix="CH_UE")
        model_ch_ue.compile(
            optimizer=Adam(learning_rate=LR),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print(model_ch_ue.summary())
        history_ch_ue = model_ch_ue.fit(
            trainCH_UE, trainLabel,
            epochs=EPOCHS, batch_size=BATCH_SIZE,
            validation_data=(valiCH_UE, valiLabel),
            callbacks=[
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=0.5, patience=30, verbose=1),
                tf.keras.callbacks.ModelCheckpoint(
                    CH_UE_MODEL_PATH, monitor='val_accuracy',
                    save_best_only=True, mode='max', verbose=1)
            ]
        )
        plot_history(history_ch_ue, "CH_UE only", os.path.join(RESULTS_DIR, "ch_ue_only"))
        model_ch_ue = load_model(CH_UE_MODEL_PATH)

    loss_p, acc_p = model_ch_ue.evaluate(testCH_UE, testLabel, verbose=0)
    pred_p = np.argmax(model_ch_ue.predict(testCH_UE, verbose=0), axis=1)
    print(f"\n  CH_UE-only  Loss: {loss_p:.6f}  Accuracy: {acc_p*100:.3f}%")
    print(classification_report(testLabel_raw, pred_p,
                                target_names=[str(lb) for lb in Label],
                                zero_division=0))
    cm_plot(testLabel_raw, pred_p, acc_p,
            title_suffix="CH_UE only", save_path=os.path.join(RESULTS_DIR, "cm_ch_ue_only.png"))
    results["CH_UE only"] = {'loss': loss_p, 'accuracy': acc_p}

    # ════════════════════════════════════
    #  C. 融合模型（直接加载，不重新训练）
    # ════════════════════════════════════
    print("\n" + "="*60)
    print("  [Model C] 加载已有融合模型（不修改）")
    print("="*60)
    model_fusion = load_model(FUSION_MODEL_PATH)
    loss_f, acc_f = model_fusion.evaluate([testCH_BE, testCH_UE], testLabel, verbose=0)
    pred_f = np.argmax(model_fusion.predict([testCH_BE, testCH_UE], verbose=0), axis=1)
    print(f"\n  Fusion     Loss: {loss_f:.6f}  Accuracy: {acc_f*100:.3f}%")
    print(classification_report(testLabel_raw, pred_f,
                                target_names=[str(lb) for lb in Label],
                                zero_division=0))
    cm_plot(testLabel_raw, pred_f, acc_f,
            title_suffix="CH_BE + CH_UE Fusion", save_path=os.path.join(RESULTS_DIR, "cm_fusion.png"))
    results["Fusion"] = {'loss': loss_f, 'accuracy': acc_f}

    # ── 汇总打印 ──
    print("\n" + "="*60)
    print(f"  {'模式':<20}  {'Loss':>10}  {'Accuracy':>10}")
    print(f"  {'-'*44}")
    for name, r in results.items():
        print(f"  {name:<20}  {r['loss']:>10.6f}  {r['accuracy']*100:>9.3f}%")
    print("="*60)

    with open(os.path.join(RESULTS_DIR, "channel_accuracy_summary.txt"), "w", encoding="utf-8") as f:
        f.write("模式,Loss,Accuracy(%)\n")
        for name, r in results.items():
            f.write(f"{name},{r['loss']:.6f},{r['accuracy']*100:.3f}\n")
    print(f"\n汇总已保存 → {RESULTS_DIR}/channel_accuracy_summary.txt")

    plot_accuracy_bar(results, save_path=os.path.join(RESULTS_DIR, "accuracy_comparison.png"))


if __name__ == "__main__":
    main()