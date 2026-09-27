import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Convolution1D, MaxPooling1D, Flatten, Input, concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# ---------------- 1. 基础参数与物理映射 ---------------- #
TrainSampleNumber = 70
TestSampleNumber = 30
ClassNumber = 21

TENG_SampleLength = 3000
TENG_Path = "locat/"
PMUT_Path = "sense/"

# 1-9 类 PLA 蛋的物理属性 (Weight, Size_mm)
PROPERTY_MAP = {
    0: (55.0, 50.0),  # Class 1
    1: (105.0, 90.0),  # Class 2
    2: (20.0, 70.0),  # Class 3
    3: (55.0, 70.0),  # Class 4
    4: (35.0, 70.0),  # Class 5 (未知盲测目标)
    5: (8.0, 50.0),  # Class 6
    6: (175.0, 90.0),  # Class 7
    7: (35.0, 50.0),  # Class 8
    8: (55.0, 90.0)  # Class 9
}

PLA_CLASS_COUNT = len(PROPERTY_MAP)  # 这里是 9 (Class 1-9)
UNKNOWN_EGG_CLASS = 5  # 1-based：盲测目标是第几类（例如 Class 5）
UNKNOWN_EGG_IDX = UNKNOWN_EGG_CLASS - 1  # 0-based
TRAIN_INDICES = [i for i in range(PLA_CLASS_COUNT) if i != UNKNOWN_EGG_IDX]
STANDARDIZE_PER_SAMPLE = True  # 每条样本做 z-score，减少幅值漂移/传感器增益影响


# ---------------- 2. 数据加载函数 ---------------- #
def sample(array1):
    return array1[:, 0:TrainSampleNumber], array1[:, TrainSampleNumber:TrainSampleNumber + TestSampleNumber], array1[:,
                                                                                                              TrainSampleNumber + TestSampleNumber:]


def Load_data(SampleLength, path):
    trainCol, testCol = [], []
    train_idx, test_idx = [], []
    for i in range(ClassNumber):
        file_path = path + str(i + 1) + ".csv"
        if not os.path.exists(file_path): continue
        Array = np.array(pd.read_csv(file_path, header=None))[:, 0:]
        a_train, a_test, _ = sample(Array)
        trainCol.append(a_train);
        testCol.append(a_test)
        train_idx.extend([i] * a_train.shape[1])
        test_idx.extend([i] * a_test.shape[1])

    def process(col):
        res = col[0]
        for item in col[1:]: res = np.concatenate((res, item), axis=1)
        return res.T.reshape(-1, SampleLength, 1)

    return (
        process(trainCol),
        process(testCol),
        np.asarray(train_idx, dtype=int),
        np.asarray(test_idx, dtype=int),
    )


# ---------------- 3. 物理归一化处理 (关键修改) ---------------- #
def _standardize_per_sample(x, eps=1e-8):
    mean = np.mean(x, axis=1, keepdims=True)
    std = np.std(x, axis=1, keepdims=True)
    std = np.maximum(std, eps)
    return (x - mean) / std


def get_physics_normalized_data(teng, pmut, raw_idx, target_indices):
    missing = set(target_indices) - set(PROPERTY_MAP.keys())
    if missing:
        raise ValueError(
            f"target_indices 里包含 PROPERTY_MAP 没有的类: {sorted(missing)}; "
            f"PROPERTY_MAP keys={sorted(PROPERTY_MAP.keys())}"
        )

    mask = np.isin(raw_idx, target_indices)
    x_teng = np.copy(teng[mask])
    x_pmut = np.copy(pmut[mask])

    # 提取每个样本对应的尺寸
    current_indices = np.array(raw_idx)[mask]

    for i, idx in enumerate(current_indices):
        size = PROPERTY_MAP[idx][1]
        # 物理逻辑：信号强度 S 与 接触面积 A (d^2) 成正比
        # 我们以 70mm 为基准进行归一化：S_norm = S / (size/70)^2
        norm_factor = (size / 70.0) ** 2
        x_teng[i] = x_teng[i] / norm_factor
        x_pmut[i] = x_pmut[i] / norm_factor

    if STANDARDIZE_PER_SAMPLE:
        x_teng = _standardize_per_sample(x_teng)
        x_pmut = _standardize_per_sample(x_pmut)

    y_weight = np.array([PROPERTY_MAP[i][0] for i in current_indices]).reshape(-1, 1)
    return x_teng, x_pmut, y_weight


# 加载原始数据
print("正在加载全量原始信号...")
raw_trainTENG, raw_testTENG, raw_train_idx, raw_test_idx = Load_data(TENG_SampleLength, TENG_Path)
raw_trainPMUT, raw_testPMUT, raw_train_idx_p, raw_test_idx_p = Load_data(TENG_SampleLength, PMUT_Path)
if not np.array_equal(raw_train_idx, raw_train_idx_p) or not np.array_equal(raw_test_idx, raw_test_idx_p):
    raise RuntimeError("locat 与 sense 的样本顺序/数量不一致，无法做融合训练。请检查 CSV 列数是否一致。")

# 执行物理归一化：消除尺寸带来的面积增益
print("执行物理归一化处理（消除尺寸对幅值的影响）...")
trT, trP, trW = get_physics_normalized_data(raw_trainTENG, raw_trainPMUT, raw_train_idx, TRAIN_INDICES)
teT, teP, teW = get_physics_normalized_data(raw_testTENG, raw_testPMUT, raw_test_idx, TRAIN_INDICES)
unT, unP, unW = get_physics_normalized_data(raw_testTENG, raw_testPMUT, raw_test_idx, [UNKNOWN_EGG_IDX])


# ---------------- 4. 构建回归模型 ---------------- #
def build_model():
    def branch(length, name):
        inputs = Input(shape=(length, 1), name=name)
        x = Convolution1D(16, 7, activation='relu', padding='same')(inputs)
        x = MaxPooling1D(4)(x)
        x = Convolution1D(32, 5, activation='relu', padding='same')(x)
        x = MaxPooling1D(4)(x)
        x = Flatten()(x)
        return inputs, x

    in_t, feat_t = branch(TENG_SampleLength, "TENG_Branch")
    in_p, feat_p = branch(TENG_SampleLength, "PMUT_Branch")

    merged = concatenate([feat_t, feat_p])
    x = Dense(256, activation='relu')(merged)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu')(x)
    out = Dense(1, activation='linear')(x)

    model = Model(inputs=[in_t, in_p], outputs=out)
    model.compile(optimizer=Adam(1e-4), loss='mse', metrics=['mae'])
    return model


model = build_model()

# ---------------- 5. 训练与物理推论 ---------------- #
print("\n训练基于物理归一化后的模型...")
callbacks = [
    EarlyStopping(monitor="val_mae", patience=30, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_mae", factor=0.5, patience=10, min_lr=1e-6, verbose=1),
]
model.fit([trT, trP], trW, epochs=400, batch_size=32, verbose=1, validation_split=0.2, callbacks=callbacks)


def _eval_by_class(tag, x_t, x_p, y, idx):
    preds = model.predict([x_t, x_p], verbose=0).reshape(-1)
    y_true = y.reshape(-1)
    idx = np.asarray(idx, dtype=int).reshape(-1)

    mae_all = float(np.mean(np.abs(preds - y_true)))
    print(f"\n[{tag}] overall MAE: {mae_all:.2f} g")

    for cls in sorted(np.unique(idx).tolist()):
        if cls not in PROPERTY_MAP:
            continue
        m = idx == cls
        mae = float(np.mean(np.abs(preds[m] - y_true[m])))
        true_w = float(PROPERTY_MAP[cls][0])
        print(f"  Class {cls + 1:>2} (true {true_w:>6.1f} g): n={int(m.sum()):>3}  MAE={mae:>6.2f} g")


known_test_idx = raw_test_idx[np.isin(raw_test_idx, TRAIN_INDICES)]
_eval_by_class("Known/Test", teT, teP, teW, known_test_idx)

print("\n" + "=" * 40)
# 盲测预测
preds = model.predict([unT, unP])
avg_pred = np.mean(preds)
true_w = PROPERTY_MAP[UNKNOWN_EGG_IDX][0]

print(f"盲测目标：Class {UNKNOWN_EGG_IDX + 1} (真实重量: {true_w}g, 尺寸: {PROPERTY_MAP[UNKNOWN_EGG_IDX][1]}mm)")
print(f"物理归一化后的预测重量: {avg_pred:.2f} g")
print(f"预测误差: {abs(avg_pred - true_w):.2f} g")

# 绘图展示
plt.figure(figsize=(9, 5))
plt.hist(preds, bins=15, color='darkorange', edgecolor='black', alpha=0.7)
plt.axvline(true_w, color='green', linewidth=2, label=f'Actual Weight: {true_w}g')
plt.axvline(avg_pred, color='red', linestyle='--', label=f'Physics-Aware Prediction: {avg_pred:.2f}g')
plt.title(f"Physics-Normalized Inference: Class {UNKNOWN_EGG_IDX + 1}")
plt.xlabel("Weight (g)")
plt.ylabel("Sample Frequency")
plt.legend()
plt.show()