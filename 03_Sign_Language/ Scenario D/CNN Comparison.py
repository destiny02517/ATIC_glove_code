import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Convolution1D, MaxPooling1D, Flatten, Input, concatenate, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# ================= 1. 环境与参数配置 =================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

TrainSampleNumber = 70
TestSampleNumber = 30
ValiSampleNumber = 30
ClassNumber = 46  # 共46类 (1-46)

TENG_SampleLength = 2000
TENG_ChannelNumber = 1
TENG_Channel = [0]
TENG_Path = "locat/"  # 请确保文件夹下是 1.csv - 46.csv

PMUT_SampleLength = 2000
PMUT_ChannelNumber = 1
PMUT_Channel = [0]
PMUT_Path = "sense/"

# ================= 2. 定义 21 对高相关性单词 (根据你的映射表更新) =================
# 格式: (单词1, 单词2, 文件ID1, 文件ID2)
# 注意: 代码中读取时会自动减1作为索引 (文件ID 1 -> Index 0)
PAIRS_INFO = [
    ("you", "me", 1, 2),
    ("my", "book", 4,11),
    ("you", "want", 1, 15),
    ("me", "want", 2, 15),
    ("father", "what", 3, 8),
    ("father", "train", 3, 17),
    ("my", "know", 4, 14),
    ("what", "train", 8, 17),
    ("who", "red", 9, 26),
    ("like", "sick", 12, 22),
    ("want", "clear", 15, 18),
    ("clear", "disk", 18, 31),
    ("clear", "goodbye", 18, 33),
    ("must", "disbelieve", 19, 25),
    ("must", "disk", 19, 31),
    ("tell", "disk", 20, 31),
    ("many", "goodbye", 21, 33),
    ("many", "now", 21, 37),
    ("many", "L", 21, 43),
    ("now", "L", 37, 43),
    ("a", "L", 41, 43),
    ("a", "I", 41, 44),
    ("L", "I", 43, 44)
]


# ================= 3. 数据处理函数 =================
def sample(array1):
    # 分割训练、测试、验证集
    return array1[:, 0:TrainSampleNumber], \
        array1[:, TrainSampleNumber:TrainSampleNumber + TestSampleNumber], \
        array1[:, TrainSampleNumber + TestSampleNumber:]


def Load_data(SampleLength, ChannelNumber, channel, path):
    trainCol, testCol, valiCol = [], [], []
    print(f"Loading data from {path} (1.csv to {ClassNumber}.csv)...")

    for i in range(ClassNumber):
        # 读取 1.csv 到 46.csv
        file_path = f"{path}{i + 1}.csv"
        try:
            df = pd.read_csv(file_path, header=None)
            Array = np.array(df)[:, 0:]

            # 数据集切分
            Array_train, Array_test, Array_Vali = sample(Array)

            trainCol.append(Array_train)
            testCol.append(Array_test)
            valiCol.append(Array_Vali)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None, None, None

    # 拼接与重塑
    def process_set(data_list):
        # 横向拼接所有类别的样本
        data = data_list[0]
        for item in data_list[1:]:
            data = np.concatenate((data, item), axis=1)
        # 转置并重塑为 (样本数, 长度, 通道数)
        data = data.T.reshape(-1, SampleLength, ChannelNumber)
        # 提取指定通道
        data_cut = []
        for d in data:
            data_cut.append(d.T[channel].T)
        return np.array(data_cut)

    return process_set(trainCol), process_set(testCol), process_set(valiCol)


# ================= 4. 模型构建 (Backbone + Classifier) =================
def create_backbone(input_shape):
    """特征提取骨干网络"""
    inputs = Input(shape=input_shape)
    x = inputs
    # 5层卷积提取特征
    for filters in [8, 16, 32, 64, 128]:
        x = Convolution1D(filters=filters, kernel_size=5, padding='same')(x)
        x = Activation("relu")(x)
        x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)

    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(1024)(x)
    x = Activation("relu")(x)
    model = Model(inputs, x)
    return model


def build_model(mode='fusion'):
    """
    mode: 'teng', 'pmut', or 'fusion'
    构建三个结构相似的模型以保证对比公平性
    """
    # 定义输入
    input_t = Input(shape=(TENG_SampleLength, TENG_ChannelNumber), name='input_teng')
    input_p = Input(shape=(PMUT_SampleLength, PMUT_ChannelNumber), name='input_pmut')

    # 骨干网络 (共享结构定义，但不共享权重，每个模型独立训练)
    backbone_t = create_backbone((TENG_SampleLength, TENG_ChannelNumber))
    backbone_p = create_backbone((PMUT_SampleLength, PMUT_ChannelNumber))

    # 获取特征
    feat_t = backbone_t(input_t)
    feat_p = backbone_p(input_p)

    if mode == 'teng':
        x = feat_t
        inputs = input_t
    elif mode == 'pmut':
        x = feat_p
        inputs = input_p
    else:  # fusion
        x = concatenate([feat_t, feat_p])
        inputs = [input_t, input_p]

    # 统一分类头
    x = Dense(1024)(x)
    x = Activation("relu")(x)
    x = Dense(ClassNumber)(x)
    outputs = Activation('softmax')(x)

    model = Model(inputs=inputs, outputs=outputs, name=f'Model_{mode}')
    return model


# ================= 5. 主程序 =================
# 1. 准备标签
trainLabel, testLabel, valiLabel = [], [], []
for j in range(ClassNumber):
    trainLabel.extend([j] * TrainSampleNumber)
    testLabel.extend([j] * TestSampleNumber)
    valiLabel.extend([j] * ValiSampleNumber)

# 转 One-hot
trainLabel = to_categorical(np.array(trainLabel), ClassNumber)
testLabel = to_categorical(np.array(testLabel), ClassNumber)
valiLabel = to_categorical(np.array(valiLabel), ClassNumber)

# 2. 加载数据
print("\n>>> 1. Loading Data...")
trainT, testT, valiT = Load_data(TENG_SampleLength, TENG_ChannelNumber, TENG_Channel, TENG_Path)
trainP, testP, valiP = Load_data(PMUT_SampleLength, PMUT_ChannelNumber, PMUT_Channel, PMUT_Path)

# 3. 训练三个模型
EPOCHS = 40
BATCH_SIZE = 64
models = {}
history_records = {}

modes = ['teng', 'pmut', 'fusion']

print("\n>>> 2. Training Models (Single vs Fusion)...")
for mode in modes:
    print(f"   Training {mode.upper()} model...")
    model = build_model(mode)
    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    # 准备对应数据
    if mode == 'teng':
        x_train, x_vali = trainT, valiT
    elif mode == 'pmut':
        x_train, x_vali = trainP, valiP
    else:
        x_train, x_vali = [trainT, trainP], [valiT, valiP]

    hist = model.fit(x_train, trainLabel,
                     epochs=EPOCHS,
                     batch_size=BATCH_SIZE,
                     validation_data=(x_vali, valiLabel),
                     verbose=0)  # verbose=0 静默训练，1显示进度

    models[mode] = model
    history_records[mode] = hist.history['val_accuracy'][-1]
    print(f"   -> {mode.upper()} Final Val Accuracy: {hist.history['val_accuracy'][-1]:.2%}")

# 4. 核心对比：21对单词
print("\n>>> 3. Evaluating on 21 High-Correlation Pairs...")
results_data = []

# 获取测试集的真实标签索引 (0,0...1,1...45,45)
test_label_idx = np.argmax(testLabel, axis=1)

for w1, w2, id1, id2 in PAIRS_INFO:
    # 文件ID 转 索引 (1 -> 0)
    idx1 = id1 - 1
    idx2 = id2 - 1

    # 提取这两个类的索引
    indices = np.where((test_label_idx == idx1) | (test_label_idx == idx2))[0]

    # 提取子集数据
    sub_T = testT[indices]
    sub_P = testP[indices]
    sub_L = testLabel[indices]

    # 评估
    _, acc_t = models['teng'].evaluate(sub_T, sub_L, verbose=0)
    _, acc_p = models['pmut'].evaluate(sub_P, sub_L, verbose=0)
    _, acc_f = models['fusion'].evaluate([sub_T, sub_P], sub_L, verbose=0)

    pair_label = f"{w1}\nvs\n{w2}"
    results_data.append({
        "Pair": pair_label,
        "TENG": acc_t,
        "PMUT": acc_p,
        "Fusion": acc_f
    })

    print(f"[{w1} vs {w2}]: TENG={acc_t:.1%} | PMUT={acc_p:.1%} | Fusion={acc_f:.1%}")

# ================= 6. 绘图 =================
df = pd.DataFrame(results_data)

plt.figure(figsize=(18, 9))
# 设置字体
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

x = np.arange(len(df))
width = 0.25

# 绘制柱状图
# 配色: Nature期刊风格 (红/蓝/绿)
b1 = plt.bar(x - width, df["TENG"], width, label='Channel 1 (TENG)', color='#F1948A', edgecolor='black', alpha=0.9)
b2 = plt.bar(x, df["PMUT"], width, label='Channel 2 (PMUT)', color='#A2D9CE', edgecolor='black', alpha=0.9)
b3 = plt.bar(x + width, df["Fusion"], width, label='Dual-Channel Fusion', color='#D7BDE2', edgecolor='black', alpha=0.9)

# 装饰
plt.ylabel('Recognition Accuracy', fontsize=14, fontweight='bold')
plt.title('Comparison of Single-Channel vs. Fusion Accuracy on Confusing Word Pairs', fontsize=16, fontweight='bold',
          pad=20)
plt.xticks(x, df["Pair"], rotation=0, fontsize=10)
plt.yticks(np.arange(0, 1.1, 0.1), [f"{v:.0%}" for v in np.arange(0, 1.1, 0.1)])
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.legend(loc='lower right', fontsize=12, frameon=True, shadow=True)

# 在Fusion柱子上标注数值
for rect in b3:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height + 0.01, f'{height:.1%}',
             ha='center', va='bottom', fontsize=8, rotation=90)

plt.tight_layout()
plt.savefig("Confusing_Pairs_Accuracy_Comparison.png", dpi=300)
plt.savefig("Confusing_Pairs_Accuracy_Comparison.svg", format='svg')
print("\n>>> Chart saved as 'Confusing_Pairs_Accuracy_Comparison.png'")
plt.show()