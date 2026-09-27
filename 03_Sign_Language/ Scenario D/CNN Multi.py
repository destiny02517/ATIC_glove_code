import os

print("当前工作目录为：", os.getcwd())
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Activation, Convolution1D, MaxPooling1D, Flatten, Input, concatenate, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# ---------------- 混淆矩阵绘制函数 (参考第一个代码) ---------------- #
def cm_plot(original_label, predict_label, Accuracy, pic=None):
    cm = confusion_matrix(original_label, predict_label)  # 由原标签和预测标签生成混淆矩阵
    plt.figure(figsize=(10, 8))
    plt.matshow(cm.T, cmap=plt.cm.Greens)  # 画混淆矩阵，配色风格使用cm.Greens
    plt.colorbar()  # 颜色标签
    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(str(round(cm[x, y] / TestSampleNumber * 100, 2)) + '%' + '\n' + str(cm[x, y]),
                         xy=(x, y), horizontalalignment='center',
                         verticalalignment='center',
                         color='white' if cm[x, y] / TestSampleNumber > 0.5 else 'black',
                         fontsize=8)
    OriginalTick = []
    for i in range(ClassNumber):
        OriginalTick.append(i)
    plt.xticks(OriginalTick, Label)
    plt.yticks(OriginalTick, Label)
    plt.ylabel('Predicted label', {'size': 16})  # 坐标轴标签
    plt.xlabel('True label', {'size': 16})  # 坐标轴标签
    plt.title('Accuracy: ' + str(round(Accuracy * 100, 3)) + '%', {'size': 16})
    plt.tick_params(axis='x', labelsize=13)
    plt.tick_params(axis='y', labelsize=13)
    if pic is not None:
        plt.savefig(str(pic) + '.jpg', dpi=300, bbox_inches='tight')
    plt.show()


# ---------------- 数据预处理函数 ---------------- #
def sample(array1):
    global TrainSampleNumber
    global TestSampleNumber
    global ValiSampleNumber
    return array1[:, 0:TrainSampleNumber], array1[:, TrainSampleNumber:TrainSampleNumber + TestSampleNumber], array1[:,
                                                                                                              TrainSampleNumber + TestSampleNumber:]


def Load_data(SampleLength, ChannelNumber, channel, path):
    trainCol = []
    testCol = []
    valiCol = []

    for i in range(ClassNumber):
        print(f"尝试加载: {path + str(i + 1) + '.csv'}")
        Array = np.array(pd.read_csv(path + str(i + 1) + ".csv", header=None))[:, 0:]
        Array_train, Array_test, Array_Vali = sample(Array)
        trainCol.append(Array_train), testCol.append(Array_test), valiCol.append(Array_Vali)

    trainSet = trainCol[0]
    testSet = testCol[0]
    valiSet = valiCol[0]

    for i in trainCol[1:]:
        trainSet = np.concatenate((trainSet, i), axis=1)
    for i in testCol[1:]:
        testSet = np.concatenate((testSet, i), axis=1)
    for i in valiCol[1:]:
        valiSet = np.concatenate((valiSet, i), axis=1)

    trainSet = trainSet.T
    trainSet = trainSet.reshape(-1, SampleLength, ChannelNumber)
    trainSet_Cut = []
    for i in trainSet:
        trainSet_Cut.append(i.T[channel].T)
    trainSet = np.array(trainSet_Cut)

    testSet = testSet.T
    testSet = testSet.reshape(-1, SampleLength, ChannelNumber)
    testSet_Cut = []
    for i in testSet:
        testSet_Cut.append(i.T[channel].T)
    testSet = np.array(testSet_Cut)

    valiSet = valiSet.T
    valiSet = valiSet.reshape(-1, SampleLength, ChannelNumber)
    valiSet_Cut = []
    for i in valiSet:
        valiSet_Cut.append(i.T[channel].T)
    valiSet = np.array(valiSet_Cut)

    return trainSet, testSet, valiSet


# ---------------- 网络结构 ---------------- #
def creat_TENG_cnn():
    input_shape = (TENG_SampleLength, TENG_ChannelNumber)
    inputs = Input(shape=input_shape)

    x = inputs
    x = Convolution1D(filters=8, kernel_size=5, padding='same')(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)
    x = Convolution1D(filters=16, kernel_size=5, padding='same')(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)
    x = Convolution1D(filters=32, kernel_size=5, padding='same')(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)
    x = Convolution1D(filters=64, kernel_size=5, padding='same')(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)
    x = Convolution1D(filters=128, kernel_size=5, padding='same')(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)

    x = Dropout(0.5)(x)

    x = Flatten()(x)
    x = Dense(3000)(x)
    x = Activation("relu")(x)

    model = Model(inputs, x)
    return model


def creat_PMUT_cnn():
    input_shape = (PMUT_SampleLength, PMUT_ChannelNumber)
    inputs = Input(shape=input_shape)

    x = inputs
    x = Convolution1D(filters=8, kernel_size=5, padding='same')(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)
    x = Convolution1D(filters=16, kernel_size=5, padding='same')(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)
    x = Convolution1D(filters=32, kernel_size=5, padding='same')(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)
    x = Convolution1D(filters=64, kernel_size=5, padding='same')(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)
    x = Convolution1D(filters=128, kernel_size=5, padding='same')(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)

    x = Dropout(0.5)(x)

    x = Flatten()(x)
    x = Dense(3000)(x)
    x = Activation("relu")(x)

    model = Model(inputs, x)
    return model


# ---------------- GPU设置 ---------------- #
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# ---------------- 参数配置 ---------------- #
Label = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16], [17], [18], [19], [20], [21], [22], [23], [24], [25], [26], [27], [28], [29], [30], [31], [32], [33], [34], [35], [36], [37], [38], [39], [40], [41], [42], [43], [44], [45], [46]]

TrainSampleNumber = 70
TestSampleNumber = 30
ValiSampleNumber = 30
ClassNumber = 46  # 修改为20类

TENG_SampleLength = 2000
TENG_ChannelNumber = 1
TENG_Channel = [0]
TENG_Path = "locat/"

PMUT_SampleLength = 2000
PMUT_ChannelNumber = 1
PMUT_Channel = [0]
PMUT_Path = "sense/"

# ---------------- 标签处理 (参考第一个代码的单分类方式) ---------------- #
trainLabel = []
for j in range(ClassNumber):
    for i in range(TrainSampleNumber):
        trainLabel.append(j)

testLabel = []
for j in range(ClassNumber):
    for i in range(TestSampleNumber):
        testLabel.append(j)

valiLabel = []
for j in range(ClassNumber):
    for i in range(ValiSampleNumber):
        valiLabel.append(j)

testLabel1 = testLabel  # 保存原始标签用于混淆矩阵

# 转换为one-hot编码
trainLabel = to_categorical(np.array(trainLabel), ClassNumber)
testLabel = to_categorical(np.array(testLabel), ClassNumber)
valiLabel = to_categorical(np.array(valiLabel), ClassNumber)

# ---------------- 数据加载 ---------------- #
print("加载TENG数据...")
trainTENG, testTENG, valiTENG = Load_data(TENG_SampleLength, TENG_ChannelNumber, TENG_Channel, TENG_Path)

print("加载PMUT数据...")
trainPMUT, testPMUT, valiPMUT = Load_data(PMUT_SampleLength, PMUT_ChannelNumber, PMUT_Channel, PMUT_Path)

# ---------------- 模型构建 ---------------- #
print("构建模型...")
TENG_cnn = creat_TENG_cnn()
PMUT_cnn = creat_PMUT_cnn()

# 融合两个CNN的输出
combinedInput = concatenate([TENG_cnn.output, PMUT_cnn.output])

x = Dense(3000)(combinedInput)
x = Activation("relu")(x)
x = Dense(3000)(x)
x = Activation("relu")(x)
x = Dense(ClassNumber)(x)
outputs = Activation('softmax')(x)  # 使用softmax进行多分类

model = Model(inputs=[TENG_cnn.input, PMUT_cnn.input], outputs=outputs)

adam = Adam(learning_rate=1e-4)

# 编译模型 (使用categorical_crossentropy进行多分类)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

# 设置回调函数
reduce_lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=30, verbose=1)
plot_model(model, to_file='fusion_model.png', show_shapes=True)
checkpoint = tf.keras.callbacks.ModelCheckpoint("CNN/FusionSense.keras", monitor='val_accuracy', verbose=1,
                                                save_best_only=True, mode='max')

print(model.summary())

# ---------------- 训练过程 ---------------- #
print('Training ----------------------')
history = model.fit(
    [trainTENG, trainPMUT], trainLabel,
    epochs=400,
    batch_size=64,
    callbacks=[reduce_lr_on_plateau, checkpoint],
    validation_data=([valiTENG, valiPMUT], valiLabel)
)

# ---------------- 训练曲线可视化 (参考第一个代码) ---------------- #
import matplotlib.pyplot as plt

# 从 history 中取出数据
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

# 绘制 Accuracy 曲线
plt.figure(figsize=(6, 4))
plt.plot(epochs, acc, color='magenta', label='Training')
plt.plot(epochs, val_acc, color='cyan', label='Validation')
plt.title('Accuracy - Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

# 绘制 Loss 曲线
plt.figure(figsize=(6, 4))
plt.plot(epochs, loss, color='magenta', label='Training')
plt.plot(epochs, val_loss, color='cyan', label='Validation')
plt.title('Loss - Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# ---------------- 测试评估 ---------------- #
print('\nTesting----------------------')
CNN_model = load_model("CNN/FusionSense.keras")
PIE_loss, PIE_accuracy = CNN_model.evaluate([testTENG, testPMUT], testLabel)
Predict_label = CNN_model.predict([testTENG, testPMUT])
Predictlabel = np.argmax(Predict_label, axis=1)
# —— 将预测类别索引转为 one-hot 0/1 矩阵并保存 —— #
pred_onehot = np.zeros((Predictlabel.shape[0], ClassNumber), dtype=int)
pred_onehot[np.arange(Predictlabel.shape[0]), Predictlabel] = 1

# 与前一版脚本保持同名输出：注意这里保存的是“预测标签矩阵”，不是accuracy
np.savetxt('accuracy_fusion.txt', pred_onehot, fmt='%d')

print('Saved one-hot predictions to accuracy_fusion.txt with shape:', pred_onehot.shape)

print('\nFusion test loss:', PIE_loss)
print('\nFusion test accuracy:', PIE_accuracy)
print('Predicted labels:', Predictlabel)

# ---------------- 混淆矩阵可视化 (参考第一个代码) ---------------- #
cm_plot(testLabel1, Predictlabel, PIE_accuracy, pic="fusion_confusion_matrix")

