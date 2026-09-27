import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = " "
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from nnom import * # type: ignore
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense,Activation,Convolution1D,MaxPooling1D,Flatten, Input, concatenate,Dropout
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.models import load_model
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE



def cm_plot(original_label, predict_label, Accuracy, pic=None):
    cm = confusion_matrix(original_label, predict_label)   # 由原标签和预测标签生成混淆矩阵
    # plt.figure()
    plt.matshow(cm.T, cmap=plt.cm.Greens)     # 画混淆矩阵，配色风格使用cm.Blues
    plt.colorbar()    # 颜色标签
    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(str(round(cm[x, y]/TestSampleNumber*100,2))+'%'+'\n'+str(cm[x,y]), xy=(x, y), horizontalalignment='center',
                         verticalalignment='center', color = 'white' if cm[x,y]/TestSampleNumber > 0.5 else 'black',fontsize = 6)
    OriginalTick = []
    for i in range(ClassNumber):
        OriginalTick.append(i)
    plt.xticks(OriginalTick,Label)
    plt.yticks(OriginalTick,Label)
    plt.ylabel('Predicted label',{'size':16})  # 坐标轴标签
    plt.xlabel('True label',{'size':16})  # 坐标轴标签
    plt.title('Accuracy: '+ str(round(Accuracy*100,3)) + '%', {'size':16})
    plt.tick_params(axis='x', labelsize = 13)
    plt.tick_params(axis='y', labelsize = 13)
    if pic is not None:
        plt.savefig(str(pic) + '.jpg')
    plt.show()

def sample(array1):
    global TrainSampleNumber
    global TestSampleNumber
    global ValiSampleNumber
    # random = array1.T
    # np.random.shuffle(random)
    # random = random.T
    return array1[:,0:TrainSampleNumber],array1[:,TrainSampleNumber:TrainSampleNumber+TestSampleNumber],array1[:,TrainSampleNumber+TestSampleNumber:]
def Load_data (SampleLength, ChannelNumber, channel, path):
    trainCol = []
    testCol = []
    valiCol = []
#这三行代码用于存储加载的训练集、测试集和验证集的数据。
    for i in range(ClassNumber):
        print(f"尝试加载: {path + str(i + 1) + '.csv'}")
#使用 pandas 库读取 CSV 文件，并将数据转换为 NumPy 数组。[:, 0:] 表示选择所有行和所有列。
        Array = np.array(pd.read_csv(path + str(i + 1) + ".csv"))[:, 0:]
#调用 sample 函数将加载的数据分为训练集、测试集和验证集。
        Array_train, Array_test, Array_Vali = sample(Array)
        trainCol.append(Array_train), testCol.append(Array_test), valiCol.append(Array_Vali)
#append:在处理动态数据时将元素逐个添加到列表中。
#将每个类别的数据添加到对应的列表中。
    trainSet = trainCol[0]
    testSet = testCol[0]
    valiSet = valiCol[0]
    for i in trainCol[1:]:
        trainSet = np.concatenate((trainSet, i), axis=1)

    for i in testCol[1:]:
        testSet = np.concatenate((testSet, i), axis=1)
    for i in valiCol[1:]:
        valiSet = np.concatenate((valiSet, i), axis=1)
#trainCol 是一个列表，包含每个类别的训练样本矩阵，每个矩阵的形状为 (样本数量, 特征数量)。
#trainSet 是一个合并后的矩阵，形状为 (总样本数量, 特征数量)，它整合了所有类别的训练数据。
    trainSet = trainSet.T
    trainSet = trainSet.reshape(-1, SampleLength, ChannelNumber)
#-1 表示自动计算该维度的大小以保持总元素数量不变。
#SampleLength 是每个样本的长度（在此上下文中通常表示时间步数或特征的数量）。
# ChannelNumber:这是特征的通道数量。在这里，如果数据是单通道（如灰度图像或单一传感器的数据），则 ChannelNumber 通常是 1。
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

    return trainSet,testSet,valiSet

def creat_TENG_cnn ():

    input_shape = (TENG_SampleLength, TENG_ChannelNumber)

    inputs = Input(shape=input_shape)

    x = inputs
    # x = Convolution1D(filters=16,kernel_size=5,padding='same')(x)
    # x = Activation("relu")(x)
    # x = MaxPooling1D(pool_size=2,strides=2,padding='same')(x)
    x = Convolution1D(filters=2,kernel_size=2,padding='same')(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(pool_size=2,strides=2,padding='same')(x)
    x = Convolution1D(filters=3,kernel_size=2,padding='same')(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(pool_size=2,strides=2,padding='same')(x)
    # x = Convolution1D(filters=128,kernel_size=5,padding='same')(x)
    # x = Activation("relu")(x)
    # x = MaxPooling1D(pool_size=2,strides=2,padding='same')(x)
    # x = Convolution1D(filters=256,kernel_size=5,padding='same')(x)
    # x = Activation("relu")(x)
    # x = MaxPooling1D(pool_size=2,strides=2,padding='same')(x)
    # x = Convolution1D(filters=512,kernel_size=5,padding='same')(x)
    # x = Activation("relu")(x)
    # x = MaxPooling1D(pool_size=2,strides=2,padding='same')(x)
    x = Dropout(0.4)(x)

    x = Flatten()(x)
    # x = Dense(3000)(x)
    x = Activation("relu")(x)

    model = Model(inputs,x)

    # model.add(Flatten())
    # model.add(Dense(1000, activation='relu'))
    # # Fully connected layer 2 to shape (10) for 10 classes
    # model.add(Dense(10))
    # model.add(Activation('softmax'))

    return model

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

Label = ['1', '2', '3', '4', '5', '6', '7', '8', '9'
         ]

# Label = ['1','2','3','4','5']

# Label = ['1','2','3','4']

TrainSampleNumber = 70
TestSampleNumber = 30
ValiSampleNumber = 30
ClassNumber = 9
TENG_SampleLength = 2999
TENG_ChannelNumber = 1
TENG_Channel = [0]
TENG_Path = "PLA sense float to int/"

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

testLabel1 = testLabel

trainLabel =  to_categorical(np.array(trainLabel), ClassNumber)
testLabel =  to_categorical(np.array(testLabel), ClassNumber)
valiLabel =  to_categorical(np.array(valiLabel), ClassNumber)

trainTENG, testTENG, valiTENG = Load_data(TENG_SampleLength,TENG_ChannelNumber,TENG_Channel,TENG_Path)

TENG_cnn = creat_TENG_cnn()

x = TENG_cnn.output
x = Dense(ClassNumber)(x)


outputs = layers.Softmax()(x)  # 添加 Softmax 层

model = Model(inputs=[TENG_cnn.input], outputs=outputs)

adam = Adam(learning_rate=1e-4)



# model compile process
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics = ['accuracy'])
reduce_lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=30, verbose=1)
plot_model(model, to_file='model_test.png',show_shapes=True)
checkpoint = tf.keras.callbacks.ModelCheckpoint("CNN/Sense.keras", monitor='val_accuracy', verbose=1, save_best_only=True,mode='max')
print(model.summary())
# fit and training process

print('Training ----------------------')
model.fit([trainTENG],trainLabel,epochs=400,batch_size=64,callbacks=[reduce_lr_on_plateau,checkpoint],validation_data=([valiTENG],valiLabel))
# ===================== 绘制训练准确率与损失函数曲线（简洁配色） =====================
history = model.history

plt.figure(figsize=(12, 5))

# 绘制损失函数
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], color='#800080', label='Train Loss')           # 紫色
plt.plot(history.history['val_loss'], color='#0000CD', label='Validation Loss')  # 蓝色
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(False)  # 关闭网格线

# 绘制准确率
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], color='#800080', label='Train Accuracy')           # 紫色
plt.plot(history.history['val_accuracy'], color='#0000CD', label='Validation Accuracy')  # 蓝色
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(False)  # 关闭网格线

plt.tight_layout()
plt.savefig("CNN/loss_accuracy_curve_clean.png")  # 保存图像
plt.show()


# # 训练完成
# print('\nTesting----------------------')
# CNN_model = load_model("CNN/Sense.keras")
# PIE_loss, PIE_accuracy = CNN_model.evaluate([testTENG], testLabel)
# Predict_label = CNN_model.predict([testTENG])
# Predictlabel = np.argmax(Predict_label, axis=1)
# print('\nPIE test loss:', PIE_loss)
# print('\nPIE test accuracy', PIE_accuracy)
# print(Predictlabel)
# cm_plot(testLabel1, Predictlabel, PIE_accuracy)
#
# print("正在导出测试集数据到 CSV...")
#
# # 1. 将三维张量 (样本数, 600, 1) 转回二维矩阵 (样本数, 600)
# test_data_flat = testTENG.reshape(testTENG.shape[0], -1)
#
# # 2. 生成列名：Time_0, Time_1, ..., Time_599
# column_names = [f'Time_{i}' for i in range(test_data_flat.shape[1])]
#
# # 3. 创建 DataFrame
# df_test = pd.DataFrame(test_data_flat, columns=column_names)
#
# # 4. 添加真实标签 (True_Label) 和 模型预测的标签 (Predicted_Label)
# # 这样你可以方便地在 Excel 里对比哪些样本分错了
# df_test['True_Label'] = testLabel1
# df_test['Predicted_Label'] = Predictlabel
#
# # 5. 保存文件
# output_filename = "CNN/test_results_export.csv"
# df_test.to_csv(output_filename, index=False)
#
# print(f"导出成功！文件保存至: {output_filename}")
#
#
# # 调用 generate_model 生成 NNoM 兼容的 .h 文件
# print("Generating NNoM model file...")
# DEF_MODEL_H_NAME = "weights.h"  # 定义硬件模型文件的名称
# x_test_sample = testTENG[:100]  # 从测试集中取 100 个样本作为参考
# generate_model(CNN_model, x_test_sample, format='hwc', name=DEF_MODEL_H_NAME)
# print(f"Model file '{DEF_MODEL_H_NAME}' has been generated successfully.")


# ... (前面的训练代码保持不变) ...

# 训练完成开始测试
print('\nTesting----------------------')
CNN_model = load_model("CNN/Sense.keras")
PIE_loss, PIE_accuracy = CNN_model.evaluate([testTENG], testLabel)

# 获取模型预测的概率分布 (样本数, 类别数)
Predict_probs = CNN_model.predict([testTENG])
# 获取预测的类别索引 (0-8)
Predictlabel = np.argmax(Predict_probs, axis=1)

print('\nPIE test accuracy:', PIE_accuracy)

# --- 功能 1: 保存 One-Hot 格式的 accuracy_fusion.txt ---
# 将预测索引转换为 One-Hot 矩阵
# 例如：如果类别是 9，索引 2 会变成 [0, 0, 1, 0, 0, 0, 0, 0, 0]
prediction_one_hot = np.zeros((Predictlabel.size, ClassNumber), dtype=int)
prediction_one_hot[np.arange(Predictlabel.size), Predictlabel] = 1

# 保存为 txt 文件，元素之间用空格分隔
fusion_path = "CNN/accuracy_fusion.txt"
np.savetxt(fusion_path, prediction_one_hot, fmt='%d', delimiter=' ')
print(f"✅ One-Hot 预测结果已保存至: {fusion_path} (供绘图软件使用)")

# --- 功能 2: 绘制并保存混淆矩阵图 ---
# 自动保存混淆矩阵图片
cm_plot(testLabel1, Predictlabel, PIE_accuracy, pic="CNN/confusion_matrix_plot")

# --- 功能 3: 导出嵌入式 C 语言测试头文件 (.h) ---
# 导出前 5 条测试样本，用于在 STM32/MCU 上进行验证
num_verify = 5
h_file_path = "CNN/embedded_test_data.h"
with open(h_file_path, "w") as f:
    f.write("#ifndef __EMBEDDED_TEST_DATA_H\n#define __EMBEDDED_TEST_DATA_H\n\n")
    f.write(f"#define TEST_SAMPLE_COUNT {num_verify}\n")
    f.write(f"#define TEST_SAMPLE_LEN {TENG_SampleLength}\n\n")

    for i in range(num_verify):
        # 提取第 i 个测试样本并展平
        raw_data = testTENG[i].flatten()
        true_lab = testLabel1[i]

        f.write(f"// Sample Index: {i}, True Label: {true_lab}\n")
        f.write(f"int8_t test_sample_{i}[{TENG_SampleLength}] = {{")
        # 将浮点数据转为整数存入数组
        f.write(", ".join(map(str, raw_data.astype(int))))
        f.write("};\n\n")
    f.write("#endif\n")
print(f"✅ 嵌入式验证数据已生成: {h_file_path}")

# --- 功能 4: 导出 CSV 详细对照表 ---
test_data_flat = testTENG.reshape(testTENG.shape[0], -1)
df_test = pd.DataFrame(test_data_flat)
df_test['True_Label'] = testLabel1
df_test['Predicted_Label'] = Predictlabel
csv_path = "CNN/test_comparison.csv"
df_test.to_csv(csv_path, index=False)
print(f"✅ 详细对比表已保存: {csv_path}")

# --- NNoM 权重导出 ---
print("Generating NNoM weights.h...")
generate_model(CNN_model, testTENG[:100], format='hwc', name="weights.h")
print(f"✅ 嵌入式验证数据已生成至: CNN/embedded_test_data.h")

# --- 原有的 CSV 导出保持不变 ---
print("正在导出测试集数据到 CSV...")
test_data_flat = testTENG.reshape(testTENG.shape[0], -1)
column_names = [f'Time_{i}' for i in range(test_data_flat.shape[1])]
df_test = pd.DataFrame(test_data_flat, columns=column_names)
df_test['True_Label'] = testLabel1
df_test['Predicted_Label'] = Predictlabel
output_filename = "CNN/test_results_export.csv"
df_test.to_csv(output_filename, index=False)
print(f"导出成功！文件保存至: {output_filename}")

# 调用 generate_model 生成 NNoM 兼容的 .h 文件
print("Generating NNoM model file...")
DEF_MODEL_H_NAME = "weights.h"
x_test_sample = testTENG[:100]
# 注意：NNoM 导出时需要确保数据范围与硬件输入一致
generate_model(CNN_model, x_test_sample, format='hwc', name=DEF_MODEL_H_NAME)
print(f"Model file '{DEF_MODEL_H_NAME}' has been generated successfully.")