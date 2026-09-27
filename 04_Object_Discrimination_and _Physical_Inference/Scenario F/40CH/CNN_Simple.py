import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = " "
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense,Activation,Convolution1D,MaxPooling1D,Flatten, Input, concatenate,Dropout
from keras.optimizers import adam_v2
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
                         verticalalignment='center',color = 'white' if cm[x,y]/TestSampleNumber > 0.5 else 'black',fontsize = 6)
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

import os
import glob
import numpy as np

def sample(file_list):
    """
    新结构下不再按列切分，而是 file_list 本身就是一个 split 的样本列表。
    为了保持原代码调用不变，这里直接返回 (train_files, test_files, vali_files) 的占位形式。
    实际上在 Load_data() 里会分别读取 train/test/val 三个文件夹。
    """
    return file_list, [], []

def Load_data(SampleLength, ChannelNumber, channel, path):
    """
    新结构：遍历 Dataset 下所有类别子文件夹（不再假设 vofa+X）
    Dataset/<class_folder>/train/*.csv
    Dataset/<class_folder>/val/*.csv
    Dataset/<class_folder>/test/*.csv

    返回：
      trainSet: (Ntrain_total, SampleLength, len(channel))
      testSet : (Ntest_total,  SampleLength, len(channel))
      valiSet : (Nval_total,   SampleLength, len(channel))
    """

    dataset_root = "Dataset"   # 若你的 Dataset 不在当前目录，可改成绝对路径

    # 1) 找 Dataset 下所有“类文件夹”（按名字排序，保证每次顺序一致）
    class_dirs = sorted([
        d for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d)) and (not d.startswith("."))
    ])
    if len(class_dirs) == 0:
        raise FileNotFoundError(f"No class folders found under: {dataset_root}")

    def load_split(split_name):
        X_list = []

        for cls_name in class_dirs:
            cls_dir = os.path.join(dataset_root, cls_name, split_name)
            files = sorted(glob.glob(os.path.join(cls_dir, "*.csv")))
            if len(files) == 0:
                raise FileNotFoundError(f"No CSV found: {cls_dir}")

            for fp in files:
                # 读取带表头的CSV：第一列是时间，后面是 I0~I39
                df = pd.read_csv(fp)

                # 只保留 I0~I39，自动去掉时间列（无论时间列叫 t / time / Timestamp 都无所谓）
                cols = [f"I{i}" for i in range(ChannelNumber)]  # I0..I39
                missing = [c for c in cols if c not in df.columns]
                if missing:
                    raise ValueError(f"{fp} 缺少这些列（示例前几个）：{missing[:8]}")

                arr = df[cols].to_numpy(dtype=np.float32)  # shape: (SampleLength, 40)

                # 检查形状
                if arr.shape != (SampleLength, ChannelNumber):
                    raise ValueError(
                        f"Shape mismatch: {fp} has {arr.shape}, expected ({SampleLength},{ChannelNumber})"
                    )

                # 选择通道（保持你原来的 channel 用法）
                arr = arr[:, channel]

                X_list.append(arr)

        X = np.stack(X_list, axis=0).astype(np.float32)
        return X

    trainSet = load_split("train")
    testSet  = load_split("test")
    valiSet  = load_split("val")

    return trainSet, testSet, valiSet



def creat_PMUT_cnn ():

    input_shape = (PMUT_SampleLength, PMUT_ChannelNumber)

    inputs = Input(shape=input_shape)

    x = inputs
    # x = Convolution1D(filters=16,kernel_size=5,padding='same')(x)
    # x = Activation("relu")(x)
    # x = MaxPooling1D(pool_size=2,strides=2,padding='same')(x)
    x = Convolution1D(filters=32,kernel_size=5,padding='same')(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(pool_size=2,strides=2,padding='same')(x)
    x = Convolution1D(filters=64,kernel_size=5,padding='same')(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(pool_size=2,strides=2,padding='same')(x)
    x = Convolution1D(filters=128,kernel_size=5,padding='same')(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(pool_size=2,strides=2,padding='same')(x)
    x = Convolution1D(filters=256,kernel_size=5,padding='same')(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(pool_size=2,strides=2,padding='same')(x)
    # x = Convolution1D(filters=512,kernel_size=5,padding='same')(x)
    # x = Activation("relu")(x)
    # x = MaxPooling1D(pool_size=2,strides=2,padding='same')(x)
    x = Dropout(0.5)(x)

    x = Flatten()(x)
    x = Dense(3000)(x)
    x = Activation("relu")(x)

    model = Model(inputs,x)
    return model

Label = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']

# channel_name = "Locat"
channel_name = "Sense"

TrainSampleNumber = 70
TestSampleNumber = 20
ValiSampleNumber = 20
ClassNumber = 21

PMUT_SampleLength = 1300
PMUT_ChannelNumber = 40
PMUT_Channel = list(range(40))   # 0~39 全部通道
PMUT_Path = f"{channel_name}/"

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

trainLabel = np_utils.to_categorical(np.array(trainLabel), ClassNumber)
testLabel = np_utils.to_categorical(np.array(testLabel), ClassNumber)
valiLabel = np_utils.to_categorical(np.array(valiLabel), ClassNumber)

# trainTENG, testTENG, valiTENG = Load_data(TENG_SampleLength,TENG_ChannelNumber,TENG_Channel,TENG_Path)
trainPMUT, testPMUT, valiPMUT = Load_data(PMUT_SampleLength,PMUT_ChannelNumber,PMUT_Channel,PMUT_Path)

# TENG_cnn = creat_TENG_cnn()
PMUT_cnn = creat_PMUT_cnn()

# combinedInput = concatenate([TENG_cnn.output, PMUT_cnn.output])

x = PMUT_cnn.output
x = Dense(ClassNumber,activation="softmax")(x)

model = Model(inputs=[PMUT_cnn.input], outputs=x)

adam = adam_v2.Adam(lr=1e-4)

# model compile process
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics = ['accuracy'])
reduce_lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=30, verbose=1)
# plot_model(model, to_file='model_test.png',show_shapes=True)
checkpoint = tf.keras.callbacks.ModelCheckpoint(f"CNN/{channel_name}.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
print(model.summary())
# fit and training process

# print('Training ----------------------')
# model.fit([trainPMUT],trainLabel,epochs=100,batch_size=16,callbacks=[reduce_lr_on_plateau,checkpoint],validation_data=([valiPMUT],valiLabel))

print('Training ----------------------')
history = model.fit([trainPMUT],trainLabel,epochs=150,batch_size=32,callbacks=[reduce_lr_on_plateau,checkpoint],validation_data=([valiPMUT],valiLabel))


# print('\nTesting----------------------')
# CNN_model = load_model(f"CNN/{channel_name}.h5")
# PIE_loss,PIE_accuracy = CNN_model.evaluate([testPMUT],testLabel)
# Predict_label = CNN_model.predict([testPMUT])
# Predictlabel = np.argmax(Predict_label,axis=1)
# print('\nPIE test loss:', PIE_loss)
# print('\nPIE test accuracy', PIE_accuracy)
# print(Predictlabel)
# cm_plot(testLabel1,Predictlabel,PIE_accuracy)

print('\nTesting----------------------')
CNN_model = load_model(f"CNN/{channel_name}.h5")
PIE_loss,PIE_accuracy = CNN_model.evaluate([testPMUT],testLabel)
Predict_label = CNN_model.predict([testPMUT])
Predictlabel = np.argmax(Predict_label,axis=1)
print('\nPIE test loss:', PIE_loss)
print('\nPIE test accuracy', PIE_accuracy)
print(Predictlabel)
cm_plot(testLabel1,Predictlabel,PIE_accuracy)

# plot the training loss and accuracy
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='validation')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

np.savetxt(f"{channel_name}.txt", Predictlabel, fmt='%.0f')