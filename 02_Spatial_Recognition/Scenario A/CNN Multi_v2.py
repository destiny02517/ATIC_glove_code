import os

print("Current working directory:", os.getcwd())
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from tensorflow.keras.utils import to_categorical, plot_model
from keras.models import Model, load_model
from keras.layers import (
    Dense, Activation, Convolution1D, MaxPooling1D,
    Flatten, Input, concatenate, Dropout
)
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt


def sample(array1):
    return (
        array1[:, 0:TrainSampleNumber],
        array1[:, TrainSampleNumber:TrainSampleNumber + TestSampleNumber],
        array1[:, TrainSampleNumber + TestSampleNumber:]
    )


def Load_data(SampleLength, ChannelNumber, channel, path):
    trainCol, testCol, valiCol = [], [], []

    for i in range(ClassNumber):
        print(f"Loading: {path + str(i + 1) + '.csv'}")
        Array = np.array(pd.read_csv(path + str(i + 1) + ".csv"))[:, 0:]
        Array_train, Array_test, Array_Vali = sample(Array)

        trainCol.append(Array_train)
        testCol.append(Array_test)
        valiCol.append(Array_Vali)

    trainSet = trainCol[0]
    testSet = testCol[0]
    valiSet = valiCol[0]

    for i in trainCol[1:]:
        trainSet = np.concatenate((trainSet, i), axis=1)

    for i in testCol[1:]:
        testSet = np.concatenate((testSet, i), axis=1)

    for i in valiCol[1:]:
        valiSet = np.concatenate((valiSet, i), axis=1)

    def cut_and_reshape(X):
        X = X.T.reshape(-1, SampleLength, ChannelNumber)
        X_cut = [x.T[channel].T for x in X]
        return np.array(X_cut)

    return (
        cut_and_reshape(trainSet),
        cut_and_reshape(testSet),
        cut_and_reshape(valiSet)
    )


def creat_TENG_cnn():
    inputs = Input(shape=(TENG_SampleLength, TENG_ChannelNumber))

    x = Convolution1D(32, 5, padding='same')(inputs)
    x = Activation("relu")(x)
    x = MaxPooling1D(2, 2, padding='same')(x)

    x = Convolution1D(64, 5, padding='same')(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(2, 2, padding='same')(x)

    x = Convolution1D(128, 5, padding='same')(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(2, 2, padding='same')(x)

    x = Convolution1D(256, 5, padding='same')(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(2, 2, padding='same')(x)

    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(3000, activation="relu")(x)

    return Model(inputs, x)


def creat_PMUT_cnn():
    inputs = Input(shape=(PMUT_SampleLength, PMUT_ChannelNumber))

    x = Convolution1D(32, 5, padding='same')(inputs)
    x = Activation("relu")(x)
    x = MaxPooling1D(2, 2, padding='same')(x)

    x = Convolution1D(64, 5, padding='same')(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(2, 2, padding='same')(x)

    x = Convolution1D(128, 5, padding='same')(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(2, 2, padding='same')(x)

    x = Convolution1D(256, 5, padding='same')(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(2, 2, padding='same')(x)

    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(1000, activation="relu")(x)

    return Model(inputs, x)


physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs Available:", len(physical_devices))

if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


Label = [
    [1], [2], [3], [4], [5], [6], [7], [8],
    [1, 2], [1, 3], [1, 4], [5, 6],
    [6, 8], [3, 7], [4, 8], [2, 6],
    [2, 3, 4], [3, 4, 6], [4, 7, 8], [3, 4, 5, 7]
]

TrainSampleNumber = 180
TestSampleNumber = 60
ValiSampleNumber = 60
ClassNumber = 20
NUM_POSITIONS = 8

TENG_SampleLength = 1250
TENG_ChannelNumber = 1
TENG_Channel = [0]
TENG_Path = "test/"

PMUT_SampleLength = 1250
PMUT_ChannelNumber = 1
PMUT_Channel = [0]
PMUT_Path = "sense/"


def to_multihot(indices, num_positions=8):
    v = np.zeros(num_positions, dtype=int)

    for k in indices:
        if 1 <= k <= num_positions:
            v[k - 1] = 1

    return v


Label8 = np.array([
    to_multihot(lst, NUM_POSITIONS)
    for lst in Label
])


def build_split_labels(repeat_n):
    parts = [
        np.tile(Label8[i], (repeat_n, 1))
        for i in range(ClassNumber)
    ]
    return np.vstack(parts)


trainLabel = build_split_labels(TrainSampleNumber)
valiLabel = build_split_labels(ValiSampleNumber)
testLabel = build_split_labels(TestSampleNumber)


print("Loading TENG data...")
trainTENG, testTENG, valiTENG = Load_data(
    TENG_SampleLength,
    TENG_ChannelNumber,
    TENG_Channel,
    TENG_Path
)

print("Loading PMUT data...")
trainPMUT, testPMUT, valiPMUT = Load_data(
    PMUT_SampleLength,
    PMUT_ChannelNumber,
    PMUT_Channel,
    PMUT_Path
)


assert trainTENG.shape[0] == trainLabel.shape[0]
assert testPMUT.shape[0] == testLabel.shape[0]


print("Building model...")

TENG_cnn = creat_TENG_cnn()
PMUT_cnn = creat_PMUT_cnn()

combined = concatenate([
    TENG_cnn.output,
    PMUT_cnn.output
])

x = Dense(3000, activation="relu")(combined)
x = Dense(1000, activation="relu")(x)
x = Dense(NUM_POSITIONS, activation='sigmoid')(x)

model = Model(
    inputs=[TENG_cnn.input, PMUT_cnn.input],
    outputs=x
)

adam = Adam(learning_rate=1e-4)

model.compile(
    optimizer=adam,
    loss='binary_crossentropy',
    metrics=[
        tf.keras.metrics.BinaryAccuracy(
            name="binary_accuracy",
            threshold=0.5
        ),
        tf.keras.metrics.Precision(
            name="precision",
            thresholds=0.5
        ),
        tf.keras.metrics.Recall(
            name="recall",
            thresholds=0.5
        ),
        tf.keras.metrics.AUC(
            name="auc"
        )
    ],
)

os.makedirs("CNN", exist_ok=True)

reduce_lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=30,
    verbose=1
)

plot_model(
    model,
    to_file='fusion_model.png',
    show_shapes=True
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "CNN/FusionSense.keras",
    monitor='val_binary_accuracy',
    verbose=1,
    save_best_only=True,
    mode='max'
)

print(model.summary())


print('Training ----------------------')

history = model.fit(
    [trainTENG, trainPMUT],
    trainLabel,
    epochs=100,
    batch_size=64,
    callbacks=[
        reduce_lr_on_plateau,
        checkpoint
    ],
    validation_data=(
        [valiTENG, valiPMUT],
        valiLabel
    )
)


acc = history.history['binary_accuracy']
val_acc = history.history['val_binary_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.figure(figsize=(6, 4))
plt.plot(epochs, acc, label='Training')
plt.plot(epochs, val_acc, label='Validation')
plt.title('Binary Accuracy - Epochs')
plt.xlabel('Epochs')
plt.ylabel('BinAcc')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(epochs, loss, label='Training')
plt.plot(epochs, val_loss, label='Validation')
plt.title('Loss - Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()


print('\nTesting ----------------------')

CNN_model = load_model("CNN/FusionSense.keras")

metrics = CNN_model.evaluate(
    [testTENG, testPMUT],
    testLabel,
    verbose=0
)

metric_names = CNN_model.metrics_names

print({
    k: float(v)
    for k, v in zip(metric_names, metrics)
})


y_prob = CNN_model.predict(
    [testTENG, testPMUT],
    verbose=0
)

y_pred = (y_prob >= 0.5).astype(int)
y_true = testLabel.astype(int)


np.savetxt(
    'accuracy_fusion.txt',
    y_pred,
    fmt='%d'
)

np.savetxt(
    'pred_positions.txt',
    y_pred,
    fmt='%d'
)

print(
    'Saved predictions to accuracy_fusion.txt / '
    'pred_positions.txt; shape:',
    y_pred.shape
)


eps = 1e-9

present = y_true.sum(axis=0)

tp = np.logical_and(
    y_true == 1,
    y_pred == 1
).sum(axis=0)

present_recall = (
    tp / (present + eps)
) * 100.0

col_acc = (
    y_true == y_pred
).mean(axis=0) * 100.0

pos_names = [
    f'pos{i}'
    for i in range(1, NUM_POSITIONS + 1)
]


print("\nPer-position recall:")

for i, name in enumerate(pos_names):
    print(
        f"  {name}: "
        f"{present_recall[i]:.2f}% "
        f"(positive samples: {present[i]})"
    )


print("\nPer-position overall accuracy:")

for i, name in enumerate(pos_names):
    print(
        f"  {name}: "
        f"{col_acc[i]:.2f}%"
    )


print(
    "\nFirst test sample of each original combination class:"
)

for cls in range(ClassNumber):
    idx = cls * TestSampleNumber

    print(
        f"class {cls + 1:02d} "
        f"{Label[cls]} -> "
        f"pred {y_pred[idx].tolist()} | "
        f"true {y_true[idx].tolist()}"
    )