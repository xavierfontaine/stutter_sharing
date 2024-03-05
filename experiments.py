import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile
from keras.callbacks import EarlyStopping
from keras.layers import (
    GRU,
    LSTM,
    AveragePooling2D,
    BatchNormalization,
    Bidirectional,
    Conv1D,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Normalization,
    ReLU,
    TimeDistributed,
)
from keras.models import Sequential
from keras.optimizers import Adam
from librosa.feature.inverse import mfcc_to_audio
from sklearn.model_selection import train_test_split

# =========
# Constants
# =========
DATA_FOLDER = "new_data"
X_FILENAME = "mfcc_data.pkl"
Y_ALL_FILENAME = "y_stutter_feature.pkl"
Y_FILENAME = "y_two_reviewer.pkl"
NAMES_FILENAME = "clip_name.pkl"
OUTPUT_FOLDER = "models"
OUTPUT_NAME = "experimental_model.tf"
SAVE: bool = False


# =========
# Functions
# ========
def load_from_pickle(folder_name, file_name):
    with open(os.path.join(folder_name, file_name), "rb") as file:
        out = pickle.load(file)
    return out


# ====
# Load
# ====
x: np.ndarray = load_from_pickle(folder_name=DATA_FOLDER, file_name=X_FILENAME)
y_all: np.ndarray = load_from_pickle(folder_name=DATA_FOLDER, file_name=Y_ALL_FILENAME)
y: np.ndarray = load_from_pickle(folder_name=DATA_FOLDER, file_name=Y_FILENAME)
names: np.ndarray = load_from_pickle(folder_name=DATA_FOLDER, file_name=NAMES_FILENAME)

print(f"{x.shape=}, {y_all.shape=}, {y.shape=}, {names.shape=}")


# ========================
# Check everything is fine
# ========================
idx = 3000  # 1000, 3000, 4000
name = names[idx]
y_i = y[idx]
x_i = x[idx]
audio_i = mfcc_to_audio(x_i, sr=8000)
soundfile.write("out.wav", audio_i, 8000, "PCM_24")
print(f"{name[0]=}, {y_i=}")


# ==================
# Model eval helpers
# ==================
def plot_loss_accuracy(history):
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="best")
    plt.show()

    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="best")
    plt.show()


# ============
# Data preproc
# ============
def preprocess_data(y: np.array, x: np.array, y_all: np.array) -> tuple:
    # Transpose x
    x = np.transpose(x, (0, 2, 1))
    # Split
    x_train, x_test, y_train, y_test, y_all_train, y_all_test = train_test_split(
        x, y, y_all, test_size=0.2, random_state=42
    )
    return x_train, x_test, y_train, y_test, y_all_train, y_all_test


# ======
# models
# ======
def init_compile_cnn(X_train: np.ndarray) -> Sequential:
    model = Sequential()

    # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
    model.add(
        Conv2D(
            8,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="same",
            input_shape=(20, 74),
            # use_bias=True,
        )
    )
    model.add(ReLU())
    model.add(BatchNormalization())

    # Second Convolution Block
    model.add(
        Conv2D(
            16,
            kernel_size=(2, 2),
            strides=(2, 2),
            padding="same",
            # use_bias=True,
            activation="relu",
        )
    )
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(2, 2)))

    # Third Convolution Block
    model.add(
        Conv2D(
            32,
            kernel_size=(2, 2),
            strides=(2, 2),
            padding="same",
            # use_bias=True,
            activation="relu",
        )
    )
    # model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(2, 2)))

    # Fourth Convolution Block
    model.add(
        Conv2D(
            64,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="same",
            # use_bias=True,
            activation="relu",
        )
    )
    model.add(AveragePooling2D(pool_size=(2, 2)))
    # model.add(BatchNormalization())

    model.add(Flatten())

    # Linear Classifier
    model.add(Dense(10, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))


# TODO: add es callback


def init_compile_simplest(x_train: np.ndarray) -> Sequential:
    # Adapt normalization layer
    normalizer = Normalization()
    normalizer.adapt(x_train)
    # Composing model
    model = Sequential()
    model.add(normalizer)
    # model.add(Bidirectional(LSTM(128, activation="tanh", return_sequences=True)))
    model.add(Conv1D(filters=64, kernel_size=16))
    # model.add(Bidirectional(LSTM(128, activation="tanh", return_sequences=False)))
    model.add(LSTM(128, activation="tanh", return_sequences=False))
    model.add(Dropout(0.5))
    # model.add(TimeDistributed(Dense(20, activation="relu")))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))
    # Optimizer
    optimizer = Adam(learning_rate=1e-3)
    # Compile
    model.compile(
        optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy", "AUC"]
    )
    # Return
    return model


def train_model(model: Sequential, x: np.ndarray, y: np.ndarray) -> dict:
    """Returns the fit history"""
    # Get callbacks
    es = EarlyStopping(patience=5, restore_best_weights=True, monitor="val_auc")
    callbacks = [es]
    # Fit
    history = model.fit(
        x=x, y=y, validation_split=0.2, epochs=20, batch_size=32, callbacks=callbacks
    )
    return history


# ===========
# Experiments
# ===========
# Split train/test
x_train, x_test, y_train, y_test, y_all_train, y_all_test = preprocess_data(
    y=y, x=x, y_all=y_all
)
# Init and fit
model = init_compile_simplest(x_train=x_train)
model.summary()
history = train_model(model=model, x=x_train, y=y_train)


# ==========
# Evaluation
# ==========
# Predict
y_pred = model.predict(x_test)
# Print accuracies
perfs = model.evaluate(x_test, y_test)
acc = perfs[1]
print(f"Reference accuracy: {max(np.mean(y_test), 1-np.mean(y_test))}")
print(f"Obtained accuracy: {acc}")
# Mean etc
for c in np.unique(y_test):
    print(f"Mean predicted value when y={c}: {np.mean(y_pred[y_test==c])}")
# Recall for each type of stutter
THRESH = 0.5
y_pred_binary = (1 * (y_pred > THRESH))[:, 0]
recalls = np.matmul(y_pred_binary, y_all_test) / np.sum(y_all_test, axis=0)
print(f"Recall for each type of stutter: {recalls}")
# Print train history
plot_loss_accuracy(history=history)


# ====
# Save
# ====
if SAVE:
    out_path = os.path.join(OUTPUT_FOLDER, OUTPUT_NAME)
    model.save(out_path, save_format="tf")
