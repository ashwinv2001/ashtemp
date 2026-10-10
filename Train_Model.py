
import os
import numpy as np
import tensorflow as tf
import pandas as pd
import faulthandler

faulthandler.enable()

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    CSVLogger
)

from DataGenerator import DataGenerator


# ============================================================
# PATHS
# ============================================================

OUTPUT_FOLDER = r"D:\Project\Preprocessed_Data"

FEATURE_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "features"
)

TRAIN_CSV = os.path.join(
    OUTPUT_FOLDER,
    "train_metadata.csv"
)

VALIDATION_CSV = os.path.join(
    OUTPUT_FOLDER,
    "validation_metadata.csv"
)

TEST_CSV = os.path.join(
    OUTPUT_FOLDER,
    "test_metadata.csv"
)

GENRE_FILE = os.path.join(
    OUTPUT_FOLDER,
    "genre_names.npy"
)

MODEL_SAVE_PATH = "best_music_model.h5"


# ============================================================
# LOAD GENRE NAMES
# ============================================================

genre_names = np.load(
    GENRE_FILE,
    allow_pickle=True
)

NUM_CLASSES = len(genre_names)

print("----------------------------------------")
print("Number of Genres :", NUM_CLASSES)
print("----------------------------------------")


# ============================================================
# HYPERPARAMETERS
# ============================================================

BATCH_SIZE = 32

EPOCHS = 35

LEARNING_RATE = 0.0001


# ============================================================
# CREATE DATA GENERATORS
# ============================================================

print("\nCreating data generators...")


# ------------------------------------------------------------
# TRAINING GENERATOR
# ------------------------------------------------------------

train_generator = DataGenerator(

    metadata_csv=TRAIN_CSV,

    feature_folder=FEATURE_FOLDER,

    genre_names=genre_names,

    batch_size=BATCH_SIZE,

    shuffle=True

)


# ------------------------------------------------------------
# VALIDATION GENERATOR
# ------------------------------------------------------------

validation_generator = DataGenerator(

    metadata_csv=VALIDATION_CSV,

    feature_folder=FEATURE_FOLDER,

    genre_names=genre_names,

    batch_size=BATCH_SIZE,

    shuffle=False

)


# ------------------------------------------------------------
# TEST GENERATOR
#
# NOTE:
# Test data is intentionally NOT used during training.
# It will be used later by Evaluate_Model.py.
# ------------------------------------------------------------

test_generator = DataGenerator(

    metadata_csv=TEST_CSV,

    feature_folder=FEATURE_FOLDER,

    genre_names=genre_names,

    batch_size=BATCH_SIZE,

    shuffle=False

)


print("\nData generators created successfully.")


# ============================================================
# INPUT SHAPE
# ============================================================

INPUT_SHAPE = train_generator.get_input_shape()

print("\nInput Shape :", INPUT_SHAPE)

print(
    "Training Samples   :",
    train_generator.get_dataset_size()
)

print(
    "Validation Samples :",
    validation_generator.get_dataset_size()
)

print(
    "Testing Samples    :",
    test_generator.get_dataset_size()
)

print("----------------------------------------")


# ============================================================
# MODEL INPUT
# ============================================================

inputs = tf.keras.layers.Input(
    shape=INPUT_SHAPE
)


# ============================================================
# CNN FEATURE EXTRACTOR
# ============================================================

x = tf.keras.layers.Conv2D(
    filters=32,
    kernel_size=(3, 3),
    padding="same",
    activation="relu"
)(inputs)

x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2)
)(x)

x = tf.keras.layers.Dropout(
    0.30
)(x)


# ------------------------------------------------------------

x = tf.keras.layers.Conv2D(
    filters=64,
    kernel_size=(3, 3),
    padding="same",
    activation="relu"
)(x)

x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2)
)(x)

x = tf.keras.layers.Dropout(
    0.30
)(x)


# ------------------------------------------------------------

x = tf.keras.layers.Conv2D(
    filters=128,
    kernel_size=(3, 3),
    padding="same",
    activation="relu"
)(x)

x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2)
)(x)

x = tf.keras.layers.Dropout(
    0.30
)(x)


# ============================================================
# CNN OUTPUT SHAPE
# ============================================================

print(
    "CNN Output Shape :",
    x.shape
)


# ============================================================
# RESHAPE FOR BiLSTM
# ============================================================

x = tf.keras.layers.TimeDistributed(

    tf.keras.layers.GlobalAveragePooling1D()

)(x)


print(
    "Sequence Shape :",
    x.shape
)


# ============================================================
# BIDIRECTIONAL LSTM
# ============================================================

x = tf.keras.layers.Bidirectional(

    tf.keras.layers.LSTM(

        units=64,

        return_sequences=True,

        dropout=0.30

    )

)(x)


x = tf.keras.layers.Bidirectional(

    tf.keras.layers.LSTM(

        units=32,

        return_sequences=True,

        dropout=0.30

    )

)(x)


print(
    "BiLSTM Output Shape :",
    x.shape
)


# ============================================================
# SELF ATTENTION
# ============================================================

attention = tf.keras.layers.Attention()(
    [x, x]
)


print(
    "Attention Output Shape :",
    attention.shape
)


# ============================================================
# GLOBAL AVERAGE POOLING
# ============================================================

x = tf.keras.layers.GlobalAveragePooling1D()(
    attention
)


print(
    "Global Pool Output Shape :",
    x.shape
)


# ============================================================
# DROPOUT
# ============================================================

x = tf.keras.layers.Dropout(
    0.50
)(x)


print(
    "Feature Vector Shape :",
    x.shape
)


# ============================================================
# FULLY CONNECTED CLASSIFIER
# ============================================================

x = tf.keras.layers.Dense(

    256,

    activation="relu"

)(x)


x = tf.keras.layers.BatchNormalization()(x)


x = tf.keras.layers.Dropout(
    0.50
)(x)


# ------------------------------------------------------------

x = tf.keras.layers.Dense(

    128,

    activation="relu"

)(x)


x = tf.keras.layers.BatchNormalization()(x)


x = tf.keras.layers.Dropout(
    0.40
)(x)


# ============================================================
# OUTPUT LAYER
# ============================================================

outputs = tf.keras.layers.Dense(

    NUM_CLASSES,

    activation="sigmoid",

    name="genre_predictions"

)(x)


# ============================================================
# CREATE MODEL
# ============================================================

model = tf.keras.Model(

    inputs=inputs,

    outputs=outputs,

    name="CNN_BiLSTM_Attention"

)


# ============================================================
# COMPILE MODEL
# ============================================================

optimizer = tf.keras.optimizers.Adam(

    learning_rate=LEARNING_RATE

)


model.compile(

    optimizer=optimizer,

    loss="binary_crossentropy",

    metrics=[

        tf.keras.metrics.BinaryAccuracy(
            name="binary_accuracy"
        ),

        tf.keras.metrics.Precision(
            name="precision"
        ),

        tf.keras.metrics.Recall(
            name="recall"
        ),

        tf.keras.metrics.AUC(
            name="auc",
            multi_label=True
        )

    ]

)


# ============================================================
# CALLBACKS
# ============================================================

early_stop = EarlyStopping(

    monitor="val_loss",

    patience=5,

    restore_best_weights=True,

    verbose=1

)


checkpoint = ModelCheckpoint(

    MODEL_SAVE_PATH,

    monitor="val_loss",

    save_best_only=True,

    verbose=1

)


reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=3,

    min_lr=1e-6,

    verbose=1

)


csv_logger = CSVLogger(

    "training_history.csv"

)


callbacks = [

    early_stop,

    checkpoint,

    reduce_lr,

    csv_logger

]


# ============================================================
# MODEL SUMMARY
# ============================================================

print("\n===================================")
print("MODEL SUMMARY")
print("===================================\n")

model.summary()


print("\n===================================")
print("Model Ready For Training")
print("===================================\n")


# ============================================================
# TRAIN MODEL
# ============================================================

print("\n========================================")
print("Training Started...")
print("========================================\n")


history = model.fit(

    train_generator,

    validation_data=validation_generator,

    epochs=EPOCHS,

    callbacks=callbacks,

    verbose=1

)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

FINAL_MODEL_PATH = "final_music_model.h5"

model.save(

    FINAL_MODEL_PATH

)


print(
    "\nFinal model saved to :",
    FINAL_MODEL_PATH
)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(
    history.history
)


history_df.to_csv(

    "training_history.csv",

    index=False

)


print(
    "Training history saved."
)


# ============================================================
# PLOT TRAINING CURVES
# ============================================================

import matplotlib.pyplot as plt


# ============================================================
# LOSS
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.plot(

    history.history["loss"],

    label="Training Loss"

)

plt.plot(

    history.history["val_loss"],

    label="Validation Loss"

)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Loss Curve")

plt.legend()

plt.grid(True)

plt.savefig(

    "loss_curve.png",

    dpi=300,

    bbox_inches="tight"

)

plt.show()


# ============================================================
# BINARY ACCURACY
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.plot(

    history.history["binary_accuracy"],

    label="Training Accuracy"

)

plt.plot(

    history.history["val_binary_accuracy"],

    label="Validation Accuracy"

)

plt.xlabel("Epoch")

plt.ylabel("Binary Accuracy")

plt.title("Accuracy Curve")

plt.legend()

plt.grid(True)

plt.savefig(

    "accuracy_curve.png",

    dpi=300,

    bbox_inches="tight"

)

plt.show()


# ============================================================
# PRECISION
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.plot(

    history.history["precision"],

    label="Training Precision"

)

plt.plot(

    history.history["val_precision"],

    label="Validation Precision"

)

plt.xlabel("Epoch")

plt.ylabel("Precision")

plt.title("Precision Curve")

plt.legend()

plt.grid(True)

plt.savefig(

    "precision_curve.png",

    dpi=300,

    bbox_inches="tight"

)

plt.show()


# ============================================================
# RECALL
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.plot(

    history.history["recall"],

    label="Training Recall"

)

plt.plot(

    history.history["val_recall"],

    label="Validation Recall"

)

plt.xlabel("Epoch")

plt.ylabel("Recall")

plt.title("Recall Curve")

plt.legend()

plt.grid(True)

plt.savefig(

    "recall_curve.png",

    dpi=300,

    bbox_inches="tight"

)

plt.show()


# ============================================================
# AUC
# ============================================================

if "auc" in history.history:

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(

        history.history["auc"],

        label="Training AUC"

    )

    plt.plot(

        history.history["val_auc"],

        label="Validation AUC"

    )

    plt.xlabel("Epoch")

    plt.ylabel("AUC")

    plt.title("AUC Curve")

    plt.legend()

    plt.grid(True)

    plt.savefig(

        "auc_curve.png",

        dpi=300,

        bbox_inches="tight"

    )

    plt.show()


# ============================================================
# TRAINING COMPLETE
# ============================================================

print("\n========================================")
print("Training Completed Successfully")
print("========================================")

print("\nImportant files:")

print("- best_music_model.h5")
print("- final_music_model.h5")
print("- training_history.csv")
print("- loss_curve.png")
print("- accuracy_curve.png")
print("- precision_curve.png")
print("- recall_curve.png")
print("- auc_curve.png")

print("\nIMPORTANT:")
print("Test data was NOT used for model training or")
print("model selection.")

print("\nNext step:")
print("Run Threshold_Optimization.py using")
print("validation_metadata.csv ONLY.")

