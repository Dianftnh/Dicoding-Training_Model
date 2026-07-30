"""
modelling.py
Basic: MLflow autolog untuk BiLSTM klasifikasi sentimen IMDB Reviews.
"""

import mlflow
import mlflow.tensorflow
import numpy as np
import pickle
import os
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

DATA_DIR = os.path.join(os.path.dirname(__file__), "dataset_preprocessing")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model_output")
os.makedirs(MODEL_DIR, exist_ok=True)

MAX_VOCAB_SIZE = 20000
MAX_SEQUENCE_LENGTH = 200
EMBEDDING_DIM = 128
BATCH_SIZE = 64
EPOCHS = 20

X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
X_val   = np.load(os.path.join(DATA_DIR, "X_val.npy"))
X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
y_val   = np.load(os.path.join(DATA_DIR, "y_val.npy"))
y_test  = np.load(os.path.join(DATA_DIR, "y_test.npy"))
with open(os.path.join(DATA_DIR, "label_encoder.pkl"), "rb") as f:
    encoder = pickle.load(f)

num_classes = len(encoder.classes_)
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print(f"Classes: {encoder.classes_}")

model = Sequential([
    Embedding(MAX_VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH),
    Bidirectional(LSTM(64)),
    Dropout(0.5),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(num_classes, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

callbacks = [
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
    ModelCheckpoint(os.path.join(MODEL_DIR, "best_model.keras"), monitor="val_accuracy", save_best_only=True, verbose=1),
]

mlflow.set_experiment("imdb-review-classification")
mlflow.tensorflow.autolog()

with mlflow.start_run(run_name="bilstm-baseline"):
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val, y_val), callbacks=callbacks, verbose=1)
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy:.4f}, Test Loss: {loss:.4f}")
    model.save(os.path.join(MODEL_DIR, "final_model.keras"))
    print(f"Model saved to {MODEL_DIR}")

print("Training selesai. Cek MLflow UI: mlflow ui")
