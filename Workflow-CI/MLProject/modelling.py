"""
modelling.py — MLflow Project untuk re-training otomatis
Menerima hyperparameter via argparse (MLflow Project entry point).
"""

import argparse
import mlflow
import mlflow.tensorflow
import numpy as np
import pickle
import os
import json
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

DATA_DIR = os.path.join(os.path.dirname(__file__), "dataset_preprocessing")

parser = argparse.ArgumentParser()
parser.add_argument("--embedding_dim", type=int, default=128)
parser.add_argument("--lstm_units", type=int, default=64)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--epochs", type=int, default=15)
args = parser.parse_args()

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

# Log params
mlflow.log_param("embedding_dim", args.embedding_dim)
mlflow.log_param("lstm_units", args.lstm_units)
mlflow.log_param("learning_rate", args.learning_rate)
mlflow.log_param("batch_size", args.batch_size)
mlflow.log_param("epochs", args.epochs)

model = Sequential([
    Embedding(20000, args.embedding_dim, input_length=200),
    Bidirectional(LSTM(args.lstm_units)),
    Dropout(0.5),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(num_classes, activation="softmax"),
])
model.compile(
    optimizer=Adam(learning_rate=args.learning_rate),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=0),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=0),
]

history = model.fit(
    X_train, y_train,
    batch_size=args.batch_size,
    epochs=args.epochs,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=0,
)

val_acc = max(history.history["val_accuracy"])
val_loss = min(history.history["val_loss"])
mlflow.log_metric("val_accuracy", val_acc)
mlflow.log_metric("val_loss", val_loss)

y_pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

test_accuracy = np.mean(y_pred == y_test)
test_precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
test_recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
test_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

mlflow.log_metric("test_accuracy", test_accuracy)
mlflow.log_metric("test_precision", test_precision)
mlflow.log_metric("test_recall", test_recall)
mlflow.log_metric("test_f1", test_f1)

# Artifact: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap=plt.cm.Blues)
plt.colorbar(im)
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Negatif", "Positif"])
ax.set_yticklabels(["Negatif", "Positif"])
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
plt.tight_layout()
mlflow.log_figure(fig, "confusion_matrix.png")
plt.close()

# Artifact: Classification Report
report = classification_report(y_test, y_pred, target_names=["Negatif", "Positif"], output_dict=True, zero_division=0)
with open("classification_report.json", "w") as f:
    json.dump(report, f, indent=2)
mlflow.log_artifact("classification_report.json")
os.remove("classification_report.json")

# Log Model
mlflow.tensorflow.log_model(model, "model")

print(f"Test Accuracy: {test_accuracy:.4f}, F1: {test_f1:.4f}")
print("Training selesai.")
