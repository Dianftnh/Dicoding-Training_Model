"""
modelling_tuning.py
Skilled/Menengah: Hyperparameter tuning + manual logging MLflow.
Manual logging (tanpa autolog) dengan metrics dan artifacts tambahan.
"""

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
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model_output")
os.makedirs(MODEL_DIR, exist_ok=True)

MAX_VOCAB_SIZE = 20000
MAX_SEQUENCE_LENGTH = 200
EPOCHS = 15

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

mlflow.set_experiment("imdb-review-tuning")


def build_model(embedding_dim=128, lstm_units=64, learning_rate=1e-3):
    model = Sequential([
        Embedding(MAX_VOCAB_SIZE, embedding_dim, input_length=MAX_SEQUENCE_LENGTH),
        Bidirectional(LSTM(lstm_units)),
        Dropout(0.5),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train_and_log(params, run_name):
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)

        model = build_model(embedding_dim=params["embedding_dim"], lstm_units=params["lstm_units"], learning_rate=params["learning_rate"])

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=0),
        ]

        history = model.fit(X_train, y_train, batch_size=params["batch_size"], epochs=params["epochs"], validation_data=(X_val, y_val), callbacks=callbacks, verbose=0)

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

        # Artifact 1: Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap=plt.cm.Blues)
        plt.colorbar(im)
        ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
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
        mlflow.log_figure(fig, f"confusion_matrix_{run_name}.png")
        plt.close()

        # Artifact 2: Training History Plot
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(history.history["accuracy"], label="Train")
        ax1.plot(history.history["val_accuracy"], label="Val")
        ax1.set_title("Accuracy")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.legend()
        ax2.plot(history.history["loss"], label="Train")
        ax2.plot(history.history["val_loss"], label="Val")
        ax2.set_title("Loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend()
        plt.tight_layout()
        mlflow.log_figure(fig2, f"training_history_{run_name}.png")
        plt.close()

        # Artifact 3: Classification Report (JSON)
        report = classification_report(y_test, y_pred, target_names=["Negatif", "Positif"], output_dict=True, zero_division=0)
        report_path = f"classification_report_{run_name}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact(report_path)
        os.remove(report_path)

        # Log model
        mlflow.tensorflow.log_model(model, f"model_{run_name}")

        # Save to model_output/
        model_path = os.path.join(MODEL_DIR, f"{run_name}.keras")
        model.save(model_path)
        print(f"  Model saved: {model_path}")

        print(f"[{run_name}] Test Acc: {test_accuracy:.4f}, F1: {test_f1:.4f}")
        return test_accuracy


# Hyperparameter Grid
param_grid = [
    {"embedding_dim": 128, "lstm_units": 64,  "learning_rate": 1e-3, "batch_size": 64, "epochs": EPOCHS},
    {"embedding_dim": 128, "lstm_units": 128, "learning_rate": 1e-3, "batch_size": 64, "epochs": EPOCHS},
    {"embedding_dim": 256, "lstm_units": 64,  "learning_rate": 5e-4, "batch_size": 64, "epochs": EPOCHS},
]

best_acc, best_run = 0, None
for i, params in enumerate(param_grid):
    run_name = f"bilstm_tuning_{i+1}"
    acc = train_and_log(params, run_name)
    if acc > best_acc:
        best_acc, best_run = acc, run_name

print(f"\nBest run: {best_run} | Test Accuracy: {best_acc:.4f}")
print("Cek MLflow UI: mlflow ui")
