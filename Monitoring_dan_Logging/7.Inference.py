import requests
import json
import numpy as np
import pickle
import os

MODEL_ENDPOINT = "http://127.0.0.1:8080/invocations"

DATA_DIR = "../Membangun_model/dataset_preprocessing"

X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
with open(os.path.join(DATA_DIR, "label_encoder.pkl"), "rb") as f:
    encoder = pickle.load(f)

sample = X_test[:5]
payload = {"instances": sample.tolist()}

response = requests.post(
    MODEL_ENDPOINT,
    headers={"Content-Type": "application/json"},
    data=json.dumps(payload),
)

print(f"Status Code : {response.status_code}")
if response.status_code == 200:
    result = response.json()
    preds = np.argmax(result["predictions"], axis=1)
    labels = encoder.inverse_transform(preds)
    actual_labels = encoder.inverse_transform(y_test[:5])
    for i in range(5):
        print(f"  Pred: {labels[i]:>7} | Actual: {actual_labels[i]:>7}")
else:
    print(f"Error: {response.text}")
