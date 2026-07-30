from prometheus_client import start_http_server, Gauge, Counter, Histogram
import requests
import numpy as np
import pickle
import json
import time
import random
import os

PREDICTIONS_TOTAL = Counter(
    'model_predictions_total',
    'Total jumlah prediksi yang dilakukan'
)
PREDICTION_LATENCY = Histogram(
    'model_prediction_latency_seconds',
    'Latency prediksi dalam detik',
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
)
ENDPOINT_UP = Gauge(
    'model_endpoint_up',
    'Status endpoint model (1=up, 0=down)'
)
PREDICTION_ERROR_RATE = Gauge(
    'model_prediction_error_rate',
    'Persentase error dari total prediksi'
)
MODEL_ACCURACY = Gauge(
    'model_accuracy_score',
    'Estimasi akurasi model saat ini'
)

MODEL_ENDPOINT = "http://127.0.0.1:8080/invocations"
DATA_DIR = "../Membangun_model/dataset_preprocessing"

def send_request(payload: dict) -> tuple:
    start = time.time()
    try:
        resp = requests.post(
            MODEL_ENDPOINT,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=5,
        )
        latency = time.time() - start
        if resp.status_code == 200:
            return resp.json(), latency, False
        return None, latency, True
    except Exception as e:
        print(f"[ERROR] Request gagal: {e}")
        return None, time.time() - start, True

def run_exporter():
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    total_requests = 0
    total_errors = 0
    correct_preds = 0

    print("[INFO] Prometheus exporter berjalan di port 8001")
    print("[INFO] Mengirim request ke model setiap 5 detik...")

    while True:
        idx = random.randint(0, len(X_test) - 1)
        sample = X_test[idx : idx + 1]
        actual = y_test[idx]

        payload = {"instances": sample.tolist()}

        result, latency, is_error = send_request(payload)

        PREDICTIONS_TOTAL.inc()
        PREDICTION_LATENCY.observe(latency)
        total_requests += 1

        if is_error:
            total_errors += 1
            ENDPOINT_UP.set(0)
            print(f"[ERROR] Request #{total_requests} gagal | Latency: {latency:.3f}s")
        else:
            ENDPOINT_UP.set(1)
            prediction = int(np.argmax(result["predictions"][0]))

            if prediction == actual:
                correct_preds += 1
            accuracy = correct_preds / total_requests
            MODEL_ACCURACY.set(accuracy)

            print(f"[OK] Request #{total_requests} | Pred: {prediction} | Actual: {actual} | Latency: {latency:.3f}s | Accuracy: {accuracy:.3f}")

        error_rate = total_errors / total_requests
        PREDICTION_ERROR_RATE.set(error_rate)

        time.sleep(5)

if __name__ == "__main__":
    start_http_server(8001)
    run_exporter()
