# Monitoring & Logging — IMDB Review Classifier

**Dicoding — Membangun Sistem Machine Learning**  
Kriteria 4: Monitoring & Logging dengan Prometheus & Grafana

---

## Struktur

```
Monitoring_dan_Logging/
├── 1.bukti_serving/                      # SS model serving berhasil
├── 2.prometheus.yml                      # Konfigurasi Prometheus
├── 3.prometheus_exporter.py              # Exporter metrik model
├── 4.bukti monitoring Prometheus/        # SS dashboard Prometheus
├── 5.bukti monitoring Grafana/           # SS dashboard Grafana
├── 6.bukti alerting Grafana/             # SS alerting rules
├── 7.Inference.py                        # Script uji inference
└── README.md
```

## Alur Menjalankan

1. **Serve model**  
   Jalankan MLflow model serving:
   ```bash
   mlflow models serve -m path/to/model --port 8080 --no-conda
   ```

2. **Uji Inference**  
   ```bash
   python "7.Inference.py"
   ```

3. **Jalankan Prometheus Exporter**  
   ```bash
   pip install prometheus_client requests numpy
   python "3.prometheus_exporter.py"
   ```
   Exporter berjalan di port `8001`, mengirim request ke model tiap 5 detik.

4. **Jalankan Prometheus**  
   Dengan Docker:
   ```bash
   docker run -p 9090:9090 -v /path/to/2.prometheus.yml:/etc/prometheus/prometheus.yml prom/prometheus
   ```

5. **Jalankan Grafana**  
   ```bash
   docker run -d -p 3000:3000 --name grafana grafana/grafana
   ```
   - Add Prometheus datasource: `http://host.docker.internal:9090`
   - Import / buat dashboard metrik model
   - Atur alerting rules

## Metrik yang Dipantau

| Metrik | Tipe | Deskripsi |
|--------|------|-----------|
| `model_predictions_total` | Counter | Total prediksi |
| `model_prediction_latency_seconds` | Histogram | Latency prediksi |
| `model_endpoint_up` | Gauge | Status endpoint |
| `model_prediction_error_rate` | Gauge | Error rate |
| `model_accuracy_score` | Gauge | Akurasi estimasi |

---

👤 **Dian Fatonah** — Dicoding 2026
