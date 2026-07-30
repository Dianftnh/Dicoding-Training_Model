# 🧠 Membangun Model — Dian Fatonah

**Submission Dicoding | Membangun Sistem Machine Learning**  
Kriteria 2: Klasifikasi Sentimen IMDB Reviews menggunakan BiLSTM + MLflow Tracking

---

## 📋 Deskripsi

Folder ini berisi implementasi **Kriteria 2 — Membangun Model Machine Learning** untuk klasifikasi sentimen IMDB Reviews. Menggunakan model BiLSTM dengan MLflow Tracking.

**Level**: Skilled/Menengah — manual logging + hyperparameter tuning

---

## 🗂️ Struktur Folder

```
Dicoding-Training_Model/
├── Membangun_model/                    # Kriteria 2 — Modelling
│   ├── modelling.py
│   ├── modelling_tuning.py
│   ├── dataset_preprocessing/
│   ├── model_output/
│   ├── screenshoot_dashboard.jpg
│   └── screenshoot_artifak.jpg
│
├── Colab_Train_Monitor.ipynb           # Notebook Colab (train + monitor)
├── Workflow-CI.txt                    # Link ke repo Dicoding-Workflow-CI
│
└── Monitoring_dan_Logging/             # Kriteria 4 — Monitoring
    ├── 2.prometheus.yml
    ├── 3.prometheus_exporter.py
    ├── 7.Inference.py
    ├── 1.bukti_serving/
    ├── 4.bukti monitoring Prometheus/
    ├── 5.bukti monitoring Grafana/
    └── 6.bukti alerting Grafana/
```

---

## 📊 Dataset

| Properti | Detail |
|----------|--------|
| **Sumber** | `keras.datasets.imdb` |
| **Jumlah** | 50.000 ulasan film |
| **Label** | 0 = Negatif, 1 = Positif |
| **Tipe** | Teks bahasa Inggris |

## 🧠 Model

**Arsitektur**: `Embedding → Bidirectional LSTM → Dropout → Dense → Softmax`

### Hyperparameter Tuning

| Run | Embedding Dim | LSTM Units | Learning Rate |
|-----|--------------|------------|--------------|
| bilstm_tuning_1 | 128 | 64 | 1e-3 |
| bilstm_tuning_2 | 128 | 128 | 1e-3 |
| bilstm_tuning_3 | 256 | 64 | 5e-4 |

---

## 🚀 Cara Menjalankan

### Colab (Recommended)

Upload `Colab_Train_Monitor.ipynb` ke Google Colab, jalankan step by step:

1. Clone repo dari GitHub
2. Install dependencies
3. Training + tuning
4. Serve model + monitoring

Atau upload folder ke Drive dan akses langsung.

### Lokal

```bash
cd Membangun_model
pip install -r requirements.txt
python modelling.py          # Basic (autolog)
python modelling_tuning.py   # Skilled (manual logging + tuning)
mlflow ui                    # Buka MLflow UI
```

---

## 📸 Screenshot

Setelah training, jalankan:

```python
!mlflow ui --host 0.0.0.0 --port 5000 &
from pyngrok import ngrok
ngrok.set_auth_token("TOKEN_ANDA")
url = ngrok.connect(5000)
print(url)
```

Ambil screenshot:
- **Dashboard**: Halaman utama MLflow (daftar runs + metrics)
- **Artifacts**: Klik salah satu run → tab Artifacts

Simpan sebagai `screenshoot_dashboard.jpg` dan `screenshoot_artifak.jpg`.

---

## 🔁 Kriteria 3 — Workflow-CI (Repo Terpisah)

**GitHub Actions + MLflow Project** untuk retraining otomatis.

Trigger otomatis saat push ke `main` yang mengubah `MLProject/`.

> Folder `Workflow-CI/` berisi kode untuk dipush ke repo GitHub terpisah.

## 📊 Kriteria 4 — Monitoring & Logging

**Prometheus + Grafana** untuk memonitor performa model saat serving.

Gunakan `Colab_Train_Monitor.ipynb` (notebook lengkap) atau `Monitoring_dan_Logging/Colab_Monitoring.ipynb` (khusus monitoring).

Jalankan exporter, Prometheus (Docker), dan Grafana (Docker) untuk melihat metrik.

---

## 👤 Author

| Info | Detail |
|------|--------|
| **Nama** | Dian Fatonah |
| **Program** | Dicoding — Membangun Sistem Machine Learning |
| **Dataset** | IMDB Reviews (Sentiment Analysis) |
| **Model** | BiLSTM + MLflow Tracking |
