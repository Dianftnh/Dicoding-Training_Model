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
Membangun_model/
├── modelling.py                        # Basic MLflow autolog
├── modelling_tuning.py                 # Hyperparameter tuning + manual logging
├── dataset_preprocessing/              # Data preprocessing (.npy, .pkl)
│   ├── X_train.npy, X_val.npy, X_test.npy
│   ├── y_train.npy, y_val.npy, y_test.npy
│   ├── label_encoder.pkl
│   └── tokenizer.pkl
├── model_output/                       # Output model (generated saat training)
├── requirements.txt                    # Dependencies
├── screenshoot_dashboard.jpg           # Screenshot MLflow dashboard
├── screenshoot_artifak.jpg             # Screenshot MLflow artifacts
└── DagsHub.txt                         # Tidak dipakai (Menengah)
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

### Lokal

```bash
cd Membangun_model
pip install -r requirements.txt
python modelling.py          # Basic (autolog)
python modelling_tuning.py   # Skilled (manual logging + tuning)
mlflow ui                    # Buka MLflow UI
```

### Colab

1. Upload folder ke Google Drive
2. Buka notebook baru, jalankan:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   %cd /content/drive/MyDrive/.../Membangun_model
   !pip install -r requirements.txt
   %run modelling_tuning.py
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

## 👤 Author

| Info | Detail |
|------|--------|
| **Nama** | Dian Fatonah |
| **Program** | Dicoding — Membangun Sistem Machine Learning |
| **Dataset** | IMDB Reviews (Sentiment Analysis) |
| **Model** | BiLSTM + MLflow Tracking |
