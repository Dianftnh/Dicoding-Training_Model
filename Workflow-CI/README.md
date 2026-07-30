# Workflow-CI — IMDB Review Classifier

**Dicoding — Membangun Sistem Machine Learning**  
Kriteria 3: CI/CD otomatis menggunakan MLflow Project + GitHub Actions

---

## Struktur

```
Workflow-CI/
├── .github/workflows/ci.yml          # GitHub Actions
├── MLProject/
│   ├── modelling.py                   # Training script (argparse)
│   ├── MLProject                      # MLflow Project definition
│   ├── conda.yaml                     # Conda environment
│   └── dataset_preprocessing/         # Preprocessed data
└── Tautan ke Docker Hub.txt           # (Advanced)
```

## Cara Kerja

Trigger: push ke `main` yang mengubah file di `MLProject/`  
Atau manual via **Actions** tab → `Run workflow`

Pipeline:
1. Checkout repo
2. Setup Python
3. Install dependencies
4. `mlflow run . --env-manager=local` dengan hyperparameter
5. Upload artifacts (mlruns/) ke GitHub

## Level

| Level | Keterangan |
|-------|-----------|
| Basic | MLProject folder + workflow trigger |
| Skilled | Menyimpan artifact ke repositori |
| Advanced | + Docker Images ke Docker Hub |

---

👤 **Dian Fatonah** — Dicoding 2026
