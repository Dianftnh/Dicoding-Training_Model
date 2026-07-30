Kriteria 3: Membuat Workflow CI
Setelah membuat dan memastikan file modelling.py berjalan dengan baik, selanjutnya Anda harus membuat
workflow CI menggunakan MLflow Project agar dapat melakukan re-training model secara otomatis ketika
trigger dipantik. 
Silakan Anda buat sebuah project repository baru di GitHub dengan struktur seperti berikut ini.
1. Workflow-CI
2. ├── .workflow
3. ├── MLProject (folder)
4.     
5.     
6.     
7.     
8.     
9.     
└── modelling.py
└── conda.yaml
└── MLProject
└── namadataset_preprocessing (bisa berupa file atau folder)
└── Tautan ke Docker Hub
└── (file tambahan jika diperlukan)
Anda dapat menggunakan file modelling.py, conda.yaml serta dataset yang sudah siap dilatih dari hasil
eksperimen sebelumnya. Pada tahap ini, Anda hanya perlu membuat struktur yang diminta beserta file
MLProjectnya saja. Namun, tidak menutup kemungkinan Anda harus menyesuaikan file modelling.py ketika
masuk ke tahap ini.
Berikut adalah penilaian lengkap untuk kriteria 3:
Reject (0 pts)
Tidak membuat folder MLProject.
Tidak membuat workflow CI menggunakan GitHub Actions.
Basic (2 pts)
Membuat folder MLProject.
Membuat Worflow CI yang dapat membuat model machine learning ketika trigger terpantik.
Skilled (3 pts)
Membuat workflow CI dan menyimpan artefak ke suatu repositori (GitHub yang sama atau Google
Drive).
Advance (4 pts)
Membuat workflow CI dan menyimpan artefak ke suatu repositori (GitHub yang sama atau Google
Drive) serta membuat Docker Images ke Docker Hub menggunakan fungsi mlflow build-docker.
Kriteria 4: Membuat Sistem Monitoring dan Logging
Monitoring dan Logging merupakan tahapan yang tidak bisa berdiri sendiri karena membutuhkan artefak yang
dihasilkan oleh kriteria tiga. Nantinya, Anda hanya akan mengumpulkan tangkapan layar mengenai skill yang
diampu dengan struktur seperti berikut ini.
├
4. ├── 3.prometheus_exporter.py
5. ├── 4.bukti monitoring Prometheus (folder)
6.     
7.     
8.     
└── 1.monitoring_<metriks>
└── 2.monitoring_<metriks>
└── dst (sesuaikan dengan poin yang diraih)
9. ├── 5.bukti monitoring Grafana (folder)
10.     
11.     
12.     
└── 1.monitoring_<metriks>
└── 2.monitoring_<metriks>
└── dst (sesuaikan dengan poin yang diraih)
13. ├── 6.bukti alerting Grafana (folder)
14.     
15.     
16.     
17.     
18.     
└── 1.rules_<metriks>
└── 2.notifikasi_<metriks>
└── 3.rules_<metriks>
└── 4.notifikasi_<metriks>
└── dst (sesuaikan dengan poin yang diraih)
19. ├── 7.inference.py
20. ├── folder/file tambahan
Penting, pastikan untuk membuat dashboard dengan nama username akun Dicoding sehingga tangkapan layar
yang Anda kirimkan akan berisikan kredensial.
Berikut adalah penilaian lengkap untuk kriteria 4:
Reject (0 pts)
Tidak melakukan serving model pada environment local.
Tidak melakukan monitoring performa sistem machine learning menggunakan Prometheus
Tidak menggunakan Grafana sebagai tools visualisasi dan alerting sistem machine learning
Basic (2 pts)
Melakukan serving model baik itu melalui artefak yang sudah dibuat atau pull Images (jika
menerapkan kriteria CI untuk melakukan push ke Docker Hub)
Bisa melalui mlflow model serve, mlflow deployments, atau pull images jika memenuhi kriteria 3
advanced.
Melakukan monitoring menggunakan Prometheus minimal dengan tiga metriks yang berbeda.
Melakukan monitoring menggunakan Grafana dengan metriks yang sama dengan Prometheus.
Skilled (3 pts)
Melakukan monitoring menggunakan Grafana dengan minimal 5 metriks yang berbeda.
Membuat satu alerting menggunakan Grafana.
Advance (4 pts)
Melakukan monitoring menggunakan Grafana dengan minimal 10 metriks yang berbeda.
Membuat tiga alerting menggunakan Grafana