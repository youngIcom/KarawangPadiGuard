# KarawangPadiGuard

KarawangPadiGuard adalah prototipe sistem deteksi dini dan prediksi risiko penyakit padi berbasis AI untuk mendukung ketahanan pangan di Kabupaten Karawang.

Proyek ini dikembangkan untuk Microsoft Elevate Training Center - AI Impact Challenge 2026 dengan fokus pada integrasi Computer Vision, risk scoring berbasis cuaca, indeks vegetasi, dan analisis Value at Risk.

## Ringkasan

Penyakit padi seperti Leaf Blast, Brown Spot, dan Bacterial Leaf Blight dapat menyebabkan kehilangan hasil panen yang signifikan. Di wilayah lumbung pangan seperti Karawang, keterlambatan diagnosis dan minimnya prioritas mitigasi berbasis data dapat berdampak langsung pada ketahanan pangan daerah.

KarawangPadiGuard menjawab masalah tersebut melalui dua kemampuan utama:

1. Diagnosis penyakit padi dari foto daun menggunakan model Computer Vision.
2. Prediksi status risiko penyakit menggunakan model XGBoost berbasis cuaca, fitur temporal, indikator penyakit, dan indeks vegetasi.

Pada tahap ini, aplikasi Streamlit dijalankan secara lokal. Training Computer Vision dilakukan di Google Colab, sedangkan Microsoft Azure digunakan sebagai fondasi MLOps ringan untuk eksperimen, model registry, dan artifact tracking.

## Fitur Utama

- Deteksi penyakit padi dari foto daun.
- Klasifikasi 6 kelas: Bacterial Leaf Blight, Brown Spot, Healthy Rice Leaf, Leaf Blast, Leaf Scald, dan Sheath Blight.
- Prediksi risiko penyakit padi dengan status Low, Medium, atau High.
- Integrasi fitur cuaca, lag variables, rolling averages, indikator penyakit, serta indeks vegetasi NDVI, NDWI, EVI, dan SAVI.
- Analisis Golden Window berdasarkan pola kelembapan rolling 7 hari.
- Analisis Value at Risk untuk menentukan kecamatan prioritas intervensi.
- Setup Azure ML untuk eksperimen risk model dan model registry.

## Model Performance

### Computer Vision Model

| Item | Nilai |
| --- | --- |
| Arsitektur | MobileNetV3Small Transfer Learning |
| Dataset | 3.829 gambar, 6 kelas |
| Accuracy | 83,55% |
| Precision | 87,22% |
| Recall | 80,16% |
| F1-score | 83,54% |
| AUC | 97,69% |

Dataset gambar menggunakan Rice Disease Dataset dari Kaggle dengan lisensi terbuka CC-BY 4.0.

### Risk Prediction Model

| Item | Nilai |
| --- | --- |
| Algoritma | XGBoost |
| Feature set | 41 fitur cuaca, temporal, indikator penyakit, dan indeks vegetasi |
| Accuracy | 98,22% |
| Precision | 98,23% |
| Recall | 98,22% |
| F1-score | 98,22% |

Catatan: model risk prediction pada tahap ini adalah validasi prototipe risk scoring berbasis data cuaca dan fitur lingkungan. Label risiko dibangun dari aturan kondisi penyakit, sehingga klaim performa tidak boleh dibaca sebagai validasi wabah lapangan produksi.

## Dataset dan Insight

Dataset yang digunakan dalam prototipe:

- Data cuaca harian yang telah diproses menjadi 3.388 record.
- Data produksi padi Karawang 2021 untuk 174 desa.
- Dataset gambar penyakit padi dari Kaggle.
- Tabel indeks vegetasi prototype berisi NDVI, NDWI, EVI, dan SAVI.

Insight utama:

- `humidity_rolling_7` menjadi salah satu fitur paling berpengaruh pada model risiko.
- Kelembapan rata-rata 7 hari di atas 85% digunakan sebagai sinyal awal Golden Window 48-72 jam untuk intervensi preventif.
- Total aset produksi padi Karawang 2021 diperkirakan Rp 3,11 triliun.
- Potensi kerugian 20-40% setara Rp 622,5 miliar hingga Rp 1,25 triliun.
- Jika intervensi dini menekan 5-10% risiko, nilai ekonomi yang berpotensi dijaga mencapai Rp 155,6-311,3 miliar.
- Kecamatan prioritas awal berdasarkan Value at Risk: Tegalwaru, Tirtajaya, Pedes, Tirtamulya, dan Cilamaya Kulon.

## Microsoft Azure

Azure digunakan sebagai lapisan MLOps ringan, bukan sebagai hosting utama aplikasi pada tahap prototipe.

Layanan Azure yang digunakan:

- Azure Machine Learning Workspace untuk eksperimen dan lifecycle model.
- Azure ML Model Registry untuk versioning model XGBoost dan MobileNetV3.
- Default workspace storage untuk artifact eksperimen dan registry.
- Azure ML Compute CPU opsional untuk menjalankan job risk model skala kecil.

Layanan yang belum digunakan sebagai fitur berjalan:

- Azure App Service atau Container Apps untuk hosting permanen.
- Azure GPU Compute.
- Managed Online Endpoint.
- Azure Functions.
- Azure Communication Services.

Azure Functions dan Azure Communication Services masuk roadmap untuk otomasi data harian dan peringatan SMS/WhatsApp pada tahap produksi.

## Struktur Project

```text
KarawangPadiGuard/
├── app.py                         # Aplikasi Streamlit lokal
├── azure/                         # Konfigurasi Azure ML
├── data/                          # Struktur data lokal dan processed placeholders
├── models/                        # Placeholder dan metadata model ringan
├── notebooks/                     # Notebook EDA dan training Colab
├── src/
│   ├── analysis/                  # Analisis strategis dan Value at Risk
│   ├── data/                      # Script pengumpulan data
│   └── models/                    # Script training model
├── requirements.txt               # Dependency aplikasi/proyek
├── requirements_azure.txt         # Dependency setup Azure
├── risk-training-job.yml          # Job Azure ML dari root project
├── setup_azure_resources.py       # Helper setup Azure ML
└── README.md
```

File besar seperti model `.keras`, `.pkl`, MLflow runs, PDF, virtual environment, dan draft dokumentasi tidak dimasukkan ke GitHub.

## Menjalankan Aplikasi Lokal

Install dependency:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Jalankan Streamlit:

```bash
streamlit run app.py
```

Akses dari browser lokal:

```text
http://localhost:8501
```

Jika ingin membuka dari HP pada jaringan Wi-Fi yang sama:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Lalu buka dari HP:

```text
http://IP_LAPTOP:8501
```

## Notebook

Notebook utama:

- `notebooks/01_eda_karawangpadi_guard.ipynb`: EDA data cuaca, produksi, indeks vegetasi, dan dataset gambar.
- `notebooks/03_train_risk_model_colab.ipynb`: training risk model XGBoost di Google Colab.
- `notebooks/04_train_cv_model_colab.ipynb`: training Computer Vision MobileNetV3Small di Google Colab.

## Azure ML Quick Reference

Buat workspace dan register model jika artifact model tersedia lokal:

```bash
az login
az extension add -n ml -y
az extension update -n ml

export AZURE_RESOURCE_GROUP=KarawangPadiGuard_RG
export AZURE_LOCATION=southeastasia
export AZURE_ML_WORKSPACE=karawangpadiguard-ml

az group create --name "$AZURE_RESOURCE_GROUP" --location "$AZURE_LOCATION"
az ml workspace create --resource-group "$AZURE_RESOURCE_GROUP" --file azure/workspace.yml
```

Jalankan job risk model opsional:

```bash
az ml compute create \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --workspace-name "$AZURE_ML_WORKSPACE" \
  --file azure/cpu-cluster.yml

az ml job create \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --workspace-name "$AZURE_ML_WORKSPACE" \
  --file risk-training-job.yml \
  --stream
```

Setelah selesai, hapus compute agar biaya tidak berjalan:

```bash
az ml compute delete \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --workspace-name "$AZURE_ML_WORKSPACE" \
  --name cpu-cluster \
  --yes
```

## Roadmap

- Integrasi Sentinel-2 aktual untuk menggantikan tabel indeks vegetasi prototype.
- Otomatisasi pipeline data cuaca dan satelit menggunakan Azure Functions.
- Integrasi peringatan SMS/WhatsApp menggunakan Azure Communication Services.
- Penyempurnaan dashboard untuk penyuluh dan Dinas Pertanian.
- Continuous learning loop dari feedback foto petani dan penyuluh.

## Referensi

- Rice Disease Dataset: https://www.kaggle.com/datasets/anshulm257/rice-disease-dataset
- MobileNetV3: https://arxiv.org/abs/1905.02244
- XGBoost: https://xgboost.readthedocs.io/
- Azure Machine Learning: https://learn.microsoft.com/azure/machine-learning/

## Author

Yesaya Situmorang  
Microsoft Elevate Training Center - AI Impact Challenge 2026
