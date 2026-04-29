# KarawangPadiGuard

> Sistem Deteksi Dini & Peringatan Penyakit Padi Berbasis AI untuk Kabupaten Karawang
> **Microsoft Elevate Training Center - AI Impact Challenge 2026**


## 🌾 Tentang Proyek

KarawangPadiGuard adalah solusi AI yang mengintegrasikan **Computer Vision**, **Machine Learning**, dan **Satellite Imagery** untuk:

1. **Deteksi Penyakit** - Identifikasi 6 jenis penyakit padi dari foto tanaman (< 5 detik)
2. **Prediksi Risiko** - Forecast risiko 7 hari ke depan berbasis data cuaca
3. **Peringatan Dini** - Alert otomatis untuk petani dan penyuluh

### 🎯 Problem Statement

Indonesia menghadapi kehilangan hasil panen **20-40%** akibat serangan hama dan penyakit padi. Di Karawang (lumbung pangan nasional), rasio penyuluh pertanian adalah **1:1.400 ha**, membuat deteksi penyakit sering terlambat.

**Solusi:** Transformasi proses identifikasi dari 3-7 hari menjadi < 5 detik dengan pendekatan preventif melalui prediksi risiko.

## 🚀 Demo

DEMO APP : ON PROGRESS

## 🤖 Model Performance

### Computer Vision (Disease Detection)

| Metrik | Nilai |
|--------|-------|
| Model | MobileNetV3Small (Transfer Learning) |
| Accuracy | **83.55%** |
| Precision | **87.22%** |
| Recall | **80.16%** |
| AUC | **97.69%** |
| F1-Score | **83.54%** |

**Dataset:** 3,829 gambar, 6 kelas (Rice Disease Dataset)

### Risk Prediction (Tabular)

| Metrik | Nilai |
|--------|-------|
| Model | XGBoost (Gradient Boosting) |
| Accuracy | **98.37%** |
| Precision | **98.37%** |
| Recall | **98.37%** |
| F1-Score | **98.37%** |

**Features:** 37 fitur cuaca (kelembapan, suhu, curah hujan, dll)

## 📊 Kelas Penyakit yang Dideteksi

| Penyakit | Pathogen | Yield Loss |
|----------|----------|------------|
| Hawar Daun Bakteri | Xanthomonas oryzae | 30-50% |
| Bercak Coklat | Cochliobolus miyabeanus | 10-25% |
| Blas Daun | Pyricularia oryzae | 20-40% |
| Hawar Seludang | Monographella albescens | 15-30% |
| Blas Seludang | Rhizoctonia solani | 20-25% |
| Daun Sehat | - | 0% |

## 🛠️ Teknologi

- **ML/DL**: TensorFlow, XGBoost, scikit-learn
- **Frontend**: Streamlit
- **Cloud**: Microsoft Azure (ML, Functions, Blob Storage, Communication Services)
- **Deployment**: HuggingFace Spaces
- **MLOps**: Weights & Biases

## 📁 Struktur Project

```
KarawangPadiGuard/
├── data/
│   ├── raw/              # Data mentah
│   ├── processed/        # Data yang sudah dibersihkan
│   ├── satellite/        # Data citra satelit Sentinel-2
│   ├── weather/          # Data cuaca BMKG
│   └── ground_truth/     # Data lapangan
├── notebooks/            # Jupyter notebooks untuk EDA
├── src/
│   ├── data/            # Script pengumpulan data
│   ├── models/          # Script training model
│   └── api/             # API endpoints
├── models/              # Model yang sudah dilatih
├── logs/                # Log files
├── app.py               # Streamlit demo app
├── Dockerfile           # Untuk deployment
└── requirements_app.txt # Dependencies untuk app
```

## 🚀 Quick Start

### Setup Environment

```bash
# Clone repository
git clone https://github.com/yesayasentosa/KarawangPadiGuard.git
cd KarawangPadiGuard

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements_app.txt
```

### Run Demo App

```bash
# Download trained models dari:
# - CV Model: https://drive.google.com/file/.../mobilenetv3_rice_disease_v1_best.keras
# - Risk Model: https://drive.google.com/file/.../xgboost_risk_prediction_v1.pkl

# Place models in models/ directory

# Run Streamlit app
streamlit run app.py
```

### Training Models

```bash
# Train Computer Vision model
python src/models/train_cv_model.py

# Train Risk Prediction model
python src/models/train_risk_model.py
```

## 📱 Fitur Demo App

1. **Deteksi Penyakit**
   - Upload foto daun padi
   - Deteksi otomatis 6 kelas penyakit
   - Tampilkan confidence score
   - Saran penanganan untuk setiap penyakit

2. **Prediksi Risiko**
   - Input data cuaca harian
   - Prediksi risiko (Low/Medium/High)
   - Analisis faktor kunci
   - Rekomendasi tindakan

3. **Dashboard**
   - Model performance metrics
   - Informasi proyek
   - Dokumentasi teknis

## 📈 Dampak Potensial

| Metric | Nilai |
|--------|-------|
| Area Studi | 70.000 ha (Kabupaten Karawang) |
| Petani | 50.000 KK |
| Yield Loss Reduction | 20-40% → < 10% |
| Estimasi Nilai Ekonomi | Rp 70-140 miliar/tahun |

## 🏆 Kompetisi

**Microsoft Elevate Training Center - AI Impact Challenge 2026**

- **Tema**: Ketahanan Pangan & Agrikultur Modern
- **Problem**: Bagaimana integrasi data satelit, citra digital, dan multimodal ML dapat menciptakan sistem peringatan dini untuk mengurangi kehilangan hasil panen?

## 👥 Tim

**Yesaya Situmorang**
- Role: Machine Learning Engineer & Fullstack Developer
- Email: yesayasentosa@gmail.com
- Program: Microsoft Elevate Training Center

## 📚 Referensi

- [Rice Disease Dataset (Kaggle)](https://www.kaggle.com/datasets/anshulm257/rice-disease-dataset)
- [MobileNetV3 Paper](https://arxiv.org/abs/1905.02244)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Sentinel-2 Data (Copernicus)](https://scihub.copernicus.eu/)

## 📄 Lisensi

© 2026 Tim PadiGuardian - Microsoft Elevate Training Center

---

**Dibuat dengan ❤️ untuk petani Indonesia**
