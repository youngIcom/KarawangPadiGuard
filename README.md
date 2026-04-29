# KarawangPadiGuard

> Sistem Deteksi Dini & Peringatan Penyakit Padi Berbasis AI untuk Kabupaten Karawang
> **Microsoft Elevate Training Center - AI Impact Challenge 2026**

[![Streamlit App](https://img.shields.io/badge/Streamlit-Demo-blue?logo=streamlit)](https://huggingface.co/spaces/yesayasentosa/karawangpadiguard)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)

## 🌾 Tentang Proyek

KarawangPadiGuard adalah solusi AI yang mengintegrasikan **Computer Vision** dan **Machine Learning** untuk:

1. **Deteksi Penyakit** - Identifikasi 6 jenis penyakit padi dari foto tanaman (< 5 detik)
2. **Prediksi Risiko** - Forecast risiko 7 hari ke depan berbasis data cuaca
3. **Peringatan Dini** - Alert otomatis untuk petani dan penyuluh

### 🎯 Problem Statement

Indonesia menghadapi kehilangan hasil panen **20-40%** akibat serangan hama dan penyakit padi. Di Karawang (lumbung pangan nasional), rasio penyuluh pertanian adalah **1:1.400 ha**, membuat deteksi penyakit sering terlambat.

**Solusi:** Transformasi proses identifikasi dari 3-7 hari menjadi < 5 detik dengan pendekatan preventif melalui prediksi risiko.

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

**Dataset:** 3.829 gambar, 6 kelas (Rice Disease Dataset)

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
- **Cloud**: Microsoft Azure (ML, Communication Services)
- **Deployment**: HuggingFace Spaces
- **MLOps**: Weights & Biases

## 📁 Struktur Project

```
KarawangPadiGuard/
├── data/
│   ├── processed/        # Data yang sudah dibersihkan
│   ├── raw/              # Data mentah
│   ├── satellite/        # Data citra satelit (roadmap)
│   └── weather/          # Data cuaca BMKG
├── models/              # Model yang sudah dilatih
│   ├── mobilenetv3_rice_disease_v1_best.keras
│   ├── xgboost_risk_prediction_v1.pkl
│   └── ...
├── notebooks/            # Jupyter notebooks untuk EDA
├── src/
│   ├── data/            # Script pengumpulan data
│   ├── models/          # Script training model
│   └── api/             # API endpoints (roadmap)
├── app.py               # Streamlit demo app
├── Dockerfile           # Untuk deployment
├── requirements_app.txt # Dependencies untuk app
└── README.md            # File ini
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
# Run Streamlit app
streamlit run app.py
```

Aplikasi akan terbuka di `http://localhost:8501`

### Training Models

```bash
# Train Computer Vision model
Jalankan file train-cv-model.ipynb

# Train Risk Prediction model
Jalankan file train-risk-model.ipynb
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
| Data Produksi | 518.785 ton (2021, 174 desa) |
| Nilai Ekonomi | Rp 3,1 Triliun |
| Potensi Penghematan | Rp 155-311 Miliar/tahun (efisiensi 5-10%) |

## 🏆 Kompetisi

**Microsoft Elevate Training Center - AI Impact Challenge 2026**

- **Tema**: Ketahanan Pangan & Agrikultur Modern
- **Problem**: Bagaimana integrasi data satelit, citra digital, dan multimodal ML dapat menciptakan sistem peringatan dini untuk mengurangi kehilangan hasil panen?
- **Status**: Submission Final

## 📦 Deliverables

| Item | Link |
|------|------|
| **GitHub Repository** | [https://github.com/youngIcom/KarawangPadiGuard.git](https://github.com/youngIcom/KarawangPadiGuard.git) |
| **Project Brief** | Lihat file `PROJECT_BRIEF_FINAL.txt` |
| **Video Demo** | [Upload Soon] |

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

© 2026 Yesaya Situmorang - Microsoft Elevate Training Center

---

**Dibuat dengan ❤️ untuk petani Indonesia**
