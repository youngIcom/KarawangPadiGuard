"""
KarawangPadiGuard - Streamlit Demo App
Microsoft Elevate Training Center - Datathon 2026

Sistem Deteksi Dini & Peringatan Penyakit Padi Berbasis AI

Author: Yesaya Situmorang
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import io

# Streamlit
import streamlit as st
from streamlit_option_menu import option_menu

# Image processing
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Check for XGBoost (optional dependency)
try:
    import xgboost
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not installed. Risk prediction feature will be disabled.")

# Page config
st.set_page_config(
    page_title="KarawangPadiGuard",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1f77b4, #2ecc71);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        font-size: 18px;
    }
</style>
""", unsafe_allow_html=True)


# ==================== CONFIGURATION ====================

CLASS_NAMES = [
    'Bacterial Leaf Blight',
    'Brown Spot',
    'Healthy Rice Leaf',
    'Leaf Blast',
    'Leaf scald',
    'Sheath Blight'
]

CLASS_NAMES_ID = [
    'Hawar Daun Bakteri',
    'Bercak Coklat',
    'Daun Sehat',
    'Blas Daun',
    'Hawar Seludang',
    'Blas Seludang'
]

CLASS_INFO = {
    'Bacterial Leaf Blight': {
        'nama_indonesia': 'Hawar Daun Bakteri',
        'pathogen': 'Xanthomonas oryzae pv. oryzae',
        'gejala': 'Daun mengering mulai dari ujung berwarna keabu-abuan',
        'penanganan': 'Gunakan varietas resistant, hindari nitrogen berlebih, semprot bacterisida pada awal serangan',
        'warna': '#e74c3c',
        'yield_loss': '30-50%'
    },
    'Brown Spot': {
        'nama_indonesia': 'Bercak Coklat',
        'pathogen': 'Cochliobolus miyabeanus',
        'gejala': 'Bercak berbentuk oval berwarna coklat',
        'penanganan': 'Berikan pupuk berimbang, gunakan fungisida berbahan aktif iprodione atau propiconazole',
        'warna': '#8b4513',
        'yield_loss': '10-25%'
    },
    'Healthy Rice Leaf': {
        'nama_indonesia': 'Daun Sehat',
        'pathogen': None,
        'gejala': 'Tanaman tumbuh normal, daun hijau sehat',
        'penanganan': 'Lanjutkan pemupukan berimbang dan pengairan berselang',
        'warna': '#2ecc71',
        'yield_loss': '0%'
    },
    'Leaf Blast': {
        'nama_indonesia': 'Blas Daun',
        'pathogen': 'Pyricularia oryzae',
        'gejala': 'Bercak berbentuk belah ketupat dengan warna abu-abu',
        'penanganan': 'Hindari nitrogen berlebih, gunakan fungisida trisiklazol saat gejala muncul',
        'warna': '#9b59b6',
        'yield_loss': '20-40%'
    },
    'Leaf scald': {
        'nama_indonesia': 'Hawar Seludang',
        'pathogen': 'Monographella albescens',
        'gejala': 'Bercak memanjang berwarna krem sampai coklat',
        'penanganan': 'Kurangi densitas tanaman, gunakan fungisida sistemik',
        'warna': '#f39c12',
        'yield_loss': '15-30%'
    },
    'Sheath Blight': {
        'nama_indonesia': 'Blas Seludang',
        'pathogen': 'Rhizoctonia solani',
        'gejala': 'Bercak oval berwarna coklat pada seludang daun',
        'penanganan': 'Hindari nitrogen berlebih, jaga jarak tanam, gunakan fungisida validamycin',
        'warna': '#16a085',
        'yield_loss': '20-25%'
    }
}

RISK_CATEGORIES = ['Low', 'Medium', 'High']
RISK_COLORS = {'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'}
RISK_ADVICE = {
    'Low': '🟢 Risiko Rendah - Lanjutkan pemantauan rutin',
    'Medium': '🟡 Risiko Sedang - Tingkatkan pengamatan, siapkan tindakan pencegahan',
    'High': '🔴 Risiko TingGI - Segera ambil tindakan preventif, perhatikan kondisi cuaca'
}


# ==================== MODEL LOADING ====================

@st.cache_resource
def load_cv_model():
    """Load Computer Vision model for disease detection"""
    try:
        model = load_model('models/mobilenetv3_rice_disease_v1_best.keras')
        return model
    except Exception as e:
        st.error(f"Error loading CV model: {e}")
        return None


@st.cache_resource
def load_risk_model():
    """Load Risk Prediction model"""
    # Check if XGBoost is available first
    if not XGBOOST_AVAILABLE:
        st.error("""
        ### ❌ XGBoost Not Installed

        The risk prediction feature requires XGBoost. Please install it:

        ```bash
        pip install xgboost
        ```

        Or run:
        ```bash
        pip install -r requirements_app.txt
        ```
        """)
        return None, None, None, None

    try:
        model = joblib.load('models/xgboost_risk_prediction_v1.pkl')
        scaler = joblib.load('models/xgboost_risk_prediction_v1_scaler.pkl')
        with open('models/xgboost_risk_prediction_v1_features.json', 'r') as f:
            features = json.load(f)
        with open('models/xgboost_risk_prediction_v1_config.json', 'r') as f:
            config = json.load(f)
        return model, scaler, features, config
    except FileNotFoundError as e:
        st.error(f"""
        ### ❌ Model Files Not Found

        Risk model files not found. Please:

        1. Train the model: `python src/models/train_risk_model.py`
        2. Or download from Kaggle output

        Missing file: {str(e)}
        """)
        return None, None, None, None
    except Exception as e:
        st.error(f"""
        ### ❌ Error Loading Risk Model

        An unexpected error occurred: {str(e)}

        Please check:
        1. XGBoost is installed: `pip install xgboost`
        2. Model files exist in `models/` directory
        3. Model files are not corrupted
        """)
        return None, None, None, None


# ==================== IMAGE PROCESSING ====================

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model prediction"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Resize
    image = image.resize(target_size)

    # Convert to array and normalize
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)
    img_array = img_array / 255.0

    return img_array


def predict_disease(model, image):
    """Predict disease from image"""
    processed_img = preprocess_image(image)
    predictions = model.predict(processed_img, verbose=0)[0]

    # Get top 3 predictions
    top_3_idx = np.argsort(predictions)[-3:][::-1]

    results = []
    for idx in top_3_idx:
        results.append({
            'class': CLASS_NAMES[idx],
            'class_id': CLASS_NAMES_ID[idx],
            'confidence': float(predictions[idx]) * 100,
            'info': CLASS_INFO[CLASS_NAMES[idx]]
        })

    return results


# ==================== RISK PREDICTION ====================

def engineer_risk_features(temp, humidity, rainfall, wind_speed, cloud_cover):
    """Create features for risk prediction"""
    current_date = datetime.now()

    features = {
        # Current weather
        'temperature': temp,
        'humidity': humidity,
        'rainfall': rainfall,
        'wind_speed': wind_speed,
        'cloud_cover': cloud_cover,

        # Temporal
        'month': current_date.month,
        'day_of_year': current_date.timetuple().tm_yday,
        'week_of_year': current_date.isocalendar()[1],

        # Season (simplified)
        'season': 'Rainy' if current_date.month in [12, 1, 2, 3, 4] else
                  'Dry' if current_date.month in [5, 6, 7, 8, 9] else 'Transitional'
    }

    # Create lag features (using current values as proxy)
    for lag in [1, 3, 7]:
        features[f'temp_lag_{lag}'] = temp
        features[f'humidity_lag_{lag}'] = humidity
        features[f'rainfall_lag_{lag}'] = rainfall

    # Rolling features (using current values as proxy)
    for window in [3, 7, 14]:
        features[f'temp_rolling_{window}'] = temp
        features[f'humidity_rolling_{window}'] = humidity
        features[f'rainfall_rolling_{window}'] = rainfall * window

    # Interactions
    features['temp_humidity_interaction'] = temp * humidity
    features['rain_intensity'] = 3 if rainfall > 20 else (2 if rainfall > 5 else (1 if rainfall > 0 else 0))
    features['rainfall_7day_cum'] = rainfall * 7
    features['temp_trend_3d'] = 0
    features['humidity_trend_3d'] = 0

    # Disease indicators
    features['blast_favorable'] = 1 if (25 <= temp <= 28 and humidity >= 90) else 0
    features['brown_spot_favorable'] = 1 if (28 <= temp <= 32 and humidity >= 85) else 0

    # Extremes
    features['extreme_heat'] = 1 if temp > 32 else 0
    features['extreme_humidity'] = 1 if humidity > 95 else 0
    features['heavy_rain'] = 1 if rainfall > 20 else 0

    # Encode season
    season_map = {'Rainy': 0, 'Dry': 1, 'Transitional': 2}
    features['season_encoded'] = season_map[features['season']]

    return features


def predict_risk(model, scaler, feature_names, weather_data):
    """Predict disease risk based on weather"""
    # Create features
    features = engineer_risk_features(**weather_data)

    # Prepare feature vector
    feature_vector = [features.get(f, 0) for f in feature_names]

    # Scale
    feature_vector_scaled = scaler.transform([feature_vector])

    # Predict
    prediction = model.predict(feature_vector_scaled)[0]
    risk_level = RISK_CATEGORIES[prediction]

    return risk_level, features


# ==================== SIDEBAR ====================

def render_sidebar():
    """Render sidebar navigation and info"""
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/rice.png", width=100)

        st.markdown("# 🌾 KarawangPadiGuard")
        st.markdown("---")

        # Navigation
        page = option_menu(
            "Menu",
            ["🏠 Beranda", "🔍 Deteksi Penyakit", "📊 Prediksi Risiko", "ℹ️ Tentang"],
            icons=["house", "search", "graph-up", "info-circle"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#f8f9fa"},
                "icon": {"color": "#1f77b4", "font-size": "18px"},
                "nav-link": {"font-size": "16px", "text-align": "left", "margin": "5px"},
                "nav-link-selected": {"background-color": "#1f77b4"},
            }
        )

        st.markdown("---")

        # Model Info
        st.markdown("### 📈 Model Performance")
        st.markdown("**Deteksi Penyakit:**")
        st.markdown("- Accuracy: **83.55%**")
        st.markdown("- AUC: **97.69%**")

        st.markdown("**Prediksi Risiko:**")
        st.markdown("- Accuracy: **98.37%**")
        st.markdown("- F1-Score: **98.37%**")

        st.markdown("---")

        # Contact
        st.markdown("### 👤 Tim")
        st.markdown("**Yesaya Situmorang**")
        st.markdown("Microsoft Elevate Training Center")

        return page


# ==================== PAGES ====================

def page_home():
    """Render home page"""
    st.markdown('<p class="main-header">🌾 KarawangPadiGuard</p>', unsafe_allow_html=True)
    st.markdown("### Sistem Deteksi Dini & Peringatan Penyakit Padi Berbasis AI")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🔍 Deteksi Instan</h3>
            <p>Identifikasi 6 jenis penyakit padi dalam &lt; 5 detik dari foto</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Prediksi Risiko</h3>
            <p>Forecast risiko 7 hari ke depan berbasis data cuaca</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🌾 Impact</h3>
            <p>Target: 70.000 ha lahan padi di Karawang</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Features
    st.subheader("🎯 Fitur Utama")

    tab1, tab2, tab3 = st.tabs(["Deteksi Penyakit", "Prediksi Risiko", "Teknologi"])

    with tab1:
        st.markdown("""
        ### 🔬 Deteksi Penyakit dari Foto

        Upload foto daun padi dan dapatkan diagnosa instan untuk:

        - **Hawar Daun Bakteri** (Xanthomonas oryzae)
        - **Bercak Coklat** (Cochliobolus miyabeanus)
        - **Blas Daun** (Pyricularia oryzae)
        - **Hawar Seludang** (Monographella albescens)
        - **Blas Seludang** (Rhizoctonia solani)
        - **Daun Sehat**

        Dilengkapi dengan saran penanganan untuk setiap penyakit!
        """)

    with tab2:
        st.markdown("""
        ### 📈 Prediksi Risiko Berbasis Cuaca

        Sistem menganalisis data cuaca untuk memprediksi risiko serangan penyakit:

        - Kelembapan tanah & udara
        - Suhu rata-rata
        - Curah hujan
        - Kecepatan angin

        Hasil: **Rendah**, **Sedang**, atau **Tinggi** dengan rekomendasi tindakan.
        """)

    with tab3:
        st.markdown("""
        ### 🤖 Teknologi yang Digunakan

        - **Computer Vision**: MobileNetV3 (Transfer Learning)
        - **Machine Learning**: XGBoost (Ensemble Learning)
        - **Framework**: TensorFlow, scikit-learn
        - **Deployment**: Streamlit, HuggingFace Spaces
        - **Cloud**: Microsoft Azure (ML, Functions, Blob Storage)
        """)

    st.markdown("---")

    # Stats
    st.subheader("📊 Dampak Potensial")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Area Studi", "70.000 ha", "Kabupaten Karawang")

    with col2:
        st.metric("Petani", "50.000 KK", "Penerima Manfaat")

    with col3:
        st.metric("Yield Loss Reduction", "20-40% → <10%", "Target")


def page_disease_detection():
    """Render disease detection page"""
    st.markdown("## 🔍 Deteksi Penyakit Padi")
    st.markdown("Upload foto daun padi untuk mendeteksi penyakit secara instan")

    # Load model
    model = load_cv_model()

    if model is None:
        st.error("❌ Model tidak tersedia. Silakan training model terlebih dahulu.")
        return

    # Upload
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📸 Upload Foto")
        uploaded_file = st.file_uploader(
            "Pilih foto daun padi...",
            type=['jpg', 'jpeg', 'png'],
            help="Format: JPG, JPEG, PNG. Max size: 10MB"
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)

            # Display original image
            st.image(image, caption="Foto yang diupload", use_column_width=True)

            # Predict button
            if st.button("🔍 Deteksi Penyakit", type="primary"):
                with st.spinner("Menganalisis gambar..."):
                    results = predict_disease(model, image)

                # Display results
                st.markdown("---")
                st.markdown("### 📋 Hasil Deteksi")

                # Top prediction
                top_result = results[0]
                confidence = top_result['confidence']

                # Color based on confidence
                if confidence > 80:
                    confidence_emoji = "🟢"
                elif confidence > 60:
                    confidence_emoji = "🟡"
                else:
                    confidence_emoji = "🔴"

                st.markdown(f"""
                <div style="padding: 20px; border-radius: 10px; background-color: {top_result['info']['warna']}20; border-left: 5px solid {top_result['info']['warna']};">
                    <h2>{top_result['class']}</h2>
                    <h3>{top_result['class_id']}</h3>
                    <p><strong>Confidence:</strong> {confidence_emoji} {confidence:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)

                # Disease info
                info = top_result['info']

                col_a, col_b = st.columns(2)

                with col_a:
                    st.markdown("#### 🦠 Pathogen")
                    st.write(info['pathogen'] if info['pathogen'] else "-")

                    st.markdown("#### ⚠️ Gejala")
                    st.write(info['gejala'])

                with col_b:
                    st.markdown("#### 💊 Penanganan")
                    st.write(info['penanganan'])

                    st.markdown("#### 📉 Potensi Kehilangan")
                    st.warning(info['yield_loss'])

                # Top 3 predictions
                st.markdown("---")
                st.markdown("### 📊 Prediksi Lengkap (Top 3)")

                for i, result in enumerate(results, 1):
                    st.markdown(f"""
                    **{i}. {result['class']}** ({result['class_id']})
                    - Confidence: **{result['confidence']:.2f}%**
                    """, unsafe_allow_html=True)

    with col2:
        st.markdown("### 📖 Cara Penggunaan")
        st.markdown("""
        1. Ambil foto daun padi yang mencurigakan
        2. Pastikan pencahayaan cukup
        3. Fokus pada bagian yang bergejala
        4. Upload foto di panel kiri
        5. Klik "Deteksi Penyakit"
        6. Dapatkan hasil diagnosa + rekomendasi

        **Tips untuk foto yang baik:**
        - Gunakan pencahayaan alami
        - Hindari bayangan
        - Fokus pada area bercak/berubah warna
        - Include edge of lesion jika ada
        """)

        st.markdown("---")

        st.markdown("### 🎯 Kelas Penyakit")
        for class_name, info in CLASS_INFO.items():
            st.markdown(f"""
            <div style="padding: 10px; margin: 5px 0; border-radius: 5px; background-color: {info['warna']}20; border-left: 3px solid {info['warna']};">
                <strong>{class_name}</strong> ({info['nama_indonesia']})<br>
                <small>Pathogen: {info['pathogen'] or '-'}</small>
            </div>
            """, unsafe_allow_html=True)


def page_risk_prediction():
    """Render risk prediction page"""
    st.markdown("## 📊 Prediksi Risiko Penyakit")
    st.markdown("Analisis risiko serangan penyakit berdasarkan kondisi cuaca")

    # Load model
    risk_model, scaler, features, config = load_risk_model()

    if risk_model is None:
        st.error("❌ Model tidak tersedia. Silakan training model terlebih dahulu.")
        return

    # Input form
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 🌡️ Input Data Cuaca")

        with st.form("weather_form"):
            st.markdown("#### Kondisi Saat Ini")

            col_a, col_b = st.columns(2)

            with col_a:
                temp = st.slider(
                    "Suhu (°C)",
                    min_value=15.0, max_value=40.0, value=28.0, step=0.5,
                    help="Suhu rata-rata harian"
                )

                humidity = st.slider(
                    "Kelembapan (%)",
                    min_value=40, max_value=100, value=85,
                    help="Kelembapan relatif udara"
                )

            with col_b:
                rainfall = st.slider(
                    "Curah Hujan (mm)",
                    min_value=0.0, max_value=50.0, value=5.0, step=0.5,
                    help="Curah hujan harian"
                )

                wind_speed = st.slider(
                    "Kecepatan Angin (km/jam)",
                    min_value=0.0, max_value=30.0, value=5.0, step=0.5,
                    help="Kecepatan angin rata-rata"
                )

            cloud_cover = st.slider(
                "Tutupan Awan (%)",
                min_value=0, max_value=100, value=60,
                help="Persentase tutupan awan"
            )

            submitted = st.form_submit_button("🔮 Prediksi Risiko", type="primary")

            if submitted:
                with st.spinner("Menganalisis data cuaca..."):
                    weather_data = {
                        'temp': temp,
                        'humidity': humidity,
                        'rainfall': rainfall,
                        'wind_speed': wind_speed,
                        'cloud_cover': cloud_cover
                    }

                    risk_level, risk_features = predict_risk(
                        risk_model, scaler, features, weather_data
                    )

                # Display results
                st.markdown("---")
                st.markdown("### 📋 Hasil Prediksi")

                risk_color = RISK_COLORS[risk_level]
                risk_emoji = "🟢" if risk_level == "Low" else ("🟡" if risk_level == "Medium" else "🔴")

                st.markdown(f"""
                <div style="padding: 30px; border-radius: 10px; background-color: {risk_color}20; border-left: 5px solid {risk_color}; text-align: center;">
                    <h1>{risk_emoji} Risiko: {risk_level}</h1>
                    <p style="font-size: 18px;">{RISK_ADVICE[risk_level]}</p>
                </div>
                """, unsafe_allow_html=True)

                # Key factors
                st.markdown("---")
                st.markdown("### 🔍 Faktor Kunci")

                factor_col1, factor_col2 = st.columns(2)

                with factor_col1:
                    st.metric("Kelembapan", f"{humidity}%", "⚠️ Tinggi" if humidity > 85 else "Normal")
                    st.metric("Suhu", f"{temp}°C", "Ideal" if 25 <= temp <= 32 else "Ekstrem")

                with factor_col2:
                    st.metric("Curah Hujan", f"{rainfall} mm", "Basah" if rainfall > 10 else "Normal")
                    st.metric("Favorable untuk Blast", "Ya" if risk_features['blast_favorable'] else "Tidak")

                # Disease risk breakdown
                st.markdown("---")
                st.markdown("### 🦠 Risiko per Penyakit")

                diseases = [
                    ('Blast', risk_features['blast_favorable']),
                    ('Brown Spot', risk_features['brown_spot_favorable'])
                ]

                for disease, favorable in diseases:
                    status = "⚠️ KONDISI FAVORABLE" if favorable else "✅ KONDISI TIDAK FAVORABLE"
                    color = "#e74c3c" if favorable else "#2ecc71"
                    st.markdown(f"""
                    <div style="padding: 15px; margin: 5px 0; border-radius: 5px; background-color: {color}20; border-left: 3px solid {color};">
                        <strong>{disease}:</strong> {status}
                    </div>
                    """, unsafe_allow_html=True)

    with col2:
        st.markdown("### 📖 Cara Penggunaan")
        st.markdown("""
        1. Masukkan data cuaca harian
        2. Klik "Prediksi Risiko"
        3. Dapatkan hasil risiko + rekomendasi

        **Data yang dibutuhkan:**
        - Suhu rata-rata harian
        - Kelembapan relatif
        - Curah hujan
        - Kecepatan angin
        - Tutupan awan

        **Interpretasi Hasil:**
        - **Low (Rendah)**: Kondisi tidak mendukung perkembangan penyakit
        - **Medium (Sedang)**: Waspadai, tingkatkan monitoring
        - **High (Tinggi)**: Segera ambil tindakan preventif
        """)

        st.markdown("---")

        st.markdown("### 🌡️ Kondisi Favorable Penyakit")

        st.markdown("""
        **Blast (Pyricularia oryzae)**
        - Suhu: 25-28°C
        - Kelembapan: >90%

        **Brown Spot (Cochliobolus miyabeanus)**
        - Suhu: 28-32°C
        - Kelembapan: >85%

        **Bacterial Blight (Xanthomonas oryzae)**
        - Suhu: 25-30°C
        - Kelembapan: >85%
        """)


def page_about():
    """Render about page"""
    st.markdown("## ℹ️ Tentang KarawangPadiGuard")

    st.markdown("""
    ### 🎯 Tujuan Proyek

    KarawangPadiGuard adalah sistem deteksi dini dan peringatan penyakit padi berbasis AI
    yang dirancang untuk membantu petani di Kabupaten Karawang mengurangi kehilangan hasil panen.

    ### 📍 Area Studi

    - **Lokasi**: Kabupaten Karawang, Jawa Barat
    - **Luas Lahan**: 70.000 hektar
    - **Jumlah Petani**: 50.000 kepala keluarga
    - **Status**: Lumbung Pangan Nasional

    ### 🤖 Teknologi

    **Computer Vision Model**
    - Arsitektur: MobileNetV3Small (Transfer Learning)
    - Dataset: 3.829 gambar, 6 kelas
    - Accuracy: 83.55%
    - AUC: 97.69%

    **Risk Prediction Model**
    - Algoritma: XGBoost (Gradient Boosting)
    - Features: 37 fitur cuaca
    - Accuracy: 98.37%
    - F1-Score: 98.37%

    ### 🏆 Kompetisi

    **Microsoft Elevate Training Center - AI Impact Challenge**

    - **Tema**: Ketahanan Pangan & Agrikultur Modern
    - **Problem**: Bagaimana mengurangi kehilangan hasil panen 20-40% akibat penyakit?
    - **Solusi**: Sistem peringatan dini berbasis AI untuk deteksi dan prediksi risiko

    ### 👨‍💻 Developer

    **Yesaya Situmorang**
    - Microsoft Elevate Training Center Participant
    - Email: yesayasentosa@gmail.com

    ### 📚 Referensi

    - Dataset Rice Disease: [Kaggle](https://www.kaggle.com/datasets/anshulm257/rice-disease-dataset)
    - MobileNetV3: [TensorFlow](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV3Small)
    - XGBoost: [Documentation](https://xgboost.readthedocs.io/)

    ### 🙏 Acknowledgments

    - Microsoft Elevate Training Center
    - Dicoding Indonesia
    - Komunitas AI Indonesia
    """)

    st.markdown("---")

    # Metrics display
    st.subheader("📊 Model Performance Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔬 Computer Vision")
        st.metric("Accuracy", "83.55%", "")
        st.metric("Precision", "87.22%", "")
        st.metric("Recall", "80.16%", "")
        st.metric("AUC", "97.69%", "")

    with col2:
        st.markdown("### 📈 Risk Prediction")
        st.metric("Accuracy", "98.37%", "")
        st.metric("Precision", "98.37%", "")
        st.metric("Recall", "98.37%", "")
        st.metric("F1-Score", "98.37%", "")


# ==================== MAIN APP ====================

def main():
    """Main application"""
    # Render sidebar and get selected page
    page = render_sidebar()

    # Route to appropriate page
    if page == "🏠 Beranda":
        page_home()
    elif page == "🔍 Deteksi Penyakit":
        page_disease_detection()
    elif page == "📊 Prediksi Risiko":
        page_risk_prediction()
    elif page == "ℹ️ Tentang":
        page_about()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>© 2026 KarawangPadiGuard | Microsoft Elevate Training Center - AI Impact Challenge</p>
        <p>Dikembangkan oleh <strong>Yesaya Situmorang</strong></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
