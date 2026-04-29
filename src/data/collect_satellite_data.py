"""
KarawangPadiGuard - Satellite Data Collection Script
Collect Sentinel-2 satellite imagery for Karawang area
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
from pathlib import Path

# ============================================
# KONFIGURASI
# ============================================

# Karawang bounding box
KARAWANG_BBOX = {
    'min_lat': -6.5,
    'max_lat': -6.1,
    'min_lon': 107.2,
    'max_lon': 107.7
}

# Sentinel-2 bands yang digunakan
SENTINEL_BANDS = {
    'B02': 'Blue',    # 10m
    'B03': 'Green',   # 10m
    'B04': 'Red',     # 10m
    'B08': 'NIR',     # 10m
    'B11': 'SWIR1',   # 20m
    'B12': 'SWIR2'    # 20m
}

# Vegetation indices
VEGETATION_INDICES = {
    'NDVI': 'Normalized Difference Vegetation Index',
    'NDWI': 'Normalized Difference Water Index',
    'EVI': 'Enhanced Vegetation Index',
    'SAVI': 'Soil Adjusted Vegetation Index'
}

# Output directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = BASE_DIR / 'data' / 'satellite'
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'

# ============================================
# FUNGSI UTAMA
# ============================================

def calculate_ndvi(nir, red):
    """
    Hitung Normalized Difference Vegetation Index

    Args:
        nir (np.array): Near Infrared band
        red (np.array): Red band

    Returns:
        np.array: NDVI values (-1 to 1)
    """
    ndvi = (nir - red) / (nir + red + 1e-8)  # Avoid division by zero
    return np.clip(ndvi, -1, 1)


def calculate_ndwi(green, nir):
    """
    Hitung Normalized Difference Water Index

    Args:
        green (np.array): Green band
        nir (np.array): Near Infrared band

    Returns:
        np.array: NDWI values (-1 to 1)
    """
    ndwi = (green - nir) / (green + nir + 1e-8)
    return np.clip(ndwi, -1, 1)


def calculate_evi(nir, red, blue):
    """
    Hitung Enhanced Vegetation Index

    Args:
        nir (np.array): Near Infrared band
        red (np.array): Red band
        blue (np.array): Blue band

    Returns:
        np.array: EVI values
    """
    L = 1.0  # Canopy background adjustment
    C1 = 6.0  # Aerosol resistance coefficient
    C2 = 7.5  # Aerosol resistance coefficient
    G = 2.5   # Gain factor

    evi = G * ((nir - red) / (nir + C1 * red - C2 * blue + L + 1e-8))
    return np.clip(evi, -1, 1)


def calculate_savi(nir, red, L=0.5):
    """
    Hitung Soil Adjusted Vegetation Index

    Args:
        nir (np.array): Near Infrared band
        red (np.array): Red band
        L (float): Soil brightness correction factor

    Returns:
        np.array: SAVI values
    """
    savi = ((nir - red) / (nir + red + L)) * (1 + L)
    return np.clip(savi, -1, 1)


def generate_synthetic_satellite_data(start_date, end_date):
    """
    Generate synthetic satellite data untuk demo

    Dalam implementasi nyata, ini akan diganti dengan:
    - SentinelHub API request
    - Google Earth Engine processing

    Args:
        start_date (str): Tanggal mulai
        end_date (str): Tanggal akhir

    Returns:
        pd.DataFrame: Synthetic satellite indices data
    """
    dates = pd.date_range(start=start_date, end=end_date, freq='W')  # Weekly

    data = []
    for date in dates:
        month = date.month

        # Seasonality factor untuk vegetasi
        # Padi musim tanam: biasanya Nov-Mar dan Apr-Ag
        if month in [11, 12, 1, 2, 3]:
            # Musim tanam pertama - vegetasi meningkat
            base_ndvi = 0.6 + 0.2 * np.sin(2 * np.pi * (month - 11) / 5)
        elif month in [4, 5, 6, 7, 8]:
            # Musim tanam kedua - vegetasi stabil tinggi
            base_ndvi = 0.75 + 0.1 * np.random.randn()
        else:
            # Musim panen - vegetasi menurun
            base_ndvi = 0.4 + 0.1 * np.random.randn()

        # Add variability
        ndvi = base_ndvi + np.random.normal(0, 0.05)
        ndvi = np.clip(ndvi, -0.2, 0.9)

        # NDWI (water) - inversely correlated with NDVI
        ndwi = 0.2 - 0.3 * ndvi + np.random.normal(0, 0.03)
        ndwi = np.clip(ndwi, -0.5, 0.8)

        # EVI (enhanced)
        evi = ndvi * 1.1 + np.random.normal(0, 0.03)
        evi = np.clip(evi, -0.2, 0.95)

        # SAVI (soil-adjusted)
        savi = ndvi * 0.95 + np.random.normal(0, 0.02)
        savi = np.clip(savi, -0.2, 0.9)

        # Anomaly detection flag
        anomaly = abs(ndvi) < 0.2  # Stress condition

        data.append({
            'date': date.strftime('%Y-%m-%d'),
            'ndvi': round(ndvi, 3),
            'ndwi': round(ndwi, 3),
            'evi': round(evi, 3),
            'savi': round(savi, 3),
            'anomaly': int(anomaly),
            'month': month
        })

    return pd.DataFrame(data)


def analyze_vegetation_trends(df):
    """
    Analisis tren vegetasi dari data satelit

    Args:
        df (pd.DataFrame): Data indeks vegetasi

    Returns:
        dict: Hasil analisis
    """
    results = {
        'mean_ndvi': df['ndvi'].mean(),
        'std_ndvi': df['ndvi'].std(),
        'mean_ndwi': df['ndwi'].mean(),
        'anomaly_count': df['anomaly'].sum(),
        'anomaly_percentage': df['anomaly'].sum() / len(df) * 100
    }

    return results


def save_satellite_data(df, filename='satellite_indices.csv'):
    """
    Simpan data satelit ke CSV

    Args:
        df (pd.DataFrame): Data satelit
        filename (str): Nama file output
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / filename

    df.to_csv(output_path, index=False)
    print(f"✅ Satellite data saved to: {output_path}")


def save_processed_satellite_data(df, filename='satellite_data.csv'):
    """
    Simpan data satelit yang sudah diproses ke folder processed

    Args:
        df (pd.DataFrame): Data satelit
        filename (str): Nama file output
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / filename

    df.to_csv(output_path, index=False)
    print(f"✅ Processed satellite data saved to: {output_path}")


def generate_satellite_summary(df):
    """
    Generate ringkasan data satelit

    Args:
        df (pd.DataFrame): Data satelit
    """
    print("\n" + "="*70)
    print("SATELLITE DATA SUMMARY")
    print("="*70)

    print(f"\n📊 Dataset Info:")
    print(f"   Records: {len(df):,}")
    print(f"   Date Range: {df['date'].min()} to {df['date'].max()}")

    print(f"\n🌿 Vegetation Indices:")
    print(f"   NDVI (Vegetation Health):")
    print(f"      Mean:   {df['ndvi'].mean():.3f}")
    print(f"      Std:    {df['ndvi'].std():.3f}")
    print(f"      Min:    {df['ndvi'].min():.3f}")
    print(f"      Max:    {df['ndvi'].max():.3f}")

    print(f"   NDWI (Water Stress):")
    print(f"      Mean:   {df['ndwi'].mean():.3f}")
    print(f"      Std:    {df['ndwi'].std():.3f}")

    print(f"   EVI (Enhanced Vegetation):")
    print(f"      Mean:   {df['evi'].mean():.3f}")
    print(f"      Std:    {df['evi'].std():.3f}")

    print(f"\n⚠️  Anomaly Detection:")
    print(f"   Anomalous weeks: {df['anomaly'].sum()} ({df['anomaly'].sum()/len(df)*100:.1f}%)")

    # Health status
    avg_ndvi = df['ndvi'].mean()
    if avg_ndvi > 0.6:
        status = "🟢 Healthy"
    elif avg_ndvi > 0.4:
        status = "🟡 Moderate Stress"
    else:
        status = "🔴 High Stress"

    print(f"\n🏥 Overall Health Status: {status}")

    print("\n" + "="*70)


# ============================================
# MAIN SCRIPT
# ============================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Collect satellite data for KarawangPadiGuard')
    parser.add_argument('--start-date', type=str, default='2024-01-01',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2026-04-26',
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, default='satellite_indices.csv',
                       help='Output filename')

    args = parser.parse_args()

    print("="*70)
    print("KARAWANGPADIGUARD - SATELLITE DATA COLLECTION")
    print("="*70)

    # Generate synthetic satellite data
    print(f"\n📡 Collecting satellite data from {args.start_date} to {args.end_date}...")
    satellite_df = generate_synthetic_satellite_data(args.start_date, args.end_date)

    # Save raw data
    save_satellite_data(satellite_df, args.output)

    # Save processed data
    save_processed_satellite_data(satellite_df, 'satellite_data.csv')

    # Generate summary
    generate_satellite_summary(satellite_df)

    print("\n✅ Satellite data collection completed!")
    print("\n📝 Note: In production, this script will:")
    print("   - Connect to SentinelHub API")
    print("   - Download Sentinel-2 imagery")
    print("   - Process bands (B02, B03, B04, B08)")
    print("   - Calculate vegetation indices")
    print("   - Detect anomalies")
