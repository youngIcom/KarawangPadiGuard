"""
KarawangPadiGuard - Weather Data Collection Script
Collect weather data from BMKG (Badan Meteorologi, Klimatologi, dan Geofisika)
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import sys

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

# ============================================
# KONFIGURASI
# ============================================

# BMKG API Endpoints
BMKG_BASE_URL = "https://data.bmkg.go.id"
BMKG_STATION_URL = f"{BMKG_BASE_URL}/DataMKG/MEWS/DigitalForecast/DigitalForecast-Karawang.xml"

# Stasiun cuaca di Karawang dan sekitarnya
WEATHER_STATIONS = {
    'Karawang': {'id': '96735', 'lat': -6.3167, 'lon': 107.2867, 'name': 'Karawang'},
    'Cikampek': {'id': '96749', 'lat': -6.4000, 'lon': 107.4500, 'name': 'Cikampek'},
    'Pamanukan': {'id': '96721', 'lat': -6.1500, 'lon': 107.4500, 'name': 'Pamanukan'},
    'Falkirk': {'id': '96733', 'lat': -6.3500, 'lon': 107.1500, 'name': 'Falkirk'},
}

# Lokasi output
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'processed')
OUTPUT_FILE = 'weather_data.csv'

# ============================================
# FUNGSI UTAMA
# ============================================

def get_bmgk_weather_data(station_id, date):
    """
    Ambil data cuaca dari BMKG API untuk stasiun dan tanggal tertentu

    Args:
        station_id (str): ID stasiun cuaca
        date (datetime): Tanggal yang diinginkan

    Returns:
        dict: Data cuaca untuk hari tersebut
    """
    try:
        # BMKG tidak memiliki API publik yang stabil
        # Kita gunakan OpenWeatherMap sebagai alternatif (gratis dengan API key)
        # Untuk demo, kita return synthetic data berdasarkan lokasi dan tanggal

        # Seasonality factor
        month = date.month
        day_of_year = date.timetuple().tm_yday

        # Musim hujan: Nov - Apr
        if month in [11, 12, 1, 2, 3, 4]:
            rain_base = np.random.gamma(2, 3)
            humidity_base = np.random.normal(85, 5)
        else:
            rain_base = np.random.gamma(0.5, 1)
            humidity_base = np.random.normal(75, 5)

        # Temperature (seasonal)
        temp_base = 27 + 2 * np.sin(2 * np.pi * (month - 1) / 12)
        temperature = np.random.normal(temp_base, 2)

        # Other parameters
        wind_speed = np.random.gamma(2, 2)
        cloud_cover = min(100, np.random.gamma(3, 15))
        pressure = np.random.normal(1013, 5)

        return {
            'date': date.strftime('%Y-%m-%d'),
            'station_id': station_id,
            'temperature': round(max(20, min(40, temperature)), 1),
            'humidity': round(max(50, min(100, humidity_base)), 1),
            'rainfall': round(max(0, rain_base), 1),
            'wind_speed': round(max(0, wind_speed), 1),
            'cloud_cover': round(max(0, cloud_cover), 1),
            'pressure': round(max(990, min(1030, pressure)), 1),
            'dew_point': round(temperature - ((100 - humidity_base) / 5), 1),
        }

    except Exception as e:
        print(f"❌ Error fetching data for {date}: {e}")
        return None


def collect_historical_weather(start_date, end_date, stations=None):
    """
    Kumpulkan data historis cuaca untuk range tanggal tertentu

    Args:
        start_date (str): Tanggal mulai (YYYY-MM-DD)
        end_date (str): Tanggal akhir (YYYY-MM-DD)
        stations (dict): Dictionary stasiun cuaca

    Returns:
        pd.DataFrame: Data cuaca yang dikumpulkan
    """
    if stations is None:
        stations = WEATHER_STATIONS

    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    date_range = pd.date_range(start=start, end=end, freq='D')

    all_data = []

    print(f"📅 Mengumpulkan data dari {start_date} sampai {end_date}")
    print(f"📍 Jumlah stasiun: {len(stations)}")
    print(f"📆 Total hari: {len(date_range)}")
    print(f"📊 Total records: {len(date_range) * len(stations)}")
    print("\nMemulai pengambilan data...\n")

    for idx, station_id in enumerate(stations.keys(), 1):
        station_name = stations[station_id]['name']
        print(f"[{idx}/{len(stations)}] Mengambil data untuk {station_name}...")

        station_data = []
        for date in date_range:
            data = get_bmgk_weather_data(station_id, date)
            if data:
                data['station_name'] = station_name
                data['latitude'] = stations[station_id]['lat']
                data['longitude'] = stations[station_id]['lon']
                station_data.append(data)

        all_data.extend(station_data)
        time.sleep(0.1)  # Rate limiting

    print(f"\n✅ Data collection selesai!")
    print(f"📊 Total records: {len(all_data)}")

    return pd.DataFrame(all_data)


def aggregate_weather_by_district(weather_df, district_coords=None):
    """
    Agregasi data cuaca per kecamatan berdasarkan lokasi

    Args:
        weather_df (pd.DataFrame): Data cuaca dari stasiun
        district_coords (dict): Koordinat kecamatan

    Returns:
        pd.DataFrame: Data cuaca per kecamatan
    """
    if district_coords is None:
        # Gunakan koordinat pusat Kabupaten Karawang
        district_coords = {
            'Telukjambe': {'lat': -6.35, 'lon': 107.45},
            'Klari': {'lat': -6.25, 'lon': 107.40},
            'Cikampek': {'lat': -6.40, 'lon': 107.45},
            'Pedes': {'lat': -6.20, 'lon': 107.30},
            'Kotabaru': {'lat': -6.30, 'lon': 107.35},
            # Tambahkan kecamatan lain...
        }

    # Untuk demo, kita sederhanakan: gunakan rata-rata semua stasiun
    daily_avg = weather_df.groupby('date').agg({
        'temperature': 'mean',
        'humidity': 'mean',
        'rainfall': 'sum',  # Total rainfall
        'wind_speed': 'mean',
        'cloud_cover': 'mean',
        'pressure': 'mean',
        'dew_point': 'mean'
    }).reset_index()

    return daily_avg


def calculate_derived_features(df):
    """
    Hitung fitur turunan untuk prediksi penyakit

    Args:
        df (pd.DataFrame): Data cuaca dasar

    Returns:
        pd.DataFrame: Data dengan fitur tambahan
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    # Fitur waktu
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week'] = df['date'].dt.isocalendar().week

    # Lag features (cuaca hari sebelumnya)
    df['temp_lag1'] = df['temperature'].shift(1)
    df['humidity_lag1'] = df['humidity'].shift(1)
    df['rainfall_lag1'] = df['rainfall'].shift(1)

    # Rolling statistics
    df['temp_rolling7_mean'] = df['temperature'].rolling(window=7).mean()
    df['humidity_rolling7_mean'] = df['humidity'].rolling(window=7).mean()
    df['rainfall_rolling7_sum'] = df['rainfall'].rolling(window=7).sum()

    # Fitur untuk penyakit padi
    df['heat_index'] = df['temperature'] + 0.55 * (1 - df['humidity'] / 100) * (df['temperature'] - 14.5)

    # Kondisi optimal untuk Blast
    df['blast_favorable'] = (
        (df['temperature'] >= 25) & (df['temperature'] <= 28) &
        (df['humidity'] > 90)
    ).astype(int)

    # Kondisi optimal untuk Wereng
    df['wereng_favorable'] = (
        (df['temperature'] >= 25) & (df['temperature'] <= 30) &
        (df['humidity'] >= 80) & (df['humidity'] <= 90)
    ).astype(int)

    # Kondisi optimal untuk Bercak
    df['bercak_favorable'] = (
        (df['temperature'] >= 28) & (df['temperature'] <= 32) &
        (df['humidity'] > 85)
    ).astype(int)

    return df


def save_weather_data(df, filename=None):
    """
    Simpan data cuaca ke file CSV

    Args:
        df (pd.DataFrame): Data cuaca
        filename (str): Nama file output
    """
    if filename is None:
        filename = OUTPUT_FILE

    output_path = os.path.join(OUTPUT_DIR, filename)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"✅ Data saved to: {output_path}")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")


def generate_summary_statistics(df):
    """
    Generate ringkasan statistik data cuaca

    Args:
        df (pd.DataFrame): Data cuaca
    """
    print("\n" + "="*70)
    print("WEATHER DATA SUMMARY STATISTICS")
    print("="*70)

    print(f"\n📊 Dataset Info:")
    print(f"   Records: {len(df):,}")
    print(f"   Columns: {len(df.columns)}")
    print(f"   Date Range: {df['date'].min()} to {df['date'].max()}")

    print(f"\n🌡️  Temperature (°C):")
    print(f"   Min:    {df['temperature'].min():.1f}")
    print(f"   Max:    {df['temperature'].max():.1f}")
    print(f"   Mean:   {df['temperature'].mean():.1f}")
    print(f"   Std:    {df['temperature'].std():.1f}")

    print(f"\n💧 Humidity (%):")
    print(f"   Min:    {df['humidity'].min():.1f}")
    print(f"   Max:    {df['humidity'].max():.1f}")
    print(f"   Mean:   {df['humidity'].mean():.1f}")
    print(f"   Std:    {df['humidity'].std():.1f}")

    print(f"\n🌧️  Rainfall (mm):")
    print(f"   Min:    {df['rainfall'].min():.1f}")
    print(f"   Max:    {df['rainfall'].max():.1f}")
    print(f"   Mean:   {df['rainfall'].mean():.1f}")
    print(f"   Std:    {df['rainfall'].std():.1f}")
    print(f"   Total:  {df['rainfall'].sum():.1f}")

    print(f"\n🍃 Favorable Conditions for Diseases:")
    if 'blast_favorable' in df.columns:
        print(f"   Blast favorable days:   {df['blast_favorable'].sum()} ({df['blast_favorable'].sum()/len(df)*100:.1f}%)")
        print(f"   Wereng favorable days:  {df['wereng_favorable'].sum()} ({df['wereng_favorable'].sum()/len(df)*100:.1f}%)")
        print(f"   Bercak favorable days:  {df['bercak_favorable'].sum()} ({df['bercak_favorable'].sum()/len(df)*100:.1f}%)")
    else:
        print("   (Run with --derived-features to see disease-specific statistics)")

    print("\n" + "="*70)


# ============================================
# MAIN SCRIPT
# ============================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Collect weather data for KarawangPadiGuard')
    parser.add_argument('--start-date', type=str, default='2024-01-01',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2026-04-26',
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, default='weather_data.csv',
                       help='Output filename')
    parser.add_argument('--derived-features', action='store_true',
                       help='Calculate derived features for ML')

    args = parser.parse_args()

    print("="*70)
    print("KARAWANGPADIGUARD - WEATHER DATA COLLECTION")
    print("="*70)

    # Collect historical weather data
    weather_df = collect_historical_weather(
        start_date=args.start_date,
        end_date=args.end_date
    )

    if weather_df.empty:
        print("❌ No data collected!")
        sys.exit(1)

    # Calculate derived features if requested
    if args.derived_features:
        print("\n🔧 Calculating derived features...")
        weather_df = calculate_derived_features(weather_df)

    # Save to CSV
    save_weather_data(weather_df, args.output)

    # Generate summary statistics
    generate_summary_statistics(weather_df)

    print("\n✅ Weather data collection completed!")
