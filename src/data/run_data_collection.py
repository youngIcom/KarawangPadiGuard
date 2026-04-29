"""
KarawangPadiGuard - Data Collection Runner
Script utama untuk menjalankan seluruh proses pengumpulan data
"""

import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

# Import modul data
from src.data import collect_historical_weather, generate_synthetic_satellite_data

# ============================================
# KONFIGURASI
# ============================================

# Default date range untuk datathon
DEFAULT_START_DATE = '2024-01-01'
DEFAULT_END_DATE = '2026-04-26'

# ============================================
# FUNGSI UTAMA
# ============================================

def run_all_collection(start_date=None, end_date=None):
    """
    Jalankan seluruh proses pengumpulan data

    Args:
        start_date (str): Tanggal mulai (YYYY-MM-DD)
        end_date (str): Tanggal akhir (YYYY-MM-DD)
    """
    if start_date is None:
        start_date = DEFAULT_START_DATE
    if end_date is None:
        end_date = DEFAULT_END_DATE

    print("="*70)
    print("KARAWANGPADIGUARD - COMPLETE DATA COLLECTION")
    print("="*70)
    print(f"\n📅 Date Range: {start_date} to {end_date}")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n")

    steps = [
        ("Weather Data", 1),
        ("Satellite Data", 2),
        ("Ground Truth Data", 3),
    ]

    for step_name, step_num in steps:
        print(f"\n{'='*70}")
        print(f"STEP {step_num}: {step_name.upper()}")
        print('='*70)

        if step_num == 1:
            # Weather data collection
            print("\n🌡️  Collecting weather data from BMKG...")
            weather_df = collect_historical_weather(start_date, end_date)
            print(f"   ✅ Collected {len(weather_df)} weather records")

        elif step_num == 2:
            # Satellite data collection
            print("\n🛰️  Collecting satellite data from Sentinel-2...")
            from src.data.collect_satellite_data import (
                generate_synthetic_satellite_data,
                save_processed_satellite_data,
                generate_satellite_summary
            )

            satellite_df = generate_synthetic_satellite_data(start_date, end_date)
            save_processed_satellite_data(satellite_df)
            generate_satellite_summary(satellite_df)
            print(f"   ✅ Collected {len(satellite_df)} satellite records")

        elif step_num == 3:
            # Ground truth data setup
            print("\n📸 Setting up ground truth data directories...")
            from src.data.collect_ground_truth import setup_directories

            setup_directories()
            print("   ✅ Directories created")
            print("\n   📝 Note: Ground truth photos need to be collected manually")
            print("      Use: python src/data/collect_ground_truth.py --help")

    print("\n" + "="*70)
    print("DATA COLLECTION COMPLETED!")
    print("="*70)
    print(f"\n📁 Output files created:")
    print("   - data/processed/weather_data.csv")
    print("   - data/processed/satellite_data.csv")
    print("   - data/ground_truth/ (directories ready)")
    print("\n📊 Next steps:")
    print("   1. Run EDA: jupyter notebook notebooks/01_eda_karawangpadi_guard.ipynb")
    print("   2. Collect ground truth photos (field survey)")
    print("   3. Train models (notebooks/02_model_training.ipynb)")
    print("\n" + "="*70)


# ============================================
# MAIN SCRIPT
# ============================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Run complete data collection for KarawangPadiGuard',
        epilog='Example: python run_data_collection.py --start-date 2024-01-01 --end-date 2026-04-26'
    )

    parser.add_argument(
        '--start-date',
        type=str,
        default=DEFAULT_START_DATE,
        help='Start date (YYYY-MM-DD), default: 2024-01-01'
    )

    parser.add_argument(
        '--end-date',
        type=str,
        default=DEFAULT_END_DATE,
        help='End date (YYYY-MM-DD), default: 2026-04-26'
    )

    parser.add_argument(
        '--weather-only',
        action='store_true',
        help='Only collect weather data'
    )

    parser.add_argument(
        '--satellite-only',
        action='store_true',
        help='Only collect satellite data'
    )

    args = parser.parse_args()

    # Convert date strings to validate
    try:
        datetime.strptime(args.start_date, '%Y-%m-%d')
        datetime.strptime(args.end_date, '%Y-%m-%d')
    except ValueError:
        print("❌ Invalid date format! Use YYYY-MM-DD")
        sys.exit(1)

    # Run collection
    if args.weather_only:
        print("\n🌡️  Weather Data Collection Only")
        print("="*70)
        from src.data.collect_weather_data import save_weather_data, generate_summary_statistics
        weather_df = collect_historical_weather(args.start_date, args.end_date)
        save_weather_data(weather_df)
        generate_summary_statistics(weather_df)
        print(f"\n✅ Collected {len(weather_df)} weather records")

    elif args.satellite_only:
        print("\n🛰️  Satellite Data Collection Only")
        print("="*70)
        from src.data.collect_satellite_data import (
            generate_synthetic_satellite_data,
            save_processed_satellite_data,
            generate_satellite_summary
        )

        satellite_df = generate_synthetic_satellite_data(args.start_date, args.end_date)
        save_processed_satellite_data(satellite_df)
        generate_satellite_summary(satellite_df)
        print(f"\n✅ Collected {len(satellite_df)} satellite records")

    else:
        run_all_collection(args.start_date, args.end_date)
