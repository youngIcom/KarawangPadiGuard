"""
KarawangPadiGuard - Ground Truth Data Collection Script
Collect and manage ground truth data (paddy photos + labels)
"""

import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import json
import shutil
from pathlib import Path

# ============================================
# KONFIGURASI
# ============================================

# Directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
GROUND_TRUTH_DIR = BASE_DIR / 'data' / 'ground_truth'
PHOTOS_DIR = GROUND_TRUTH_DIR / 'photos'
LABELS_FILE = GROUND_TRUTH_DIR / 'labels.csv'

# Disease categories
DISEASE_CATEGORIES = {
    'HEALTHY': {
        'label': 0,
        'name': 'Sehat',
        'description': 'Tanaman padi sehat, tidak ada gejala penyakit',
        'color': [0, 255, 0]  # Green
    },
    'BLAST': {
        'label': 1,
        'name': 'Blast (Blas)',
        'description': 'Penyakit blas, bercak berbentuk belah ketupat',
        'pathogen': 'Pyricularia oryzae',
        'color': [139, 69, 19]  # Brown
    },
    'WERENG': {
        'label': 2,
        'name': 'Wereng Coklat',
        'description': 'Serangan wereng coklat, tanaman mengering',
        'pathogen': 'Nilaparvata lugens',
        'color': [210, 105, 30]  # Chocolate
    },
    'BERCAK': {
        'label': 3,
        'name': 'Bercak Coklat',
        'description': 'Bercak coklat pada daun',
        'pathogen': 'Helminthosporium oryzae',
        'color': [160, 82, 45]  # Sienna
    }
}

SEVERITY_LEVELS = {
    'RINGAN': {'value': 1, 'description': '1-20% area terkena'},
    'SEDANG': {'value': 2, 'description': '21-50% area terkena'},
    'BERAT': {'value': 3, 'description': '>50% area terkena'}
}

# ============================================
# FUNGSI UTAMA
# ============================================

def setup_directories():
    """Buat struktur direktori untuk ground truth data"""
    directories = [
        PHOTOS_DIR / 'healthy',
        PHOTOS_DIR / 'blast',
        PHOTOS_DIR / 'wereng',
        PHOTOS_DIR / 'bercak',
        PHOTOS_DIR / 'unlabeled',
        GROUND_TRUTH_DIR / 'train',
        GROUND_TRUTH_DIR / 'val',
        GROUND_TRUTH_DIR / 'test'
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    print("✅ Directories created successfully")


def validate_photo(photo_path):
    """
    Validasi kualitas foto sebelum diproses

    Args:
        photo_path (str): Path ke file foto

    Returns:
        tuple: (is_valid, message)
    """
    if not os.path.exists(photo_path):
        return False, "File tidak ditemukan"

    # Check file size (min 10KB)
    file_size = os.path.getsize(photo_path)
    if file_size < 10240:  # 10KB
        return False, f"File terlalu kecil ({file_size/1024:.1f} KB)"

    # Check image dimensions
    img = cv2.imread(str(photo_path))
    if img is None:
        return False, "File bukan gambar valid"

    height, width = img.shape[:2]

    # Check minimum resolution
    if width < 224 or height < 224:
        return False, f"Resolusi terlalu rendah ({width}x{height})"

    # Check if image is too blurry (Laplacian variance)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 100:  # Threshold untuk blur detection
        return False, f"Gambar terlalu blur (variance: {laplacian_var:.1f})"

    # Check brightness
    brightness = np.mean(gray)
    if brightness < 30 or brightness > 225:
        return False, f"Pencahayaan tidak sesuai (brightness: {brightness:.1f})"

    return True, "Valid"


def process_photo(photo_path, disease, severity, metadata=None):
    """
    Proses foto tanaman dan simpan ke direktori yang sesuai

    Args:
        photo_path (str): Path ke foto asli
        disease (str): Kategori penyakit
        severity (str): Tingkat keparahan
        metadata (dict): Metadata tambahan

    Returns:
        dict: Informasi foto yang sudah diproses
    """
    # Validate photo
    is_valid, message = validate_photo(photo_path)
    if not is_valid:
        print(f"❌ {message}: {photo_path}")
        return None

    # Read image
    img = cv2.imread(str(photo_path))
    if img is None:
        print(f"❌ Gagal membaca gambar: {photo_path}")
        return None

    # Generate filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    disease_lower = disease.lower()
    filename = f"{disease_lower}_{timestamp}.jpg"

    # Save to disease-specific directory
    output_path = PHOTOS_DIR / disease_lower / filename

    # Optional: Resize to standard size
    target_size = (640, 640)
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    # Save processed image
    cv2.imwrite(str(output_path), img_resized)

    # Prepare metadata
    photo_info = {
        'filename': filename,
        'original_path': str(photo_path),
        'processed_path': str(output_path),
        'disease': disease,
        'severity': severity,
        'label': DISEASE_CATEGORIES[disease]['label'],
        'timestamp': timestamp,
        'width': target_size[0],
        'height': target_size[1],
        'file_size': os.path.getsize(output_path),
        'date_collected': datetime.now().isoformat()
    }

    if metadata:
        photo_info.update(metadata)

    return photo_info


def create_label_dataframe(photos_info):
    """
    Buat DataFrame dari info foto untuk label CSV

    Args:
        photos_info (list): List dari dictionary info foto

    Returns:
        pd.DataFrame: DataFrame label
    """
    df = pd.DataFrame(photos_info)

    # Reorder columns
    columns_order = [
        'filename', 'disease', 'severity', 'label',
        'width', 'height', 'file_size',
        'date_collected'
    ]

    # Add optional columns if exist
    for col in df.columns:
        if col not in columns_order:
            columns_order.append(col)

    df = df[[col for col in columns_order if col in df.columns]]

    return df


def split_train_val_test(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, stratify=True):
    """
    Split dataset menjadi train, validation, dan test set

    Args:
        df (pd.DataFrame): Label dataframe
        train_ratio (float): Rasio data training
        val_ratio (float): Rasio data validasi
        test_ratio (float): Rasio data test
        stratify (bool): Stratified split berdasarkan label

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    from sklearn.model_selection import train_test_split

    if stratify:
        stratify_col = df['label']
    else:
        stratify_col = None

    # First split: train + val vs test
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        stratify=stratify_col,
        random_state=42
    )

    # Second split: train vs val
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)

    if stratify:
        stratify_col_train_val = train_val_df['label']
    else:
        stratify_col_train_val = None

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio_adjusted,
        stratify=stratify_col_train_val,
        random_state=42
    )

    print(f"\n📊 Dataset Split:")
    print(f"   Train:      {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"   Validation: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"   Test:       {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")

    return train_df, val_df, test_df


def copy_images_to_split(train_df, val_df, test_df):
    """
    Salin foto ke direktori train/val/test

    Args:
        train_df (pd.DataFrame): Data training
        val_df (pd.DataFrame): Data validasi
        test_df (pd.DataFrame): Data test
    """
    def copy_split(df, split_name):
        split_dir = GROUND_TRUTH_DIR / split_name

        for _, row in df.iterrows():
            src = Path(row['processed_path'])
            disease_dir = split_dir / row['disease'].lower()
            disease_dir.mkdir(exist_ok=True)

            dst = disease_dir / row['filename']
            shutil.copy2(src, dst)

        print(f"✅ Copied {len(df)} images to {split_name}/")

    copy_split(train_df, 'train')
    copy_split(val_df, 'val')
    copy_split(test_df, 'test')


def generate_class_weights(df):
    """
    Hitung class weights untuk imbalanced dataset

    Args:
        df (pd.DataFrame): Label dataframe

    Returns:
        dict: Class weights
    """
    class_counts = df['disease'].value_counts()
    total_samples = len(df)

    n_classes = len(class_counts)
    class_weights = {}

    for idx, (disease, count) in enumerate(class_counts.items()):
        weight = total_samples / (n_classes * count)
        class_weights[disease] = round(weight, 2)

    print("\n⚖️  Class Weights:")
    for disease, weight in class_weights.items():
        count = class_counts[disease]
        print(f"   {disease:10s}: weight={weight:.2f} (n={count})")

    return class_weights


def save_ground_truth_data(df, filename='labels.csv'):
    """
    Simpan label dataframe ke CSV

    Args:
        df (pd.DataFrame): Label dataframe
        filename (str): Nama file output
    """
    output_path = GROUND_TRUTH_DIR / filename
    df.to_csv(output_path, index=False)
    print(f"\n✅ Labels saved to: {output_path}")


def load_ground_truth_data(filename='labels.csv'):
    """
    Load label dataframe dari CSV

    Args:
        filename (str): Nama file label

    Returns:
        pd.DataFrame: Label dataframe
    """
    input_path = GROUND_TRUTH_DIR / filename

    if not input_path.exists():
        print(f"❌ Label file not found: {input_path}")
        return None

    df = pd.read_csv(input_path)
    print(f"✅ Loaded {len(df)} labels from {input_path}")

    return df


def generate_dataset_statistics(df):
    """
    Generate statistik dataset

    Args:
        df (pd.DataFrame): Label dataframe
    """
    print("\n" + "="*70)
    print("GROUND TRUTH DATASET STATISTICS")
    print("="*70)

    print(f"\n📊 Dataset Overview:")
    print(f"   Total samples: {len(df):,}")

    print(f"\n🦠 Disease Distribution:")
    disease_counts = df['disease'].value_counts()
    for disease, count in disease_counts.items():
        percentage = count / len(df) * 100
        print(f"   {disease:10s}: {count:5d} samples ({percentage:5.1f}%)")

    print(f"\n⚠️  Severity Distribution:")
    severity_counts = df['severity'].value_counts()
    for severity, count in severity_counts.items():
        percentage = count / len(df) * 100
        print(f"   {severity:10s}: {count:5d} samples ({percentage:5.1f}%)")

    print(f"\n📐 Image Statistics:")
    print(f"   Average width:  {df['width'].mean():.0f} pixels")
    print(f"   Average height: {df['height'].mean():.0f} pixels")
    print(f"   Avg file size:  {df['file_size'].mean() / 1024:.1f} KB")

    print("\n" + "="*70)


# ============================================
# MAIN SCRIPT
# ============================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Collect ground truth data for KarawangPadiGuard')
    parser.add_argument('--setup', action='store_true',
                       help='Setup directory structure')
    parser.add_argument('--process', type=str,
                       help='Process a single photo file')
    parser.add_argument('--disease', type=str, choices=['HEALTHY', 'BLAST', 'WERENG', 'BERCAK'],
                       help='Disease category')
    parser.add_argument('--severity', type=str, choices=['RINGAN', 'SEDANG', 'BERAT'],
                       help='Severity level')
    parser.add_argument('--split', action='store_true',
                       help='Split dataset into train/val/test')
    parser.add_argument('--stats', action='store_true',
                       help='Generate dataset statistics')

    args = parser.parse_args()

    if args.setup:
        setup_directories()

    elif args.process and args.disease and args.severity:
        photo_info = process_photo(args.process, args.disease, args.severity)

        if photo_info:
            # Load existing labels
            df = load_ground_truth_data()
            if df is None:
                df = pd.DataFrame([photo_info])
            else:
                df = pd.concat([df, pd.DataFrame([photo_info])], ignore_index=True)

            # Save updated labels
            save_ground_truth_data(df)
            print(f"✅ Photo processed and label added")

    elif args.split:
        # Load labels
        df = load_ground_truth_data()

        if df is not None and len(df) > 0:
            # Split dataset
            train_df, val_df, test_df = split_train_val_test(df)

            # Copy images
            copy_images_to_split(train_df, val_df, test_df)

            # Save split labels
            save_ground_truth_data(train_df, 'train_labels.csv')
            save_ground_truth_data(val_df, 'val_labels.csv')
            save_ground_truth_data(test_df, 'test_labels.csv')

            # Calculate class weights
            class_weights = generate_class_weights(train_df)

    elif args.stats:
        # Load and display statistics
        df = load_ground_truth_data()
        if df is not None:
            generate_dataset_statistics(df)

    else:
        print("Please specify an action. Use --help for more information.")
