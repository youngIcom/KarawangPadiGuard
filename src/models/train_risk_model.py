"""
Training Script for Risk Prediction Model (Tabular)
KarawangPadiGuard - Microsoft Elevate Datathon

Predicts disease risk based on weather patterns and historical data

Author: Yesaya Situmorang
Date: 2026-04-28
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from contextlib import nullcontext

# Machine Learning
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score
)
import xgboost as xgb

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Monitoring is optional so this script can run in a clean Google Colab runtime.
try:
    import mlflow
    import mlflow.sklearn
    import mlflow.xgboost
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

    class _NoOpMlflow:
        def set_tracking_uri(self, *args, **kwargs):
            pass

        def set_experiment(self, *args, **kwargs):
            pass

        def start_run(self, *args, **kwargs):
            return nullcontext()

        def log_metric(self, *args, **kwargs):
            pass

        def log_metrics(self, *args, **kwargs):
            pass

        def log_params(self, *args, **kwargs):
            pass

        def log_artifact(self, *args, **kwargs):
            pass

    mlflow = _NoOpMlflow()

# Configuration
CONFIG = {
    # Data paths
    'weather_data_path': './data/processed/weather_data.csv',
    'satellite_data_path': './data/processed/satellite_data.csv',
    'production_data_path': './data/processed/dataset_produksi_padi_karawang_cleaned.csv',
    'output_dir': './models',
    'logs_dir': './logs',

    # Model
    'model_name': 'xgboost_risk_prediction_v1',

    # Features
    'risk_threshold_humidity': 85,  # Above this humidity = high risk
    'risk_threshold_temp_min': 25,  # Min temp for disease
    'risk_threshold_temp_max': 32,  # Max temp for disease
    'risk_rainfall_threshold': 5,   # mm of rain

    # Risk categories
    'risk_categories': ['Low', 'Medium', 'High'],

    # Training
    'test_size': 0.2,
    'random_state': 42,
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_jobs': -1,
    'gpu_enabled': True,
    'predictor': 'auto',

    # Kaggle/Colab runtime defaults
    'kaggle_gpu_enabled': True,
    'kaggle_n_estimators': 300,
    'colab_n_estimators': 200
}

# Disease-specific conditions
DISEASE_CONDITIONS = {
    'Blast': {
        'temp_min': 25,
        'temp_max': 28,
        'humidity_min': 90,
        'name': 'Leaf Blast (Pyricularia oryzae)'
    },
    'Brown_Spot': {
        'temp_min': 28,
        'temp_max': 32,
        'humidity_min': 85,
        'name': 'Brown Spot (Cochliobolus miyabeanus)'
    },
    'Bacterial_Blight': {
        'temp_min': 25,
        'temp_max': 30,
        'humidity_min': 85,
        'name': 'Bacterial Leaf Blight (Xanthomonas oryzae)'
    },
    'Sheath_Blight': {
        'temp_min': 28,
        'temp_max': 32,
        'humidity_min': 90,
        'name': 'Sheath Blight (Rhizoctonia solani)'
    }
}


def is_azureml_run():
    """Detect whether the script runs inside Azure ML."""
    return os.environ.get("AZUREML_RUN_ID") is not None


def is_kaggle_environment():
    """Detect whether the script runs inside Kaggle."""
    return os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None or Path("/kaggle/input").exists()


def is_colab_environment():
    """Detect whether the script runs inside Google Colab."""
    return "COLAB_RELEASE_TAG" in os.environ or Path("/content").exists()


def setup_tracking():
    """Initialize tracking based on environment."""
    if not MLFLOW_AVAILABLE:
        print("MLflow is not installed. Continuing without experiment tracking.")
        return

    if is_azureml_run():
        print("Detected Azure ML environment. Using MLflow for tracking.")
    elif is_kaggle_environment():
        print("Detected Kaggle environment. Logging locally (Artifacts).")
    elif is_colab_environment():
        print("Detected Google Colab environment. Using local MLflow tracking.")
        mlflow.set_tracking_uri("file:./mlruns")
    else:
        print("Detected Local environment. Using local MLflow tracking.")
        mlflow.set_tracking_uri("file:./mlruns")
    
    try:
        mlflow.set_experiment('karawang-padi-guard-risk')
    except Exception as exc:
        print(f"Warning: MLflow experiment setup skipped: {exc}")


def safe_mlflow_log_metric(name, value):
    """Log a metric without letting tracking backend issues fail training."""
    if not MLFLOW_AVAILABLE:
        return
    try:
        mlflow.log_metric(name, value)
    except Exception as exc:
        print(f"Warning: MLflow metric logging skipped for {name}: {exc}")


def safe_mlflow_log_params(params):
    """Log params without letting tracking backend issues fail training."""
    if not MLFLOW_AVAILABLE:
        return
    try:
        mlflow.log_params(params)
    except Exception as exc:
        print(f"Warning: MLflow params logging skipped: {exc}")


def safe_mlflow_log_artifact(path):
    """Log artifacts best-effort; Azure ML still captures files under ./outputs."""
    if not MLFLOW_AVAILABLE:
        return
    try:
        mlflow.log_artifact(str(path))
    except Exception as exc:
        print(f"Warning: MLflow artifact logging skipped for {path}: {exc}")


def get_bool_env(var_name: str, default: bool) -> bool:
    """Read boolean environment variable with a safe default."""
    value = os.environ.get(var_name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


def auto_detect_kaggle_csv(input_root: Path, preferred_keywords):
    """Find a likely CSV file from /kaggle/input using keyword ranking."""
    if not input_root.exists():
        return None

    candidates = [path for path in input_root.rglob("*.csv") if path.is_file()]
    if not candidates:
        return None

    candidates.sort(
        key=lambda p: (
            not all(keyword in p.name.lower() for keyword in preferred_keywords),
            not any(keyword in str(p).lower() for keyword in preferred_keywords),
            len(p.parts),
            p.name.lower(),
        )
    )
    return candidates[0]


def first_existing_path(paths):
    """Return the first existing file path from an ordered candidate list."""
    for path in paths:
        path = Path(path)
        if path.exists() and path.is_file():
            return str(path)
    return None


def configure_colab_paths():
    """Configure deterministic Colab paths without scanning all Google Drive."""
    project_roots = [
        Path(os.environ["COLAB_PROJECT_ROOT"]) if os.environ.get("COLAB_PROJECT_ROOT") else None,
        Path("/content/drive/MyDrive/Datathon/KarawangPadiGuard"),
        Path("/content/drive/MyDrive/KarawangPadiGuard"),
        Path("/content/KarawangPadiGuard"),
        Path.cwd(),
    ]
    project_roots = [path for path in project_roots if path is not None]

    data_roots = []
    for root in project_roots:
        data_roots.extend([
            root / "data" / "preprocess",
            root / "data" / "processed",
            root / "data" / "raw",
        ])

    weather_override = os.environ.get("COLAB_WEATHER_DATA_PATH")
    satellite_override = os.environ.get("COLAB_SATELLITE_DATA_PATH")
    production_override = os.environ.get("COLAB_PRODUCTION_DATA_PATH")

    weather_path = weather_override or first_existing_path(
        [
            data_root / filename
            for data_root in data_roots
            for filename in [
                "weather_scraped.csv",
                "weather_real.csv",
                "bmkg_weather.csv",
                "openmeteo_weather.csv",
                "cuaca_karawang.csv",
                "weather_data.csv",
            ]
        ]
    )
    satellite_path = satellite_override or first_existing_path(
        [
            data_root / filename
            for data_root in data_roots
            for filename in [
                "sentinel2_indices.csv",
                "sentinel_2_indices.csv",
                "satellite_scraped.csv",
                "satellite_real.csv",
                "gee_satellite_indices.csv",
                "satellite_data.csv",
            ]
        ]
    )
    production_path = production_override or first_existing_path(
        [
            data_root / filename
            for data_root in data_roots
            for filename in [
                "dataset_produksi_padi_karawang_cleaned.csv",
                "produksi_padi_karawang.csv",
            ]
        ]
    )

    if weather_path:
        CONFIG["weather_data_path"] = weather_path
    if satellite_path:
        CONFIG["satellite_data_path"] = satellite_path
    if production_path:
        CONFIG["production_data_path"] = production_path

    output_root = next((root for root in project_roots if root.exists()), Path("/content"))
    CONFIG["output_dir"] = os.environ.get("COLAB_OUTPUT_DIR", str(output_root / "models"))
    CONFIG["logs_dir"] = os.environ.get("COLAB_LOGS_DIR", str(output_root / "logs"))
    CONFIG["gpu_enabled"] = get_bool_env("COLAB_GPU_ENABLED", False)
    CONFIG["n_estimators"] = int(os.environ.get("COLAB_N_ESTIMATORS", CONFIG["colab_n_estimators"]))


def configure_runtime():
    """Configure paths and model defaults for Colab/Kaggle/local runtimes."""
    print("=" * 60)
    print("RUNTIME CONFIGURATION")
    print("=" * 60)

    azure_mode = is_azureml_run()
    kaggle_mode = is_kaggle_environment()
    colab_mode = is_colab_environment() and not kaggle_mode and not azure_mode

    if azure_mode:
        CONFIG['output_dir'] = os.environ.get("AZUREML_OUTPUT_DIR", "./outputs/models")
        CONFIG['logs_dir'] = os.environ.get("AZUREML_LOGS_DIR", "./outputs/logs")
        CONFIG['gpu_enabled'] = get_bool_env("AZUREML_GPU_ENABLED", False)
        CONFIG['n_estimators'] = int(os.environ.get("AZUREML_N_ESTIMATORS", CONFIG['n_estimators']))
        print("Azure ML mode detected. Writing artifacts under ./outputs.")

    if colab_mode:
        configure_colab_paths()
        print("Google Colab mode detected. Applied deterministic Drive/content paths.")

    if kaggle_mode:
        input_root = Path("/kaggle/input")
        weather_override = os.environ.get("KAGGLE_WEATHER_DATA_PATH")
        satellite_override = os.environ.get("KAGGLE_SATELLITE_DATA_PATH")
        production_override = os.environ.get("KAGGLE_PRODUCTION_DATA_PATH")

        if weather_override:
            CONFIG['weather_data_path'] = weather_override
        elif CONFIG['weather_data_path'].startswith("./"):
            weather_detected = auto_detect_kaggle_csv(input_root, ("weather", "cuaca"))
            if weather_detected is not None:
                CONFIG['weather_data_path'] = str(weather_detected)

        if satellite_override:
            CONFIG['satellite_data_path'] = satellite_override
        elif CONFIG['satellite_data_path'].startswith("./"):
            satellite_detected = auto_detect_kaggle_csv(
                input_root, ("satellite", "sentinel", "ndvi")
            )
            if satellite_detected is not None:
                CONFIG['satellite_data_path'] = str(satellite_detected)

        if production_override:
            CONFIG['production_data_path'] = production_override
        elif CONFIG['production_data_path'].startswith("./"):
            production_detected = auto_detect_kaggle_csv(
                input_root, ("produksi", "production", "karawang")
            )
            if production_detected is not None:
                CONFIG['production_data_path'] = str(production_detected)

        CONFIG['output_dir'] = os.environ.get("KAGGLE_OUTPUT_DIR", "/kaggle/working/models")
        CONFIG['logs_dir'] = os.environ.get("KAGGLE_LOGS_DIR", "/kaggle/working/logs")
        CONFIG['gpu_enabled'] = get_bool_env("KAGGLE_GPU_ENABLED", CONFIG['kaggle_gpu_enabled'])
        CONFIG['n_estimators'] = int(os.environ.get("KAGGLE_N_ESTIMATORS", CONFIG['kaggle_n_estimators']))
        print("Kaggle mode detected. Applied T4 defaults.")

    print(f"Weather data path: {CONFIG['weather_data_path']}")
    print(f"Satellite data path: {CONFIG['satellite_data_path']}")
    print(f"Production data path: {CONFIG['production_data_path']}")
    print(f"Output dir: {CONFIG['output_dir']}")
    print(f"Logs dir: {CONFIG['logs_dir']}")
    print(f"GPU enabled: {CONFIG['gpu_enabled']}")
    print(f"n_estimators: {CONFIG['n_estimators']}")


def create_output_directories():
    """Create output directories if they don't exist"""
    Path(CONFIG['output_dir']).mkdir(parents=True, exist_ok=True)
    Path(CONFIG['logs_dir']).mkdir(parents=True, exist_ok=True)


def load_and_prepare_data():
    """
    Load weather and satellite data and prepare features
    """
    print("=" * 60)
    print("LOADING AND PREPARING DATA (Weather + Satellite)")
    print("=" * 60)

    # Load weather data
    weather_path = Path(CONFIG['weather_data_path'])
    if not weather_path.exists():
        raise FileNotFoundError(f"Weather data not found at {weather_path}")
    
    df_weather = pd.read_csv(weather_path)
    df_weather['date'] = pd.to_datetime(df_weather['date'])
    
    # Load satellite data
    sat_path = Path(CONFIG['satellite_data_path'])
    if not sat_path.exists():
        raise FileNotFoundError(
            f"Satellite data not found at {sat_path}. "
            "Risk training requires weather + satellite features."
        )

    df_sat = pd.read_csv(sat_path)
    df_sat['date'] = pd.to_datetime(df_sat['date'])
    
    # Merge weather and satellite (weekly to daily)
    df = pd.merge_asof(
        df_weather.sort_values('date'),
        df_sat.sort_values('date'),
        on='date',
        direction='backward'
    )
    print(f"Merged satellite data: {df_sat.shape} records")

    df = df.sort_values('date').reset_index(drop=True)
    
    # Fill missing satellite data (at the beginning)
    if 'ndvi' in df.columns:
        df[['ndvi', 'ndwi', 'evi', 'savi']] = df[['ndvi', 'ndwi', 'evi', 'savi']].bfill()

    print(f"Loaded combined data: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Feature engineering
    df = engineer_features(df)

    # Calculate risk labels
    df = calculate_risk_labels(df)

    print(f"\nFinal dataset shape: {df.shape}")
    return df


def engineer_features(df):
    """
    Engineer features for risk prediction
    """
    print("\nEngineering features...")

    # Temporal features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(np.int32)
    df['season'] = df['month'].map({
        12: 'Rainy', 1: 'Rainy', 2: 'Rainy', 3: 'Rainy', 4: 'Rainy',
        5: 'Dry', 6: 'Dry', 7: 'Dry', 8: 'Dry', 9: 'Dry',
        10: 'Transitional', 11: 'Transitional'
    })

    # Lag features (previous days)
    for lag in [1, 3, 7]:
        df[f'temp_lag_{lag}'] = df['temperature'].shift(lag)
        df[f'humidity_lag_{lag}'] = df['humidity'].shift(lag)
        df[f'rainfall_lag_{lag}'] = df['rainfall'].shift(lag)

    # Rolling averages
    for window in [3, 7, 14]:
        df[f'temp_rolling_{window}'] = df['temperature'].rolling(window=window).mean()
        df[f'humidity_rolling_{window}'] = df['humidity'].rolling(window=window).mean()
        df[f'rainfall_rolling_{window}'] = df['rainfall'].rolling(window=window).sum()

    # Weather interactions
    df['temp_humidity_interaction'] = df['temperature'] * df['humidity']
    df['rain_intensity'] = pd.cut(df['rainfall'], bins=[-1, 0, 5, 20, float('inf')],
                                   labels=[0, 1, 2, 3]).astype(int)

    # Disease-specific indicators
    df['blast_favorable'] = (
        (df['temperature'] >= 25) & (df['temperature'] <= 28) &
        (df['humidity'] >= 90)
    ).astype(int)

    df['brown_spot_favorable'] = (
        (df['temperature'] >= 28) & (df['temperature'] <= 32) &
        (df['humidity'] >= 85)
    ).astype(int)

    # Cumulative rainfall (last 7 days)
    df['rainfall_7day_cum'] = df['rainfall'].rolling(7).sum()

    # Temperature trend
    df['temp_trend_3d'] = df['temperature'].diff(3)
    df['humidity_trend_3d'] = df['humidity'].diff(3)

    # Extreme conditions
    df['extreme_heat'] = (df['temperature'] > 32).astype(int)
    df['extreme_humidity'] = (df['humidity'] > 95).astype(int)
    df['heavy_rain'] = (df['rainfall'] > 20).astype(int)

    # Drop rows with NaN (from lag features)
    df = df.dropna()

    # Encode categorical
    le = LabelEncoder()
    df['season_encoded'] = le.fit_transform(df['season'])

    print(f"Features created: {df.shape[1]}")

    return df


def calculate_risk_labels(df):
    """
    Calculate risk labels based on weather conditions and disease patterns
    """
    print("\nCalculating risk labels...")

    conditions = []

    for _, row in df.iterrows():
        risk_score = 0

        # Check each disease condition
        for disease, condition in DISEASE_CONDITIONS.items():
            if (condition['temp_min'] <= row['temperature'] <= condition['temp_max'] and
                row['humidity'] >= condition['humidity_min']):
                risk_score += 1

        # Additional risk factors
        if row['rainfall_7day_cum'] > 30:
            risk_score += 1
        if row['humidity_rolling_7'] > 85:
            risk_score += 1

        # Categorize risk
        if risk_score >= 4:
            conditions.append('High')
        elif risk_score >= 2:
            conditions.append('Medium')
        else:
            conditions.append('Low')

    df['risk_category'] = conditions
    df['risk_score'] = [CONFIG['risk_categories'].index(c) for c in conditions]

    return df


def prepare_features_and_target(df):
    """
    Prepare feature matrix and target vector
    """
    print("\nPreparing features and target...")

    # Select features
    feature_cols = [
        # Current weather
        'temperature', 'humidity', 'rainfall', 'wind_speed', 'cloud_cover',
        
        # Satellite Indices (Multimodal)
        'ndvi', 'ndwi', 'evi', 'savi',

        # Temporal
        'month', 'day_of_year', 'week_of_year', 'season_encoded',

        # Lag features
        'temp_lag_1', 'humidity_lag_1', 'rainfall_lag_1',
        'temp_lag_3', 'humidity_lag_3', 'rainfall_lag_3',
        'temp_lag_7', 'humidity_lag_7', 'rainfall_lag_7',

        # Rolling features
        'temp_rolling_3', 'humidity_rolling_3', 'rainfall_rolling_3',
        'temp_rolling_7', 'humidity_rolling_7', 'rainfall_rolling_7',
        'temp_rolling_14', 'humidity_rolling_14', 'rainfall_rolling_14',

        # Interactions
        'temp_humidity_interaction', 'rain_intensity',
        'rainfall_7day_cum', 'temp_trend_3d', 'humidity_trend_3d',

        # Disease indicators
        'blast_favorable', 'brown_spot_favorable',

        # Extremes
        'extreme_heat', 'extreme_humidity', 'heavy_rain'
    ]

    X = df[feature_cols].copy()
    y = df['risk_score'].copy()

    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")

    return X, y, feature_cols


def split_data(X, y):
    """
    Split data into train and test sets (time-based)
    """
    print("\nSplitting data...")

    # Time-based split (use last 20% for testing)
    split_idx = int(len(X) * (1 - CONFIG['test_size']))

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    print(f"Train set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_xgboost_model(X_train, y_train):
    """
    Train XGBoost classifier
    """
    print("\n" + "=" * 60)
    print("TRAINING XGBOOST MODEL")
    print("=" * 60)

    xgb_params = {
        'n_estimators': CONFIG['n_estimators'],
        'max_depth': CONFIG['max_depth'],
        'learning_rate': CONFIG['learning_rate'],
        'random_state': CONFIG['random_state'],
        'objective': 'multi:softmax',
        'num_class': 3,
        'eval_metric': 'mlogloss',
        'n_jobs': CONFIG.get('n_jobs', -1),
        'predictor': CONFIG.get('predictor', 'auto')
    }

    if CONFIG.get('gpu_enabled', True):
        # XGBoost 2.0+ uses 'device' parameter
        xgb_params.update({
            'tree_method': 'hist',
            'device': 'cuda'
        })
        print("Using GPU acceleration for XGBoost (device=cuda)")
    else:
        xgb_params.update({'tree_method': 'hist'})
        print("Using CPU training for XGBoost")

    model = xgb.XGBClassifier(**xgb_params)

    try:
        model.fit(
            X_train,
            y_train,
            verbose=False
        )
    except xgb.core.XGBoostError as exc:
        if CONFIG.get('gpu_enabled', True):
            print(f"GPU training failed: {exc}")
            print("Retrying XGBoost training on CPU.")
            xgb_params.pop('device', None)
            xgb_params['tree_method'] = 'hist'
            CONFIG['gpu_enabled'] = False
            model = xgb.XGBClassifier(**xgb_params)
            model.fit(
                X_train,
                y_train,
                verbose=False
            )
        else:
            raise

    print("Model trained successfully!")
    print(f"Training iterations: {model.n_estimators}")

    return model


def evaluate_model(model, X_test, y_test, feature_names):
    """
    Evaluate model performance
    """
    print("\n" + "=" * 60)
    print("EVALUATING MODEL")
    print("=" * 60)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("\nOverall Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

    # Log metrics to MLflow
    safe_mlflow_log_metric('test_accuracy', accuracy)
    safe_mlflow_log_metric('test_precision', precision)
    safe_mlflow_log_metric('test_recall', recall)
    safe_mlflow_log_metric('test_f1_score', f1)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=CONFIG['risk_categories'],
        yticklabels=CONFIG['risk_categories']
    )
    plt.title('Confusion Matrix - Risk Prediction')
    plt.ylabel('True Risk')
    plt.xlabel('Predicted Risk')
    plt.tight_layout()

    cm_path = Path(CONFIG['output_dir']) / f"{CONFIG['model_name']}_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    safe_mlflow_log_artifact(cm_path)
    
    print(f"\nConfusion matrix saved to: {cm_path}")

    # Feature importance
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(data=feature_importance.head(20), x='importance', y='feature')
    plt.title('Top 20 Feature Importances')
    plt.xlabel('Importance')
    plt.tight_layout()

    imp_path = Path(CONFIG['output_dir']) / f"{CONFIG['model_name']}_feature_importance.png"
    plt.savefig(imp_path, dpi=150)
    safe_mlflow_log_artifact(imp_path)
    
    print(f"Feature importance plot saved to: {imp_path}")

    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))

    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist()
    }


def save_model_and_artifacts(model, scaler, metrics, feature_names):
    """
    Save model and artifacts
    """
    print("\n" + "=" * 60)
    print("SAVING MODEL AND ARTIFACTS")
    print("=" * 60)

    import joblib

    # Save model
    model_path = Path(CONFIG['output_dir']) / f"{CONFIG['model_name']}.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    # Save scaler
    scaler_path = Path(CONFIG['output_dir']) / f"{CONFIG['model_name']}_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to: {scaler_path}")

    # Save feature names
    feature_path = Path(CONFIG['output_dir']) / f"{CONFIG['model_name']}_features.json"
    with open(feature_path, 'w') as f:
        json.dump(feature_names, f, indent=2)
    print(f"Feature names saved to: {feature_path}")
    
    # Save config
    config_path = Path(CONFIG['output_dir']) / f"{CONFIG['model_name']}_config.json"
    with open(config_path, 'w') as f:
        json.dump(CONFIG, f, indent=2)
    print(f"Config saved to: {config_path}")
    
    # Save metrics
    metrics_with_timestamp = {
        'timestamp': datetime.now().isoformat(),
        'model_name': CONFIG['model_name'],
        'metrics': metrics,
        'config': CONFIG,
        'disease_conditions': DISEASE_CONDITIONS
    }

    metrics_path = Path(CONFIG['output_dir']) / f"{CONFIG['model_name']}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_with_timestamp, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")


def predict_future_risk(model, scaler, df, days=7):
    """
    Predict risk for the next N days
    """
    print("\n" + "=" * 60)
    print(f"PREDICTING RISK FOR NEXT {days} DAYS")
    print("=" * 60)

    # Get the last row for forecasting
    last_row = df.iloc[[-1]].copy()

    predictions = []
    current_date = last_row['date'].iloc[0]

    for day in range(1, days + 1):
        future_date = current_date + pd.Timedelta(days=day)
        predictions.append({
            'date': future_date.strftime('%Y-%m-%d'),
            'predicted_risk': 'Predicted via Model'
        })

    print(f"\nNext {days} days forecast generated")
    return predictions


def main():
    """Main training pipeline"""
    print("\n" + "=" * 60)
    print("KARAWANG PADI GUARD - RISK PREDICTION TRAINING")
    print("=" * 60)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    setup_tracking()

    with mlflow.start_run(run_name=CONFIG['model_name']):
        try:
            configure_runtime()
            create_output_directories()
            
            # Log params
            safe_mlflow_log_params(CONFIG)

            df = load_and_prepare_data()
            X, y, feature_names = prepare_features_and_target(df)
            X_train, X_test, y_train, y_test, scaler = split_data(X, y)

            model = train_xgboost_model(X_train, y_train)
            metrics = evaluate_model(model, X_test, y_test, feature_names)

            # Log metrics manually
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    safe_mlflow_log_metric(k, v)

            save_model_and_artifacts(model, scaler, metrics, feature_names)
            
            # Log artifacts to MLflow
            model_path = Path(CONFIG['output_dir']) / f"{CONFIG['model_name']}.pkl"
            safe_mlflow_log_artifact(model_path)
            
            print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("\nTraining completed successfully!")
            
        except Exception as e:
            print(f"\nError during training: {str(e)}")
            raise


if __name__ == "__main__":
    main()
