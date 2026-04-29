"""
Training Script for Rice Disease Detection (Computer Vision)
KarawangPadiGuard - Microsoft Elevate Datathon

Author: Yesaya Situmorang
Date: 2026-04-28
"""

import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, applications, mixed_precision
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
)
from tensorflow.keras.optimizers import Adam

# Scikit-learn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Monitoring
import wandb

# Configuration
CONFIG = {
    # Data paths
    'dataset_path': './data/raw/rice_disease_dataset',
    'output_dir': './models',
    'logs_dir': './logs',

    # Model
    'model_name': 'mobilenetv3_rice_disease_v1',
    'base_model': 'MobileNetV3Small',
    'img_size': (224, 224),
    'num_classes': 6,

    # Training
    'batch_size': 32,
    'kaggle_batch_size': 64,
    'epochs': 30,
    'learning_rate': 1e-3,
    'fine_tune_learning_rate': 1e-4,
    'validation_split': 0.2,
    'test_split': 0.1,
    'fine_tune_epochs': 10,
    'fine_tune_trainable_layers': 30,
    'cache_dataset': False,
    'kaggle_cache_dataset': True,
    'cache_dir': None,
    'shuffle_buffer_size': 2048,
    'enable_xla': True,
    'use_mixed_precision': True,

    # Class names
    'class_names': [
        'Bacterial Leaf Blight',
        'Brown Spot',
        'Healthy Rice Leaf',
        'Leaf Blast',
        'Leaf scald',
        'Sheath Blight'
    ],

    # Class names in Indonesian
    'class_names_id': [
        'Hawar Daun Bakteri',
        'Bercak Coklat',
        'Daun Sehat',
        'Blas Daun',
        'Hawar Seludang',
        'Blas Seludang'
    ]
}

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}


class WandbCallback(keras.callbacks.Callback):
    """Custom callback to log metrics to Weights & Biases"""
    def on_epoch_end(self, epoch, logs=None):
        if logs:
            wandb.log({
                'epoch': epoch,
                'train_loss': logs.get('loss'),
                'train_accuracy': logs.get('accuracy'),
                'train_precision': logs.get('precision'),
                'train_recall': logs.get('recall'),
                'train_auc': logs.get('auc'),
                'val_loss': logs.get('val_loss'),
                'val_accuracy': logs.get('val_accuracy'),
                'val_precision': logs.get('val_precision'),
                'val_recall': logs.get('val_recall'),
                'val_auc': logs.get('val_auc'),
            })


def get_bool_env(var_name: str, default: bool) -> bool:
    """Read boolean environment variable with a safe default."""
    value = os.environ.get(var_name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


def is_kaggle_environment():
    """Detect whether the script is running inside Kaggle."""
    return os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None or Path("/kaggle/input").exists()


def resolve_runtime_paths():
    """Adjust default paths so the script runs cleanly on Kaggle and locally."""
    if is_kaggle_environment():
        input_root = Path("/kaggle/input")
        dataset_override = os.environ.get("KAGGLE_DATASET_PATH")

        if dataset_override:
            CONFIG['dataset_path'] = dataset_override
        elif CONFIG['dataset_path'].startswith("./") and input_root.exists():
            inferred_dataset = auto_detect_kaggle_dataset(input_root)
            if inferred_dataset is not None:
                CONFIG['dataset_path'] = str(inferred_dataset)

        CONFIG['output_dir'] = os.environ.get("KAGGLE_OUTPUT_DIR", "/kaggle/working/models")
        CONFIG['logs_dir'] = os.environ.get("KAGGLE_LOGS_DIR", "/kaggle/working/logs")


def auto_detect_kaggle_dataset(input_root: Path):
    """Try to locate a plausible class-folder dataset without expensive deep scans."""
    if not input_root.exists():
        return None

    candidates = []
    scanned = set()

    def register_candidate(candidate: Path):
        candidate_resolved = candidate.resolve()
        if candidate_resolved in scanned:
            return
        scanned.add(candidate_resolved)

        if not candidate.is_dir():
            return

        class_dirs = [child for child in candidate.iterdir() if child.is_dir()]
        if len(class_dirs) < 2:
            return

        class_dirs_with_images = 0
        for class_dir in class_dirs:
            has_image = any(
                img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS
                for img_path in class_dir.rglob("*")
            )
            if has_image:
                class_dirs_with_images += 1
            if class_dirs_with_images >= 2:
                candidates.append(candidate)
                return

    for path in input_root.iterdir():
        if not path.is_dir():
            continue

        register_candidate(path)
        for level_1 in [d for d in path.iterdir() if d.is_dir()]:
            register_candidate(level_1)
            for level_2 in [d for d in level_1.iterdir() if d.is_dir()]:
                register_candidate(level_2)

    if not candidates:
        return None

    preferred_keywords = ("rice", "padi", "disease", "leaf", "blast")
    candidates.sort(
        key=lambda p: (
            not any(keyword in str(p).lower() for keyword in preferred_keywords),
            len(p.parts),
        )
    )
    return candidates[0]


def configure_runtime():
    """Enable GPU-friendly settings for Kaggle / T4 training."""
    print("=" * 60)
    print("RUNTIME CONFIGURATION")
    print("=" * 60)

    kaggle_mode = is_kaggle_environment()
    resolve_runtime_paths()

    if kaggle_mode:
        CONFIG['batch_size'] = int(os.environ.get("KAGGLE_BATCH_SIZE", CONFIG['kaggle_batch_size']))
        CONFIG['cache_dataset'] = get_bool_env("KAGGLE_CACHE_DATASET", CONFIG['kaggle_cache_dataset'])
        CONFIG['cache_dir'] = os.environ.get("KAGGLE_CACHE_DIR", "/kaggle/working/tf_cache")
        print("Kaggle mode detected. Applied T4 defaults.")

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU detected: {len(gpus)}")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as err:
                print(f"Could not set memory growth for {gpu.name}: {err}")

        if CONFIG.get('enable_xla', True):
            tf.config.optimizer.set_jit(True)
            print("XLA JIT: enabled")

        if CONFIG.get('use_mixed_precision', True):
            mixed_precision.set_global_policy('mixed_float16')
            print(f"Mixed precision policy: {mixed_precision.global_policy()}")
    else:
        print("GPU not detected. Training will run on CPU.")

    print(f"Dataset path: {CONFIG['dataset_path']}")
    print(f"Output dir: {CONFIG['output_dir']}")
    print(f"Logs dir: {CONFIG['logs_dir']}")
    print(f"Batch size: {CONFIG['batch_size']}")
    print(f"Cache dataset: {CONFIG['cache_dataset']}")


def create_output_directories():
    """Create output directories if they don't exist"""
    Path(CONFIG['output_dir']).mkdir(parents=True, exist_ok=True)
    Path(CONFIG['logs_dir']).mkdir(parents=True, exist_ok=True)


def list_images_with_labels(dataset_path: Path) -> Tuple[List[str], List[int], List[str]]:
    """Collect image paths and integer labels from a class-per-folder dataset."""
    class_dirs = sorted([path for path in dataset_path.iterdir() if path.is_dir()])
    if not class_dirs:
        raise ValueError(f"No class folders found under {dataset_path}")

    class_names = [path.name for path in class_dirs]
    image_paths = []
    labels = []

    for class_idx, class_dir in enumerate(class_dirs):
        class_images = sorted(
            [
                str(path)
                for path in class_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ]
        )
        image_paths.extend(class_images)
        labels.extend([class_idx] * len(class_images))

    if not image_paths:
        raise ValueError(f"No images found under {dataset_path}")

    return image_paths, labels, class_names


def decode_and_resize_image(file_path, label):
    """Load an image file and convert label to one-hot."""
    image = tf.io.read_file(file_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, CONFIG['img_size'])
    image = tf.cast(image, tf.float32)
    label = tf.one_hot(label, depth=CONFIG['num_classes'])
    return image, label


def build_tf_dataset(file_paths, labels, training=False, split_name='dataset'):
    """Build a tf.data pipeline from file paths."""
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    if training:
        shuffle_buffer = min(len(file_paths), int(CONFIG.get('shuffle_buffer_size', 2048)))
        dataset = dataset.shuffle(buffer_size=shuffle_buffer, seed=42, reshuffle_each_iteration=True)

    options = tf.data.Options()
    options.experimental_deterministic = not training
    dataset = dataset.with_options(options)

    dataset = dataset.map(
        decode_and_resize_image,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not training
    )

    if CONFIG.get('cache_dataset', False):
        cache_dir = CONFIG.get('cache_dir')
        if cache_dir:
            cache_path = Path(cache_dir) / f"{split_name}.cache"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            dataset = dataset.cache(str(cache_path))
        else:
            dataset = dataset.cache()

    dataset = dataset.batch(CONFIG['batch_size'])
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def load_data():
    """Load and prepare image dataset."""
    print("=" * 60)
    print("LOADING DATASET")
    print("=" * 60)

    dataset_path = Path(CONFIG['dataset_path'])

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. "
            "For Kaggle, attach the dataset and optionally set KAGGLE_DATASET_PATH."
        )

    has_split = all((dataset_path / s).exists() for s in ['train', 'val', 'test'])

    # Drill down logic: If the path contains only one directory and no images, 
    # the real dataset is likely one level deeper (common in Kaggle/Zips)
    if not has_split:
        class_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
        if len(class_dirs) == 1:
            sub_dirs = [d for d in class_dirs[0].iterdir() if d.is_dir()]
            if len(sub_dirs) >= 2:
                print(f"ℹ️ Single parent directory '{class_dirs[0].name}' detected. Drilling down to sub-folders...")
                dataset_path = class_dirs[0]

    if has_split:
        print("Dataset already has train/val/test split")

        train_files, train_labels, class_names = list_images_with_labels(dataset_path / 'train')
        val_files, val_labels, _ = list_images_with_labels(dataset_path / 'val')
        test_files, test_labels, _ = list_images_with_labels(dataset_path / 'test')
    else:
        print("Dataset not split. Creating stratified train/val/test split...")

        all_files, all_labels, class_names = list_images_with_labels(dataset_path)

        train_files, holdout_files, train_labels, holdout_labels = train_test_split(
            all_files,
            all_labels,
            test_size=CONFIG['validation_split'] + CONFIG['test_split'],
            random_state=42,
            stratify=all_labels
        )

        relative_test_size = CONFIG['test_split'] / (CONFIG['validation_split'] + CONFIG['test_split'])
        val_files, test_files, val_labels, test_labels = train_test_split(
            holdout_files,
            holdout_labels,
            test_size=relative_test_size,
            random_state=42,
            stratify=holdout_labels
        )

    CONFIG['class_names'] = class_names
    CONFIG['num_classes'] = len(class_names)
    print(f"Classes found: {CONFIG['class_names']}")

    train_ds = build_tf_dataset(train_files, train_labels, training=True, split_name='train')
    val_ds = build_tf_dataset(val_files, val_labels, training=False, split_name='val')
    test_ds = build_tf_dataset(test_files, test_labels, training=False, split_name='test')

    print(f"\nTrain samples: {len(train_files)}")
    print(f"Val samples: {len(val_files)}")
    print(f"Test samples: {len(test_files)}")

    return train_ds, val_ds, test_ds


def build_model(num_classes):
    """
    Build model using transfer learning with MobileNetV3
    """
    print("=" * 60)
    print("BUILDING MODEL")
    print("=" * 60)

    # Data augmentation
    data_augmentation = models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name="data_augmentation")

    # Load pre-trained model
    base_model = applications.MobileNetV3Small(
        input_shape=(*CONFIG['img_size'], 3),
        include_top=False,
        weights='imagenet',
        include_preprocessing=True
    )

    # Freeze base model
    base_model.trainable = False

    # Build model
    inputs = layers.Input(shape=(*CONFIG['img_size'], 3))
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)

    model = models.Model(inputs, outputs, name=CONFIG['model_name'])

    print(f"\nModel: {CONFIG['model_name']}")
    print(f"Base: {CONFIG['base_model']}")
    print(f"Input shape: {(*CONFIG['img_size'], 3)}")
    print(f"Output classes: {num_classes}")

    model.summary()

    return model, base_model


def compile_model(model, learning_rate=None):
    """Compile model with optimizer and metrics"""
    optimizer = Adam(learning_rate=learning_rate or CONFIG['learning_rate'])

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )

    return model


def create_callbacks():
    """Create training callbacks"""
    checkpoint_path = Path(CONFIG['output_dir']) / f"{CONFIG['model_name']}_best.keras"
    csv_path = Path(CONFIG['logs_dir']) / f"{CONFIG['model_name']}_training_log.csv"

    callbacks = [
        # Save best model
        ModelCheckpoint(
            str(checkpoint_path),
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),

        # Early stopping
        EarlyStopping(
            monitor='val_accuracy',
            patience=8,
            mode='max',
            verbose=1,
            restore_best_weights=True
        ),

        # Reduce learning rate
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1
        ),

        # CSV logger
        CSVLogger(str(csv_path)),
        
        # Custom W&B Callback
        WandbCallback()
    ]

    return callbacks


def train_model(model, train_ds, val_ds):
    """
    Train the model
    """
    print("=" * 60)
    print("TRAINING MODEL")
    print("=" * 60)

    callbacks = create_callbacks()
    
    # Log training config to W&B
    wandb.config.update({
        'learning_rate': CONFIG['learning_rate'],
        'batch_size': CONFIG['batch_size'],
        'epochs': CONFIG['epochs'],
        'model_name': CONFIG['model_name'],
        'base_model': CONFIG['base_model']
    })

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=CONFIG['epochs'],
        callbacks=callbacks,
        verbose=1
    )

    return history


def fine_tune_model(model, base_model, train_ds, val_ds):
    """
    Fine-tune the model by unfreezing some layers
    """
    print("=" * 60)
    print("FINE-TUNING MODEL")
    print("=" * 60)

    # Unfreeze the base model
    base_model.trainable = True

    # Freeze all layers except the last N
    for layer in base_model.layers[:-CONFIG['fine_tune_trainable_layers']]:
        layer.trainable = False

    # Recompile with lower learning rate
    model = compile_model(model, learning_rate=CONFIG['fine_tune_learning_rate'])

    # Fine-tune
    initial_epoch = CONFIG['epochs']
    total_epochs = CONFIG['epochs'] + CONFIG['fine_tune_epochs']
    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=initial_epoch,
        epochs=total_epochs,
        callbacks=create_callbacks(),
        verbose=1
    )

    return history_fine


def evaluate_model(model, test_ds):
    """
    Evaluate model on test set
    """
    print("=" * 60)
    print("EVALUATING MODEL")
    print("=" * 60)

    # Get predictions
    y_pred_probs = model.predict(test_ds)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Get true labels
    y_true = np.concatenate([y for x, y in test_ds], axis=0)
    y_true = np.argmax(y_true, axis=1)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(
        y_true,
        y_pred,
        target_names=CONFIG['class_names'],
        digits=4
    ))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=CONFIG['class_names'],
        yticklabels=CONFIG['class_names']
    )
    plt.title('Confusion Matrix - Rice Disease Detection')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    cm_path = Path(CONFIG['output_dir']) / f"{CONFIG['model_name']}_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    print(f"\nConfusion matrix saved to: {cm_path}")
    
    # Log confusion matrix to W&B
    wandb.log({'confusion_matrix': wandb.Image(str(cm_path))})

    # Calculate metrics
    test_loss, test_acc, test_precision, test_recall, test_auc = model.evaluate(test_ds, verbose=0)

    print("\n" + "=" * 60)
    print("TEST METRICS")
    print("=" * 60)
    print(f"Test Accuracy:  {test_acc:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall:    {test_recall:.4f}")
    print(f"Test AUC:       {test_auc:.4f}")
    f1_score = (
        2 * (test_precision * test_recall) / (test_precision + test_recall)
        if (test_precision + test_recall) > 0 else 0.0
    )
    print(f"Test F1-Score:  {f1_score:.4f}")
    print("=" * 60)
    
    # Log test metrics to W&B
    wandb.log({
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_auc': test_auc,
        'test_f1_score': f1_score
    })

    return {
        'accuracy': float(test_acc),
        'precision': float(test_precision),
        'recall': float(test_recall),
        'auc': float(test_auc),
        'f1_score': float(f1_score)
    }


def save_model_and_artifacts(model, metrics):
    """
    Save model, config, and metrics
    """
    print("\n" + "=" * 60)
    print("SAVING MODEL AND ARTIFACTS")
    print("=" * 60)

    # Save final model
    final_model_path = Path(CONFIG['output_dir']) / f"{CONFIG['model_name']}_final.keras"
    model.save(final_model_path)
    print(f"Model saved to: {final_model_path}")
    
    # Log model to W&B
    wandb.save(str(final_model_path))

    # Save as TensorFlow SavedModel format
    saved_model_path = Path(CONFIG['output_dir']) / f"{CONFIG['model_name']}_savedmodel"
    try:
        model.export(saved_model_path)
    except AttributeError:
        tf.saved_model.save(model, saved_model_path)
    print(f"SavedModel exported to: {saved_model_path}")
    
    # Log SavedModel to W&B
    wandb.save(str(saved_model_path))

    # Save config
    config_path = Path(CONFIG['output_dir']) / f"{CONFIG['model_name']}_config.json"
    with open(config_path, 'w') as f:
        json.dump(CONFIG, f, indent=2)
    print(f"Config saved to: {config_path}")
    
    # Log config to W&B
    wandb.save(str(config_path))

    # Save metrics
    metrics_with_timestamp = {
        'timestamp': datetime.now().isoformat(),
        'model_name': CONFIG['model_name'],
        'metrics': metrics,
        'config': CONFIG
    }

    metrics_path = Path(CONFIG['output_dir']) / f"{CONFIG['model_name']}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_with_timestamp, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")
    
    # Log final metrics to W&B
    wandb.log({
        'final_metrics': metrics,
        'timestamp': datetime.now().isoformat()
    })

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)


def plot_training_history(history, history_fine=None):
    """
    Plot and save training history
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    if history_fine:
        axes[0].plot(
            range(len(history.history['accuracy']), len(history.history['accuracy']) + len(history_fine.history['accuracy'])),
            history_fine.history['accuracy'],
            label='Fine-tune Train Accuracy',
            linestyle='--'
        )
        axes[0].plot(
            range(len(history.history['accuracy']), len(history.history['accuracy']) + len(history_fine.history['accuracy'])),
            history_fine.history['val_accuracy'],
            label='Fine-tune Val Accuracy',
            linestyle='--'
        )
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    if history_fine:
        axes[1].plot(
            range(len(history.history['loss']), len(history.history['loss']) + len(history_fine.history['loss'])),
            history_fine.history['loss'],
            label='Fine-tune Train Loss',
            linestyle='--'
        )
        axes[1].plot(
            range(len(history.history['loss']), len(history.history['loss']) + len(history_fine.history['loss'])),
            history_fine.history['val_loss'],
            label='Fine-tune Val Loss',
            linestyle='--'
        )
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    history_path = Path(CONFIG['output_dir']) / f"{CONFIG['model_name']}_training_history.png"
    plt.savefig(history_path, dpi=150)
    print(f"Training history plot saved to: {history_path}")
    
    # Log training history plot to W&B
    wandb.log({'training_history': wandb.Image(str(history_path))})


def main():
    """Main training pipeline"""
    print("\n" + "=" * 60)
    print("KARAWANG PADI GUARD - DISEASE DETECTION TRAINING")
    print("=" * 60)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60 + "\n")

    # Initialize Weights & Biases
    wandb.init(
        project='karawang-padi-guard',
        name=CONFIG['model_name'],
        config=CONFIG,
        tags=['computer-vision', 'disease-detection', 'mobilenetv3', 'transfer-learning'],
        notes='Rice disease detection using MobileNetV3 with transfer learning and fine-tuning'
    )
    
    print(f"W&B Project: {wandb.run.project}")
    print(f"W&B Run Name: {wandb.run.name}")
    print(f"W&B Run ID: {wandb.run.id}\n")

    try:
        configure_runtime()

        # Create directories
        create_output_directories()

        # Load data
        train_ds, val_ds, test_ds = load_data()

        # Build model
        model, base_model = build_model(num_classes=CONFIG['num_classes'])

        # Compile model
        model = compile_model(model)

        # Train model
        history = train_model(model, train_ds, val_ds)

        # Fine-tune model
        history_fine = fine_tune_model(model, base_model, train_ds, val_ds)

        # Plot training history
        plot_training_history(history, history_fine)

        # Evaluate model
        metrics = evaluate_model(model, test_ds)

        # Save model and artifacts
        save_model_and_artifacts(model, metrics)

        print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nTraining completed successfully!")
        
        # Finish W&B run
        wandb.finish()
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        wandb.finish(exit_code=1)
        raise


if __name__ == "__main__":
    main()
