"""
Training Pipeline for IDR Speed Estimation Model.

Handles:
  - Dataset preparation
  - Model compilation and training
  - Early stopping with best weights restoration
  - Metrics reporting (MAE, RMSE)
  - Model and normalization stats saving
"""
import os
import json
import numpy as np
import tensorflow as tf
from src.config import (
    LEARNING_RATE, BATCH_SIZE, EPOCHS,
    EARLY_STOP_PATIENCE, MODEL_DIR, RESULTS_DIR, RANDOM_SEED,
)
from src.model import build_model, get_model_summary
from src.preprocessing import prepare_dataset


def train_model(file_paths, model_name="idr_speed_model"):
    """
    Full training pipeline.

    Args:
        file_paths: List of CSV file paths for training data
        model_name: Base name for saved model files

    Returns:
        (model, history, metrics, norm_stats)
    """
    # Reproducibility
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("\n[1/5] Preparing dataset...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test), (mean, std) = \
        prepare_dataset(file_paths)

    print("\n[2/5] Building model...")
    window_size = X_train.shape[1]
    n_channels = X_train.shape[2]
    model = build_model(window_size=window_size, n_channels=n_channels)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=["mae"],
    )

    print(get_model_summary(model))
    print(f"\n  Parameters: {model.count_params():,}")

    print("\n[3/5] Training...")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    print("\n[4/5] Evaluating on test set...")
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    y_pred = model.predict(X_test, verbose=0).flatten()
    test_rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

    metrics = {
        "test_mae_ms": float(test_mae),
        "test_rmse_ms": float(test_rmse),
        "test_loss_mse": float(test_loss),
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "epochs_trained": int(len(history.history["loss"])),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
    }

    print(f"\n  Test MAE:  {metrics['test_mae_ms']:.4f} m/s")
    print(f"  Test RMSE: {metrics['test_rmse_ms']:.4f} m/s")
    print(f"  Epochs trained: {metrics['epochs_trained']}")

    print("\n[5/5] Saving model and artifacts...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Save Keras model
    model_path = os.path.join(MODEL_DIR, f"{model_name}.keras")
    model.save(model_path)
    print(f"  Model saved: {model_path}")

    # Save normalization stats (needed for inference)
    norm_stats = {"mean": mean.tolist(), "std": std.tolist()}
    norm_path = os.path.join(MODEL_DIR, "normalization_stats.json")
    with open(norm_path, "w") as f:
        json.dump(norm_stats, f, indent=2)
    print(f"  Normalization stats saved: {norm_path}")

    # Save metrics
    metrics_path = os.path.join(RESULTS_DIR, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved: {metrics_path}")

    # Save training history
    history_path = os.path.join(RESULTS_DIR, "training_history.json")
    hist_data = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(history_path, "w") as f:
        json.dump(hist_data, f, indent=2)

    return model, history, metrics, (mean, std)


if __name__ == "__main__":
    from src.data_generator import generate_dataset
    print("Generating synthetic data...")
    files = generate_dataset()
    print("\nTraining model...")
    model, history, metrics, norm_stats = train_model(files)
