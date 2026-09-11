"""
CNN-LSTM Model for IMU-based Speed Estimation.

Architecture:
    Input (batch, window_size, 6)
      → Conv1D(64, k=5, causal) → BatchNorm → ReLU
      → Conv1D(64, k=5, causal) → BatchNorm → ReLU
      → Conv1D(128, k=3, causal) → BatchNorm → ReLU
      → LSTM(64)
      → Dense(32, relu)
      → Dense(1, linear) → forward speed (m/s)

Designed to be:
  - Lightweight (~150K parameters) for smartphone deployment
  - TFLite-compatible (no unsupported ops)
  - Configurable via config.py
"""
import tensorflow as tf
from src.config import (
    WINDOW_SIZE, CONV_FILTERS, KERNEL_SIZE,
    LSTM_UNITS, DENSE_UNITS, DROPOUT_RATE,
)


def build_model(window_size=None, n_channels=6):
    """
    Build the CNN-LSTM speed estimation model.

    Args:
        window_size: Number of timesteps per window (default from config)
        n_channels: Number of IMU channels (default 6: ax,ay,az,gx,gy,gz)

    Returns:
        Compiled-ready Keras model (not yet compiled - caller chooses optimizer)
    """
    if window_size is None:
        window_size = WINDOW_SIZE

    inputs = tf.keras.Input(shape=(window_size, n_channels), name="imu_input")

    # --- Convolutional feature extraction ---
    # Layer 1: Extract local temporal patterns
    x = tf.keras.layers.Conv1D(
        CONV_FILTERS[0], KERNEL_SIZE, padding="causal", name="conv1"
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name="bn1")(x)
    x = tf.keras.layers.ReLU(name="relu1")(x)

    # Layer 2: Higher-level features
    x = tf.keras.layers.Conv1D(
        CONV_FILTERS[1], KERNEL_SIZE, padding="causal", name="conv2"
    )(x)
    x = tf.keras.layers.BatchNormalization(name="bn2")(x)
    x = tf.keras.layers.ReLU(name="relu2")(x)

    # Layer 3: Compress into richer representation
    x = tf.keras.layers.Conv1D(
        CONV_FILTERS[2], 3, padding="causal", name="conv3"
    )(x)
    x = tf.keras.layers.BatchNormalization(name="bn3")(x)
    x = tf.keras.layers.ReLU(name="relu3")(x)

    # --- Temporal sequence modeling ---
    x = tf.keras.layers.LSTM(LSTM_UNITS, name="lstm")(x)
    x = tf.keras.layers.Dropout(DROPOUT_RATE, name="dropout")(x)

    # --- Regression head ---
    x = tf.keras.layers.Dense(DENSE_UNITS, activation="relu", name="dense1")(x)
    output = tf.keras.layers.Dense(1, activation="linear", name="speed_output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=output, name="IDR_CNN_LSTM")
    return model


def get_model_summary(model):
    """Return model summary as string."""
    lines = []
    model.summary(print_fn=lambda x: lines.append(x))
    return "\n".join(lines)


if __name__ == "__main__":
    model = build_model()
    model.summary()
    print(f"\nTotal parameters: {model.count_params():,}")
