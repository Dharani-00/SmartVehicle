"""
Inference Pipeline for IDR Speed Estimation.

Separated from training - loads a trained model and runs predictions
on new IMU data to produce speed estimates and dead-reckoned trajectory.
"""
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from src.config import (
    WINDOW_SIZE, STRIDE, SAMPLE_RATE_HZ, MODEL_DIR,
    GNSS_OUTAGE_START_S, GNSS_OUTAGE_END_S,
)
from src.preprocessing import load_sequence, normalize, create_sliding_windows


def load_trained_model(model_name="idr_speed_model"):
    """Load the trained Keras model and normalization stats."""
    model_path = os.path.join(MODEL_DIR, f"{model_name}.keras")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found: {model_path}")

    model = tf.keras.models.load_model(model_path)

    norm_path = os.path.join(MODEL_DIR, "normalization_stats.json")
    if not os.path.exists(norm_path):
        raise FileNotFoundError(f"Normalization stats not found: {norm_path}")

    with open(norm_path, "r") as f:
        norm_data = json.load(f)

    mean = np.array(norm_data["mean"], dtype=np.float32)
    std = np.array(norm_data["std"], dtype=np.float32)

    return model, mean, std


def predict_speed(model, imu_data, mean, std, timestamps=None,
                  window_size=None, stride=None):
    """
    Predict forward speed from raw IMU data.

    Args:
        model: Trained Keras model
        imu_data: (N, 6) raw IMU readings [ax, ay, az, gx, gy, gz]
        mean, std: Normalization stats from training
        timestamps: (N,) actual timestamps in seconds. If None, assumes uniform sampling.
        window_size: Override window size
        stride: Override stride

    Returns:
        predicted_speeds: (M,) predicted speed for each window
        window_centers: (M,) center timestamp of each window
        window_dt: (M,) actual time span of each stride (for integration)
    """
    if window_size is None:
        window_size = WINDOW_SIZE
    if stride is None:
        stride = STRIDE

    if timestamps is None:
        timestamps = np.arange(len(imu_data)) / SAMPLE_RATE_HZ

    # Validate
    if imu_data.shape[1] != 6:
        raise ValueError(f"Expected 6 IMU channels, got {imu_data.shape[1]}")
    if np.any(np.isnan(imu_data)):
        raise ValueError("NaN values in IMU data")

    # Normalize
    imu_norm = normalize(imu_data.astype(np.float32), mean, std)

    # Create dummy speed (not used for prediction, only for window creation API)
    dummy_speed = np.zeros(len(imu_norm), dtype=np.float32)
    windows, _, window_times = create_sliding_windows(
        imu_norm, dummy_speed, timestamps, window_size, stride
    )

    # Predict
    predicted_speeds = model.predict(windows, verbose=0).flatten()
    # Speed should be non-negative
    predicted_speeds = np.maximum(predicted_speeds, 0.0)

    # Compute actual time delta between consecutive windows using timestamps
    window_dt = np.zeros(len(window_times))
    for i in range(1, len(window_times)):
        window_dt[i] = window_times[i] - window_times[i - 1]
    window_dt[0] = stride / SAMPLE_RATE_HZ  # First window uses nominal dt

    return predicted_speeds, window_times, window_dt


def integrate_heading_from_gyro(imu_data, timestamps, stride, window_size):
    """
    Estimate heading changes by integrating gyroscope Z-axis (yaw rate).

    This is separated from the neural network so it can be later replaced
    with a more sophisticated orientation estimator (complementary filter,
    Madgwick, ESKF, etc.).

    Args:
        imu_data: (N, 6) raw IMU [ax, ay, az, gx, gy, gz]
        timestamps: (N,) timestamps in seconds
        stride: Window stride in samples
        window_size: Window size in samples

    Returns:
        headings: (M,) cumulative heading at each window center (radians)
    """
    n_windows = (len(imu_data) - window_size) // stride + 1
    headings = np.zeros(n_windows)

    heading = 0.0
    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        # Integrate gyro_z over the stride period for this window
        if i > 0:
            stride_start = start
            stride_end = min(start + stride, len(imu_data))
            gz_segment = imu_data[stride_start:stride_end, 5]  # gz is column 5
            ts_segment = timestamps[stride_start:stride_end]
            # Use trapezoidal integration with actual timestamps
            if len(ts_segment) > 1:
                dt_samples = np.diff(ts_segment)
                gz_avg = (gz_segment[:-1] + gz_segment[1:]) / 2.0
                heading += np.sum(gz_avg * dt_samples)
        headings[i] = heading

    return headings


def dead_reckon_trajectory(speeds, headings, window_dt):
    """
    Reconstruct 2D trajectory from predicted speeds and headings.

    Dead reckoning:
        x[i] = x[i-1] + speed[i] * cos(heading[i]) * dt
        y[i] = y[i-1] + speed[i] * sin(heading[i]) * dt

    Args:
        speeds: (M,) predicted forward speeds (m/s)
        headings: (M,) heading at each step (radians)
        window_dt: (M,) time interval for each step (seconds)

    Returns:
        x, y: (M,) trajectory coordinates (meters)
        distance: (M,) cumulative distance travelled
    """
    n = len(speeds)
    x = np.zeros(n)
    y = np.zeros(n)
    distance = np.zeros(n)

    for i in range(1, n):
        ds = speeds[i] * window_dt[i]
        x[i] = x[i - 1] + ds * np.cos(headings[i])
        y[i] = y[i - 1] + ds * np.sin(headings[i])
        distance[i] = distance[i - 1] + ds

    return x, y, distance


def run_inference_on_sequence(filepath, model, mean, std):
    """
    Complete inference pipeline for a single IMU sequence.

    Returns a dict with all results needed for trajectory plotting.
    """
    # Load raw data
    df = pd.read_csv(filepath)
    imu_data = df[["ax", "ay", "az", "gx", "gy", "gz"]].values.astype(np.float32)
    timestamps = df["timestamp"].values

    # Ground truth (if available)
    has_gt = "gt_speed" in df.columns and "gt_x" in df.columns
    gt_speed = df["gt_speed"].values if has_gt else None
    gt_x = df["gt_x"].values if has_gt else None
    gt_y = df["gt_y"].values if has_gt else None

    # Predict speed
    pred_speeds, window_times, window_dt = predict_speed(
        model, imu_data, mean, std, timestamps
    )

    # Estimate heading from gyroscope
    headings = integrate_heading_from_gyro(
        imu_data, timestamps, STRIDE, WINDOW_SIZE
    )

    # Dead reckon trajectory
    dr_x, dr_y, dr_distance = dead_reckon_trajectory(
        pred_speeds, headings, window_dt
    )

    # GNSS-denied simulation: blend reference + dead reckoning
    gnss_x, gnss_y = simulate_gnss_fusion(
        dr_x, dr_y, gt_x, gt_y, window_times, timestamps
    )

    # Compute ground-truth trajectory at window centers (for comparison)
    gt_x_windows = None
    gt_y_windows = None
    gt_speed_windows = None
    if has_gt:
        # Subsample ground truth at window center indices
        window_indices = [
            WINDOW_SIZE // 2 + i * STRIDE
            for i in range(len(window_times))
        ]
        window_indices = [min(idx, len(gt_x) - 1) for idx in window_indices]
        gt_x_windows = gt_x[window_indices]
        gt_y_windows = gt_y[window_indices]
        gt_speed_windows = gt_speed[window_indices]

    return {
        "window_times": window_times,
        "pred_speeds": pred_speeds,
        "headings": headings,
        "dr_x": dr_x,
        "dr_y": dr_y,
        "dr_distance": dr_distance,
        "gnss_x": gnss_x,
        "gnss_y": gnss_y,
        "gt_x": gt_x_windows,
        "gt_y": gt_y_windows,
        "gt_speed": gt_speed_windows,
        "window_dt": window_dt,
    }


def simulate_gnss_fusion(dr_x, dr_y, gt_x, gt_y, window_times, timestamps):
    """
    Simulate GNSS availability: use reference when available, DR when denied.

    This is a DEMONSTRATION of the future architecture.
    It is NOT a complete GNSS+INS fusion implementation.
    """
    n = len(dr_x)
    fused_x = np.zeros(n)
    fused_y = np.zeros(n)

    if gt_x is None:
        return dr_x.copy(), dr_y.copy()

    # Subsample ground truth at window times
    window_indices = [
        WINDOW_SIZE // 2 + i * STRIDE
        for i in range(n)
    ]
    window_indices = [min(idx, len(gt_x) - 1) for idx in window_indices]

    for i in range(n):
        t = window_times[i]
        gnss_available = not (GNSS_OUTAGE_START_S <= t <= GNSS_OUTAGE_END_S)

        if gnss_available:
            fused_x[i] = gt_x[window_indices[i]]
            fused_y[i] = gt_y[window_indices[i]]
        else:
            if i == 0:
                fused_x[i] = dr_x[i]
                fused_y[i] = dr_y[i]
            else:
                # Continue dead reckoning from last known position
                dx = dr_x[i] - dr_x[i - 1]
                dy = dr_y[i] - dr_y[i - 1]
                fused_x[i] = fused_x[i - 1] + dx
                fused_y[i] = fused_y[i - 1] + dy

    return fused_x, fused_y
