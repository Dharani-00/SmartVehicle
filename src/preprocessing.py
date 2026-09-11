"""
IMU Data Preprocessing for IDR Prototype.

Handles:
  - Loading CSV data
  - Normalization (zero-mean, unit-variance)
  - Sliding window creation
  - Train/validation/test splitting

Designed so real IMU data can replace synthetic data with no model changes.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import (
    WINDOW_SIZE, STRIDE, SAMPLE_RATE_HZ,
    VALIDATION_SPLIT, TEST_SPLIT, RANDOM_SEED,
)


def load_sequence(filepath):
    """
    Load a single IMU sequence from CSV.

    Expected columns: timestamp, ax, ay, az, gx, gy, gz, gt_speed, ...
    Returns (imu_data, ground_truth_speed, timestamps).
    """
    df = pd.read_csv(filepath)

    required_cols = ["timestamp", "ax", "ay", "az", "gx", "gy", "gz", "gt_speed"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Validate no NaN in IMU channels
    imu_cols = ["ax", "ay", "az", "gx", "gy", "gz"]
    if df[imu_cols].isna().any().any():
        raise ValueError(f"NaN values found in IMU data: {filepath}")

    timestamps = df["timestamp"].values
    imu_data = df[imu_cols].values.astype(np.float32)
    gt_speed = df["gt_speed"].values.astype(np.float32)

    return imu_data, gt_speed, timestamps


def compute_normalization_stats(imu_sequences):
    """
    Compute mean and std across all sequences for normalization.

    Returns (mean, std) each of shape (6,).
    """
    all_data = np.vstack(imu_sequences)
    mean = all_data.mean(axis=0)
    std = all_data.std(axis=0)
    # Prevent division by zero
    std[std < 1e-8] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def normalize(imu_data, mean, std):
    """Apply zero-mean unit-variance normalization."""
    return (imu_data - mean) / std


def create_sliding_windows(imu_data, gt_speed, timestamps, window_size=None,
                           stride=None, sample_rate=None):
    """
    Create overlapping sliding windows from a single sequence.

    Each window contains `window_size` consecutive IMU samples.
    The target for each window is the mean speed over that window.

    Uses actual timestamps to compute the true time span of each window
    rather than assuming perfectly uniform sampling.

    Returns:
        windows: (N, window_size, 6) IMU data windows
        targets: (N,) mean speed for each window
        window_times: (N,) center timestamp of each window
    """
    if window_size is None:
        window_size = WINDOW_SIZE
    if stride is None:
        stride = STRIDE
    if sample_rate is None:
        sample_rate = SAMPLE_RATE_HZ

    n_samples = len(imu_data)
    if n_samples < window_size:
        raise ValueError(
            f"Sequence length ({n_samples}) < window_size ({window_size})"
        )

    # Validate input shapes
    if imu_data.shape[1] != 6:
        raise ValueError(f"Expected 6 IMU channels, got {imu_data.shape[1]}")

    windows = []
    targets = []
    window_times = []

    for start in range(0, n_samples - window_size + 1, stride):
        end = start + window_size
        window = imu_data[start:end]
        target = np.mean(gt_speed[start:end])
        center_time = timestamps[start + window_size // 2]

        windows.append(window)
        targets.append(target)
        window_times.append(center_time)

    windows = np.array(windows, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)
    window_times = np.array(window_times, dtype=np.float64)

    return windows, targets, window_times


def prepare_dataset(file_paths, window_size=None, stride=None):
    """
    Full preprocessing pipeline: load → normalize → window → split.

    Returns:
        (X_train, y_train), (X_val, y_val), (X_test, y_test),
        normalization_stats (mean, std)
    """
    if window_size is None:
        window_size = WINDOW_SIZE
    if stride is None:
        stride = STRIDE

    # Load all sequences
    imu_sequences = []
    speed_sequences = []
    timestamp_sequences = []

    for fp in file_paths:
        imu, speed, ts = load_sequence(fp)
        imu_sequences.append(imu)
        speed_sequences.append(speed)
        timestamp_sequences.append(ts)

    # Compute normalization from all data
    mean, std = compute_normalization_stats(imu_sequences)

    # Normalize and create windows
    all_windows = []
    all_targets = []

    for imu, speed, ts in zip(imu_sequences, speed_sequences, timestamp_sequences):
        imu_norm = normalize(imu, mean, std)
        windows, targets, _ = create_sliding_windows(
            imu_norm, speed, ts, window_size, stride
        )
        all_windows.append(windows)
        all_targets.append(targets)

    X = np.concatenate(all_windows, axis=0)
    y = np.concatenate(all_targets, axis=0)

    print(f"  Total windows: {len(X)}")
    print(f"  Input shape: {X.shape}")
    print(f"  Speed range: [{y.min():.2f}, {y.max():.2f}] m/s")

    # Split: train / (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(VALIDATION_SPLIT + TEST_SPLIT), random_state=RANDOM_SEED
    )
    # Split temp into val / test
    val_fraction = VALIDATION_SPLIT / (VALIDATION_SPLIT + TEST_SPLIT)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_fraction), random_state=RANDOM_SEED
    )

    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), (mean, std)
