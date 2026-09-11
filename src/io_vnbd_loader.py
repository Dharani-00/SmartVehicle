"""
IO-VNBD Dataset Loader for IDR Prototype.

Loads the IO-VNBD (Inertial Odometry - Vehicle Navigation Benchmark Dataset)
smartphone IMU data and vehicle reference data for training and evaluation.

Dataset: Synchronised smartphone (S) and vehicle (V) recordings.
Source: University of Warwick / Coventry University

The S and V files are pre-synchronized row-by-row at ~10 Hz.
"""
import os
import numpy as np
import pandas as pd
from src.config import MODEL_DIR


# Default dataset paths
IOVNBD_BASE = r"C:\Users\BHAVYA\Downloads\IO-VNBD\Synchronised V abd S datasets\Categorised IOVNB Dataset"
S1_SMARTPHONE = os.path.join(IOVNBD_BASE, "S (Driver A)", "S1", "S-S1.csv")
S1_VEHICLE = os.path.join(IOVNBD_BASE, "S (Driver A)", "S1", "V-S1.csv")


def load_iovnbd_s1(smartphone_path=None, vehicle_path=None, cache_dir=None):
    """
    Load and process the IO-VNBD S1 dataset.

    The smartphone and vehicle CSVs are pre-synchronized (same row count,
    same timestamps). Both sampled at 10 Hz.

    Returns:
        dict with keys:
            'timestamp_s': (N,) time in seconds from start
            'imu': (N, 6) [ax, ay, az, gyro_yaw, gyro_pitch, gyro_roll]
            'target_speed_mps': (N,) vehicle velocity in m/s
            'vehicle_heading_deg': (N,) vehicle heading in degrees
            'vehicle_lat': (N,) vehicle latitude
            'vehicle_lon': (N,) vehicle longitude
            'smartphone_lat': (N,) smartphone GPS latitude
            'smartphone_lon': (N,) smartphone GPS longitude
            'sample_rate_hz': float
            'duration_s': float
            'n_samples': int
    """
    if smartphone_path is None:
        smartphone_path = S1_SMARTPHONE
    if vehicle_path is None:
        vehicle_path = S1_VEHICLE

    # Check if cached version exists
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "data", "iovnbd_cache")

    cache_file = os.path.join(cache_dir, "s1_processed.npz")
    if os.path.exists(cache_file):
        print(f"  Loading cached dataset: {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
        result = {k: data[k] for k in data.files}
        result['sample_rate_hz'] = float(result['sample_rate_hz'])
        result['duration_s'] = float(result['duration_s'])
        result['n_samples'] = int(result['n_samples'])
        _print_summary(result)
        return result

    # Load raw CSVs
    print(f"  Loading smartphone data: {os.path.basename(smartphone_path)}")
    s_df = pd.read_csv(smartphone_path, encoding='latin-1')

    print(f"  Loading vehicle data: {os.path.basename(vehicle_path)}")
    v_df = pd.read_csv(vehicle_path)

    if len(s_df) != len(v_df):
        raise ValueError(
            f"Row count mismatch: smartphone={len(s_df)}, vehicle={len(v_df)}"
        )

    # Strip leading/trailing spaces from column names
    s_df.columns = [c.strip() for c in s_df.columns]
    v_df.columns = [c.strip() for c in v_df.columns]

    # Extract timestamps - smartphone uses "TIME SINCE START (ms)"
    ts_ms = s_df["TIME SINCE START (ms)"].values.astype(np.float64)
    timestamp_s = (ts_ms - ts_ms[0]) / 1000.0  # Convert to seconds from start

    # Measure actual sample rate
    dt_median = np.median(np.diff(timestamp_s))
    sample_rate_hz = 1.0 / dt_median

    # Extract smartphone IMU channels
    # Column names have encoding artifacts for ² - use positional index
    # Columns 9-11: ACCELEROMETER X, Y, Z (m/s²)
    # Columns 15-17: GYROSCOPE Yaw, Pitch, Roll (rad/s)
    acc_x = s_df.iloc[:, 9].values.astype(np.float32)
    acc_y = s_df.iloc[:, 10].values.astype(np.float32)
    acc_z = s_df.iloc[:, 11].values.astype(np.float32)
    gyro_yaw = s_df.iloc[:, 15].values.astype(np.float32)
    gyro_pitch = s_df.iloc[:, 16].values.astype(np.float32)
    gyro_roll = s_df.iloc[:, 17].values.astype(np.float32)

    imu = np.column_stack([acc_x, acc_y, acc_z, gyro_yaw, gyro_pitch, gyro_roll])

    # Extract vehicle reference velocity (km/h → m/s)
    vel_kmh = v_df["Velocity (km/hr)"].values.astype(np.float32)
    target_speed_mps = vel_kmh / 3.6

    # Vehicle heading (degrees, for trajectory reconstruction)
    vehicle_heading_deg = v_df["Heading (degrees)"].values.astype(np.float32)

    # Vehicle reference position (ground truth)
    vehicle_lat = v_df["Latitude (degrees)"].values.astype(np.float64)
    vehicle_lon = v_df["Longitude (degrees)"].values.astype(np.float64)

    # Smartphone GPS position
    smartphone_lat = s_df["GPS LATITUDE (degrees)"].values.astype(np.float64)
    smartphone_lon = s_df.iloc[:, 1].values.astype(np.float64)  # Has leading space

    # Remove any rows with NaN in critical columns
    valid_mask = (
        ~np.isnan(imu).any(axis=1) &
        ~np.isnan(target_speed_mps) &
        ~np.isnan(vehicle_heading_deg) &
        ~np.isnan(vehicle_lat) &
        ~np.isnan(vehicle_lon)
    )

    n_invalid = (~valid_mask).sum()
    if n_invalid > 0:
        print(f"  WARNING: Removing {n_invalid} invalid rows")
        timestamp_s = timestamp_s[valid_mask]
        imu = imu[valid_mask]
        target_speed_mps = target_speed_mps[valid_mask]
        vehicle_heading_deg = vehicle_heading_deg[valid_mask]
        vehicle_lat = vehicle_lat[valid_mask]
        vehicle_lon = vehicle_lon[valid_mask]
        smartphone_lat = smartphone_lat[valid_mask]
        smartphone_lon = smartphone_lon[valid_mask]

    duration_s = timestamp_s[-1] - timestamp_s[0]
    n_samples = len(timestamp_s)

    result = {
        'timestamp_s': timestamp_s,
        'imu': imu,
        'target_speed_mps': target_speed_mps,
        'vehicle_heading_deg': vehicle_heading_deg,
        'vehicle_lat': vehicle_lat,
        'vehicle_lon': vehicle_lon,
        'smartphone_lat': smartphone_lat,
        'smartphone_lon': smartphone_lon,
        'sample_rate_hz': sample_rate_hz,
        'duration_s': duration_s,
        'n_samples': n_samples,
    }

    # Cache for fast re-loading
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(cache_file, **result)
    print(f"  Cached processed data: {cache_file}")

    _print_summary(result)
    return result


def _print_summary(data):
    """Print dataset summary statistics."""
    print(f"\n  IO-VNBD S1 Dataset Summary:")
    print(f"    Samples:       {data['n_samples']:,}")
    print(f"    Duration:      {data['duration_s']:.1f} s ({data['duration_s']/60:.1f} min)")
    print(f"    Sample rate:   {data['sample_rate_hz']:.1f} Hz")
    print(f"    Speed range:   [{data['target_speed_mps'].min():.2f}, "
          f"{data['target_speed_mps'].max():.2f}] m/s "
          f"([{data['target_speed_mps'].min()*3.6:.1f}, "
          f"{data['target_speed_mps'].max()*3.6:.1f}] km/h)")
    print(f"    Mean speed:    {data['target_speed_mps'].mean():.2f} m/s "
          f"({data['target_speed_mps'].mean()*3.6:.1f} km/h)")


def prepare_iovnbd_for_training(data, window_seconds=2.0, stride_seconds=0.5,
                                train_fraction=0.7):
    """
    Prepare IO-VNBD data for model training.

    Uses chronological split (NOT random) since this is a time series.

    Args:
        data: dict from load_iovnbd_s1()
        window_seconds: window duration in seconds
        stride_seconds: stride duration in seconds
        train_fraction: fraction for training (rest is test)

    Returns:
        (X_train, y_train), (X_test, y_test), norm_stats, metadata
    """
    from src.preprocessing import compute_normalization_stats, normalize

    sample_rate = data['sample_rate_hz']
    window_size = int(round(window_seconds * sample_rate))
    stride = int(round(stride_seconds * sample_rate))

    print(f"\n  Windowing parameters:")
    print(f"    Window: {window_seconds}s = {window_size} samples")
    print(f"    Stride: {stride_seconds}s = {stride} samples")

    imu = data['imu']
    speed = data['target_speed_mps']
    timestamps = data['timestamp_s']
    n_total = len(imu)

    # Chronological split
    split_idx = int(n_total * train_fraction)
    print(f"    Chronological split at sample {split_idx} "
          f"(t={timestamps[split_idx]:.1f}s)")

    train_imu = imu[:split_idx]
    train_speed = speed[:split_idx]
    test_imu = imu[split_idx:]
    test_speed = speed[split_idx:]

    # Compute normalization from TRAINING data only
    mean, std = compute_normalization_stats([train_imu])

    # Normalize
    train_imu_norm = normalize(train_imu, mean, std)
    test_imu_norm = normalize(test_imu, mean, std)

    # Create sliding windows
    X_train, y_train = _create_windows(train_imu_norm, train_speed,
                                       window_size, stride)
    X_test, y_test = _create_windows(test_imu_norm, test_speed,
                                     window_size, stride)

    print(f"    Train windows: {len(X_train)}")
    print(f"    Test windows:  {len(X_test)}")

    metadata = {
        'window_size': window_size,
        'stride': stride,
        'sample_rate_hz': sample_rate,
        'split_idx': split_idx,
        'window_seconds': window_seconds,
        'stride_seconds': stride_seconds,
    }

    return (X_train, y_train), (X_test, y_test), (mean, std), metadata


def _create_windows(imu_norm, speed, window_size, stride):
    """Create sliding windows from normalized IMU data."""
    n = len(imu_norm)
    windows = []
    targets = []

    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        windows.append(imu_norm[start:end])
        targets.append(np.mean(speed[start:end]))

    return np.array(windows, dtype=np.float32), np.array(targets, dtype=np.float32)


if __name__ == "__main__":
    print("Loading IO-VNBD S1 dataset...")
    data = load_iovnbd_s1()
    print("\nPreparing for training...")
    (X_train, y_train), (X_test, y_test), norm_stats, meta = \
        prepare_iovnbd_for_training(data)
    print(f"\nReady: X_train={X_train.shape}, X_test={X_test.shape}")
