"""
IDR Prototype - IO-VNBD Real Data Demo

Runs the full pipeline on REAL vehicle IMU data from the IO-VNBD dataset:
  1. Load IO-VNBD S1 (smartphone IMU + vehicle reference)
  2. Train CNN-LSTM speed estimator on first 70% chronologically
  3. Evaluate on remaining 30%
  4. Reconstruct trajectory using:
     A. Vehicle reference heading + AI speed
     B. Smartphone gyro heading + AI speed
  5. Simulate GNSS outage
  6. Generate comparison plots and metrics

Usage:
    python run_iovnbd_demo.py

Dataset: IO-VNBD S1 (University of Warwick, ~86 min drive, 10 Hz)
"""
import os
import sys
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import RANDOM_SEED, RESULTS_DIR, MODEL_DIR
from src.io_vnbd_loader import load_iovnbd_s1, prepare_iovnbd_for_training
from src.model import build_model
from src.preprocessing import normalize


# ======================================================================
# Configuration for IO-VNBD
# ======================================================================
WINDOW_SECONDS = 2.0
STRIDE_SECONDS = 0.5
TRAIN_FRACTION = 0.7
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# GNSS outage simulation: choose a segment in the TEST portion
# Test portion starts at 70% of ~5174s = ~3622s from start
# Simulate 120s outage starting 60s into the test portion
GNSS_OUTAGE_OFFSET_S = 60.0   # seconds after test start
GNSS_OUTAGE_DURATION_S = 120.0

RESULTS_SUBDIR = os.path.join(RESULTS_DIR, "iovnbd")


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Compute distance between two GPS coordinates in meters.
    Uses the Haversine formula.
    """
    R = 6371000  # Earth radius in meters
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def latlon_to_local_xy(lat, lon, ref_lat, ref_lon):
    """
    Convert lat/lon to local XY coordinates (meters) relative to a reference.
    Uses simple equirectangular projection (valid for short distances).
    """
    R = 6371000
    ref_lat_rad = np.radians(ref_lat)
    x = R * np.radians(lon - ref_lon) * np.cos(ref_lat_rad)
    y = R * np.radians(lat - ref_lat)
    return x, y


def main():
    print("=" * 60)
    print("  IDR Prototype - IO-VNBD Real Data Demo")
    print("  AI Dead Reckoning on Real Vehicle IMU Data")
    print("=" * 60)

    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # ------------------------------------------------------------------
    # Step 1: Load dataset
    # ------------------------------------------------------------------
    print("\n[STEP 1] Loading IO-VNBD S1 dataset...")
    data = load_iovnbd_s1()

    sample_rate = data['sample_rate_hz']
    window_size = int(round(WINDOW_SECONDS * sample_rate))
    stride = int(round(STRIDE_SECONDS * sample_rate))

    # ------------------------------------------------------------------
    # Step 2: Prepare data and train model
    # ------------------------------------------------------------------
    print("\n[STEP 2] Preparing data and training model...")
    (X_train, y_train), (X_test, y_test), (mean, std), meta = \
        prepare_iovnbd_for_training(
            data,
            window_seconds=WINDOW_SECONDS,
            stride_seconds=STRIDE_SECONDS,
            train_fraction=TRAIN_FRACTION,
        )

    # Build model matched to actual window size
    model = build_model(window_size=window_size, n_channels=6)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=["mae"],
    )
    print(f"\n  Model parameters: {model.count_params():,}")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3,
            min_lr=1e-6, verbose=1,
        ),
    ]

    print("\n  Training...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    # Save model and normalization stats
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "idr_iovnbd_model.keras")
    model.save(model_path)
    norm_path = os.path.join(MODEL_DIR, "iovnbd_normalization_stats.json")
    with open(norm_path, "w") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist()}, f)
    print(f"  Model saved: {model_path}")

    # ------------------------------------------------------------------
    # Step 3: Evaluate speed prediction
    # ------------------------------------------------------------------
    print("\n[STEP 3] Evaluating speed prediction on test set...")
    y_pred = model.predict(X_test, verbose=0).flatten()
    y_pred = np.maximum(y_pred, 0.0)

    speed_mae = np.mean(np.abs(y_test - y_pred))
    speed_rmse = np.sqrt(np.mean((y_test - y_pred)**2))
    # MAPE (only where speed > 1 m/s to avoid division by near-zero)
    valid_speed_mask = y_test > 1.0
    if valid_speed_mask.sum() > 0:
        speed_mape = np.mean(
            np.abs(y_test[valid_speed_mask] - y_pred[valid_speed_mask])
            / y_test[valid_speed_mask]
        ) * 100
    else:
        speed_mape = float('nan')

    print(f"  Speed MAE:  {speed_mae:.3f} m/s ({speed_mae*3.6:.2f} km/h)")
    print(f"  Speed RMSE: {speed_rmse:.3f} m/s ({speed_rmse*3.6:.2f} km/h)")
    print(f"  Speed MAPE: {speed_mape:.1f}% (for speeds > 1 m/s)")

    # ------------------------------------------------------------------
    # Step 4: Reconstruct trajectory on test portion
    # ------------------------------------------------------------------
    print("\n[STEP 4] Reconstructing trajectory...")

    split_idx = meta['split_idx']
    test_timestamps = data['timestamp_s'][split_idx:]
    test_speed_gt = data['target_speed_mps'][split_idx:]
    test_heading_deg = data['vehicle_heading_deg'][split_idx:]
    test_vehicle_lat = data['vehicle_lat'][split_idx:]
    test_vehicle_lon = data['vehicle_lon'][split_idx:]
    test_imu = data['imu'][split_idx:]

    # Predict speed for the entire test sequence using sliding windows
    test_imu_norm = normalize(test_imu, mean, std)
    n_test = len(test_imu_norm)
    n_windows = (n_test - window_size) // stride + 1

    # Create windows for full test sequence
    test_windows = np.array([
        test_imu_norm[i*stride:i*stride+window_size]
        for i in range(n_windows)
    ], dtype=np.float32)

    pred_speeds_windows = model.predict(test_windows, verbose=0).flatten()
    pred_speeds_windows = np.maximum(pred_speeds_windows, 0.0)

    # Map window predictions back to sample-level (center of each window)
    window_center_indices = np.array([
        i * stride + window_size // 2 for i in range(n_windows)
    ])
    window_center_times = test_timestamps[window_center_indices]

    # Ground truth at window centers
    gt_speeds_at_windows = test_speed_gt[window_center_indices]
    gt_heading_at_windows = test_heading_deg[window_center_indices]

    # Time deltas between windows
    window_dt = np.zeros(n_windows)
    window_dt[1:] = np.diff(window_center_times)
    window_dt[0] = STRIDE_SECONDS

    # --- Trajectory A: Vehicle reference heading + AI predicted speed ---
    ref_heading_rad = np.radians(gt_heading_at_windows)
    dr_x_ref = np.zeros(n_windows)
    dr_y_ref = np.zeros(n_windows)
    for i in range(1, n_windows):
        ds = pred_speeds_windows[i] * window_dt[i]
        # Heading: 0=North, clockwise. Convert to math convention (0=East, CCW)
        angle = np.radians(90.0 - gt_heading_at_windows[i])
        dr_x_ref[i] = dr_x_ref[i-1] + ds * np.cos(angle)
        dr_y_ref[i] = dr_y_ref[i-1] + ds * np.sin(angle)

    # --- Trajectory B: Smartphone gyro heading + AI predicted speed ---
    # Integrate gyroscope yaw for heading
    gyro_yaw = test_imu[:, 3]  # rad/s
    gyro_heading = np.zeros(n_test)
    for i in range(1, n_test):
        dt_sample = test_timestamps[i] - test_timestamps[i-1]
        gyro_heading[i] = gyro_heading[i-1] + gyro_yaw[i] * dt_sample

    # Initialize gyro heading to match vehicle heading at start
    initial_heading_rad = np.radians(90.0 - test_heading_deg[0])
    gyro_heading_at_windows = gyro_heading[window_center_indices] + initial_heading_rad

    dr_x_gyro = np.zeros(n_windows)
    dr_y_gyro = np.zeros(n_windows)
    for i in range(1, n_windows):
        ds = pred_speeds_windows[i] * window_dt[i]
        dr_x_gyro[i] = dr_x_gyro[i-1] + ds * np.cos(gyro_heading_at_windows[i])
        dr_y_gyro[i] = dr_y_gyro[i-1] + ds * np.sin(gyro_heading_at_windows[i])

    # --- Ground truth trajectory (from vehicle GPS) ---
    ref_lat = test_vehicle_lat[0]
    ref_lon = test_vehicle_lon[0]
    gt_x_full, gt_y_full = latlon_to_local_xy(
        test_vehicle_lat, test_vehicle_lon, ref_lat, ref_lon
    )
    gt_x = gt_x_full[window_center_indices]
    gt_y = gt_y_full[window_center_indices]

    # ------------------------------------------------------------------
    # Step 5: GNSS outage simulation
    # ------------------------------------------------------------------
    print("\n[STEP 5] Simulating GNSS outage...")

    test_start_time = test_timestamps[0]
    outage_start_t = test_start_time + GNSS_OUTAGE_OFFSET_S
    outage_end_t = outage_start_t + GNSS_OUTAGE_DURATION_S

    print(f"  Outage: {GNSS_OUTAGE_OFFSET_S}s to "
          f"{GNSS_OUTAGE_OFFSET_S + GNSS_OUTAGE_DURATION_S}s after test start "
          f"({GNSS_OUTAGE_DURATION_S}s duration)")

    # Fused trajectory: GNSS when available, DR during outage
    fused_x = np.zeros(n_windows)
    fused_y = np.zeros(n_windows)
    outage_mask = np.zeros(n_windows, dtype=bool)

    for i in range(n_windows):
        t = window_center_times[i]
        in_outage = outage_start_t <= t <= outage_end_t
        outage_mask[i] = in_outage

        if not in_outage:
            fused_x[i] = gt_x[i]
            fused_y[i] = gt_y[i]
        else:
            if i == 0:
                fused_x[i] = gt_x[i]
                fused_y[i] = gt_y[i]
            else:
                # Dead reckon from previous fused position
                ds = pred_speeds_windows[i] * window_dt[i]
                angle = np.radians(90.0 - gt_heading_at_windows[i])
                fused_x[i] = fused_x[i-1] + ds * np.cos(angle)
                fused_y[i] = fused_y[i-1] + ds * np.sin(angle)

    # ------------------------------------------------------------------
    # Step 6: Compute metrics
    # ------------------------------------------------------------------
    print("\n[STEP 6] Computing metrics...")

    # Position errors (reference heading trajectory)
    pos_error_ref = np.sqrt((dr_x_ref - gt_x)**2 + (dr_y_ref - gt_y)**2)

    # Total distance
    gt_distance = np.sum(np.sqrt(np.diff(gt_x)**2 + np.diff(gt_y)**2))
    dr_distance_ref = np.sum(
        np.sqrt(np.diff(dr_x_ref)**2 + np.diff(dr_y_ref)**2))

    # GNSS outage metrics
    outage_pos_error = np.sqrt(
        (fused_x[outage_mask] - gt_x[outage_mask])**2 +
        (fused_y[outage_mask] - gt_y[outage_mask])**2
    )

    # Drift percentage
    drift_pct = pos_error_ref[-1] / max(gt_distance, 1.0) * 100

    metrics = {
        "speed_mae_mps": float(speed_mae),
        "speed_rmse_mps": float(speed_rmse),
        "speed_mape_pct": float(speed_mape),
        "speed_mae_kmh": float(speed_mae * 3.6),
        "speed_rmse_kmh": float(speed_rmse * 3.6),
        "gt_total_distance_m": float(gt_distance),
        "dr_total_distance_m": float(dr_distance_ref),
        "distance_error_pct": float(
            abs(dr_distance_ref - gt_distance) / gt_distance * 100),
        "final_position_error_m": float(pos_error_ref[-1]),
        "max_position_error_m": float(pos_error_ref.max()),
        "mean_position_error_m": float(pos_error_ref.mean()),
        "drift_percentage": float(drift_pct),
        "gnss_outage_duration_s": float(GNSS_OUTAGE_DURATION_S),
        "gnss_outage_mean_error_m": float(outage_pos_error.mean()),
        "gnss_outage_max_error_m": float(outage_pos_error.max()),
        "epochs_trained": int(len(history.history["loss"])),
        "dataset": "IO-VNBD S1",
        "train_fraction": TRAIN_FRACTION,
        "n_samples": int(data['n_samples']),
        "duration_min": float(data['duration_s'] / 60),
        "sample_rate_hz": float(sample_rate),
    }

    # ------------------------------------------------------------------
    # Step 7: Generate plots
    # ------------------------------------------------------------------
    print("\n[STEP 7] Generating plots...")
    os.makedirs(RESULTS_SUBDIR, exist_ok=True)

    # --- Plot 1: Velocity comparison ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    t_plot = window_center_times - test_start_time  # seconds into test

    ax = axes[0]
    ax.plot(t_plot, gt_speeds_at_windows * 3.6, 'b-', linewidth=1,
            label='Vehicle Reference', alpha=0.8)
    ax.plot(t_plot, pred_speeds_windows * 3.6, 'r-', linewidth=0.8,
            label='AI Prediction', alpha=0.7)
    ax.axvspan(GNSS_OUTAGE_OFFSET_S, GNSS_OUTAGE_OFFSET_S + GNSS_OUTAGE_DURATION_S,
               alpha=0.1, color='red', label='Simulated GNSS Outage')
    ax.set_ylabel("Speed (km/h)", fontsize=11)
    ax.set_title("IO-VNBD S1: AI Speed Estimation vs Vehicle Reference", fontsize=13)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    speed_error = (pred_speeds_windows - gt_speeds_at_windows) * 3.6
    ax.plot(t_plot, speed_error, 'r-', linewidth=0.6, alpha=0.7)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvspan(GNSS_OUTAGE_OFFSET_S, GNSS_OUTAGE_OFFSET_S + GNSS_OUTAGE_DURATION_S,
               alpha=0.1, color='red')
    ax.set_xlabel("Time into test segment (s)", fontsize=11)
    ax.set_ylabel("Speed Error (km/h)", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    vel_path = os.path.join(RESULTS_SUBDIR, "velocity_comparison.png")
    plt.savefig(vel_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {vel_path}")

    # --- Plot 2: Trajectory comparison ---
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    ax.plot(gt_x, gt_y, 'b-', linewidth=2, label='Ground Truth (Vehicle GPS)', alpha=0.8)
    ax.plot(dr_x_ref, dr_y_ref, 'r--', linewidth=1.5,
            label='DR: Ref Heading + AI Speed', alpha=0.7)
    ax.plot(dr_x_gyro, dr_y_gyro, 'g--', linewidth=1.2,
            label='DR: Gyro Heading + AI Speed', alpha=0.6)

    # Highlight GNSS outage on fused trajectory
    ax.plot(fused_x[outage_mask], fused_y[outage_mask],
            'orange', linewidth=3, alpha=0.7, label='DR During GNSS Outage')

    ax.scatter(gt_x[0], gt_y[0], s=150, c='green', marker='^',
               zorder=5, label='Start')
    ax.scatter(gt_x[-1], gt_y[-1], s=150, c='red', marker='v',
               zorder=5, label='End')

    ax.set_xlabel("East (m)", fontsize=12)
    ax.set_ylabel("North (m)", fontsize=12)
    ax.set_title("IO-VNBD S1: Trajectory Comparison\n"
                 "(Test segment: last 30% of drive)", fontsize=13)
    ax.legend(loc="best", fontsize=9)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    traj_path = os.path.join(RESULTS_SUBDIR, "trajectory_comparison.png")
    plt.savefig(traj_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {traj_path}")

    # --- Plot 3: GNSS outage demonstration ---
    fig, ax = plt.subplots(1, 1, figsize=(11, 8))

    # Show only the region around the outage
    margin = 30  # extra windows to show context
    outage_indices = np.where(outage_mask)[0]
    if len(outage_indices) > 0:
        plot_start = max(0, outage_indices[0] - margin)
        plot_end = min(n_windows, outage_indices[-1] + margin)
    else:
        plot_start, plot_end = 0, n_windows

    sl = slice(plot_start, plot_end)
    ax.plot(gt_x[sl], gt_y[sl], 'b-', linewidth=2.5,
            label='Ground Truth', alpha=0.9)
    ax.plot(fused_x[sl], fused_y[sl], 'g-', linewidth=1.5,
            label='GNSS + DR Fusion', alpha=0.8)

    # Highlight the outage portion
    outage_sl = outage_mask[sl]
    fused_x_sl = fused_x[plot_start:plot_end]
    fused_y_sl = fused_y[plot_start:plot_end]
    if outage_sl.any():
        ax.plot(fused_x_sl[outage_sl], fused_y_sl[outage_sl],
                'orange', linewidth=3.5, alpha=0.8,
                label=f'AI Dead Reckoning ({GNSS_OUTAGE_DURATION_S:.0f}s outage)')

    # Mark outage start/end
    if len(outage_indices) > 0:
        ax.scatter(gt_x[outage_indices[0]], gt_y[outage_indices[0]],
                   s=120, c='orange', marker='x', zorder=5,
                   linewidths=3, label='GNSS Lost')
        ax.scatter(gt_x[outage_indices[-1]], gt_y[outage_indices[-1]],
                   s=120, c='purple', marker='x', zorder=5,
                   linewidths=3, label='GNSS Restored')

    ax.set_xlabel("East (m)", fontsize=12)
    ax.set_ylabel("North (m)", fontsize=12)
    ax.set_title(f"GNSS Outage Demonstration ({GNSS_OUTAGE_DURATION_S:.0f}s)\n"
                 "SIMULATED outage on IO-VNBD S1 data", fontsize=13)
    ax.legend(loc="best", fontsize=10)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outage_path = os.path.join(RESULTS_SUBDIR, "gnss_outage_demo.png")
    plt.savefig(outage_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {outage_path}")

    # Save metrics
    metrics_path = os.path.join(RESULTS_SUBDIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved: {metrics_path}")

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  IO-VNBD IDR DEMO - RESULTS")
    print("=" * 60)
    print(f"\n  Dataset:         IO-VNBD S1")
    print(f"  Samples:         {data['n_samples']:,}")
    print(f"  Duration:        {data['duration_s']/60:.1f} min")
    print(f"  Sampling rate:   {sample_rate:.1f} Hz")
    print(f"  Train/Test:      {TRAIN_FRACTION*100:.0f}% / "
          f"{(1-TRAIN_FRACTION)*100:.0f}% (chronological)")
    print(f"\n  Speed Estimation:")
    print(f"    MAE:           {speed_mae:.3f} m/s ({speed_mae*3.6:.2f} km/h)")
    print(f"    RMSE:          {speed_rmse:.3f} m/s ({speed_rmse*3.6:.2f} km/h)")
    print(f"    MAPE:          {speed_mape:.1f}% (speeds > 1 m/s)")
    print(f"\n  Trajectory (Ref Heading + AI Speed):")
    print(f"    Distance GT:   {gt_distance:.1f} m")
    print(f"    Distance DR:   {dr_distance_ref:.1f} m")
    print(f"    Distance err:  {metrics['distance_error_pct']:.1f}%")
    print(f"    Final error:   {pos_error_ref[-1]:.1f} m")
    print(f"    Max error:     {pos_error_ref.max():.1f} m")
    print(f"    Mean error:    {pos_error_ref.mean():.1f} m")
    print(f"    Drift:         {drift_pct:.2f}%")
    print(f"\n  GNSS Outage ({GNSS_OUTAGE_DURATION_S:.0f}s, SIMULATED):")
    print(f"    Mean error:    {outage_pos_error.mean():.1f} m")
    print(f"    Max error:     {outage_pos_error.max():.1f} m")
    print(f"\n  Epochs trained:  {metrics['epochs_trained']}")
    print(f"  Model params:    {model.count_params():,}")
    print("=" * 60)
    print(f"\n  Output files:")
    print(f"    {vel_path}")
    print(f"    {traj_path}")
    print(f"    {outage_path}")
    print(f"    {metrics_path}")
    print(f"    {model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
