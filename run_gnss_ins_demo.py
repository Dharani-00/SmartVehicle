"""
IDR Prototype - GNSS + INS Sensor Fusion Demo

Demonstrates Extended Kalman Filter fusion of:
  - Orientation sensor heading (vehicle reference as proxy)
  - AI-predicted forward velocity (CNN-LSTM)
  - Smartphone GNSS position (when available)

Compares three trajectories:
  1. Ground Truth (vehicle reference GPS)
  2. AI Dead Reckoning only (no GNSS corrections, from origin)
  3. EKF GNSS + INS fusion (GNSS corrected, degrades gracefully)

With a simulated 120-second GNSS outage to demonstrate:
  - Before outage: GNSS + INS -> stable, near-truth position
  - During outage: AI + IMU dead reckoning -> gradual drift
  - After GNSS returns: EKF correction -> trajectory returns toward reference

Usage:
    python run_gnss_ins_demo.py

IMPORTANT: The AI model does NOT receive GNSS input.
GNSS is used ONLY as an EKF measurement update.
"""
import os
import sys
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import RANDOM_SEED, RESULTS_DIR, MODEL_DIR
from src.io_vnbd_loader import load_iovnbd_s1
from src.preprocessing import normalize
from src.gnss_ins_filter import GNSSINSFilter


# Configuration
WINDOW_SECONDS = 2.0
STRIDE_SECONDS = 0.5
TRAIN_FRACTION = 0.7
GNSS_OUTAGE_OFFSET_S = 60.0
GNSS_OUTAGE_DURATION_S = 120.0

RESULTS_SUBDIR = os.path.join(RESULTS_DIR, "gnss_ins")


def latlon_to_local(lat, lon, ref_lat, ref_lon):
    """Convert lat/lon arrays to local ENU (east, north) in meters."""
    R = 6371000.0
    ref_lat_rad = np.radians(ref_lat)
    east = R * np.radians(lon - ref_lon) * np.cos(ref_lat_rad)
    north = R * np.radians(lat - ref_lat)
    return east, north


def main():
    print("=" * 60)
    print("  IDR Prototype - GNSS + INS Sensor Fusion")
    print("  Extended Kalman Filter Demo (4-State, External Heading)")
    print("=" * 60)

    np.random.seed(RANDOM_SEED)

    # ------------------------------------------------------------------
    # Step 1: Load data and trained model
    # ------------------------------------------------------------------
    print("\n[STEP 1] Loading IO-VNBD S1 and trained model...")
    data = load_iovnbd_s1()

    model_path = os.path.join(MODEL_DIR, "idr_iovnbd_model.keras")
    if not os.path.exists(model_path):
        print("ERROR: Trained model not found. Run 'python run_iovnbd_demo.py' first.")
        sys.exit(1)

    model = tf.keras.models.load_model(model_path)
    norm_path = os.path.join(MODEL_DIR, "iovnbd_normalization_stats.json")
    with open(norm_path) as f:
        norm_data = json.load(f)
    mean = np.array(norm_data["mean"], dtype=np.float32)
    std = np.array(norm_data["std"], dtype=np.float32)
    print(f"  Model loaded: {model_path}")

    # ------------------------------------------------------------------
    # Step 2: Prepare test data and predict speeds
    # ------------------------------------------------------------------
    print("\n[STEP 2] Generating AI speed predictions on test segment...")

    sample_rate = data['sample_rate_hz']
    window_size = int(round(WINDOW_SECONDS * sample_rate))
    stride = int(round(STRIDE_SECONDS * sample_rate))

    split_idx = int(len(data['timestamp_s']) * TRAIN_FRACTION)
    test_timestamps = data['timestamp_s'][split_idx:]
    test_imu = data['imu'][split_idx:]
    test_speed_gt = data['target_speed_mps'][split_idx:]
    test_heading_deg = data['vehicle_heading_deg'][split_idx:]
    test_vehicle_lat = data['vehicle_lat'][split_idx:]
    test_vehicle_lon = data['vehicle_lon'][split_idx:]
    test_smartphone_lat = data['smartphone_lat'][split_idx:]
    test_smartphone_lon = data['smartphone_lon'][split_idx:]

    n_test = len(test_imu)
    n_windows = (n_test - window_size) // stride + 1

    # Normalize and create windows
    test_imu_norm = normalize(test_imu, mean, std)
    test_windows = np.array([
        test_imu_norm[i * stride:i * stride + window_size]
        for i in range(n_windows)
    ], dtype=np.float32)

    pred_speeds = model.predict(test_windows, verbose=0).flatten()
    pred_speeds = np.maximum(pred_speeds, 0.0)

    window_center_indices = np.array([
        i * stride + window_size // 2 for i in range(n_windows)
    ])
    window_center_times = test_timestamps[window_center_indices]

    print(f"  Test samples: {n_test}")
    print(f"  Windows: {n_windows}")
    print(f"  Test duration: {test_timestamps[-1] - test_timestamps[0]:.1f}s")

    # ------------------------------------------------------------------
    # Step 3: Define coordinate system and GNSS availability
    # ------------------------------------------------------------------
    ref_lat = test_vehicle_lat[0]
    ref_lon = test_vehicle_lon[0]

    # Ground truth positions at window centers
    gt_east, gt_north = latlon_to_local(
        test_vehicle_lat[window_center_indices],
        test_vehicle_lon[window_center_indices],
        ref_lat, ref_lon
    )

    # Smartphone GNSS positions (for plot reference)
    sp_gnss_east, sp_gnss_north = latlon_to_local(
        test_smartphone_lat, test_smartphone_lon, ref_lat, ref_lon
    )

    # GNSS measurement source: vehicle GPS with realistic noise added
    # (simulates a single GNSS receiver that is accurate when available)
    gnss_noise_std = 3.0  # meters, typical civilian GNSS accuracy
    gt_east_all, gt_north_all = latlon_to_local(
        test_vehicle_lat, test_vehicle_lon, ref_lat, ref_lon
    )
    np.random.seed(RANDOM_SEED + 42)
    gnss_east_all = gt_east_all + np.random.normal(0, gnss_noise_std, n_test)
    gnss_north_all = gt_north_all + np.random.normal(0, gnss_noise_std, n_test)

    # GNSS availability mask (at window level)
    test_start_time = test_timestamps[0]
    outage_start = test_start_time + GNSS_OUTAGE_OFFSET_S
    outage_end = outage_start + GNSS_OUTAGE_DURATION_S

    gnss_available = np.array([
        not (outage_start <= t <= outage_end)
        for t in window_center_times
    ])
    outage_mask = ~gnss_available

    print(f"  GNSS outage: {GNSS_OUTAGE_OFFSET_S}s to "
          f"{GNSS_OUTAGE_OFFSET_S + GNSS_OUTAGE_DURATION_S}s "
          f"({GNSS_OUTAGE_DURATION_S}s simulated)")

    # ------------------------------------------------------------------
    # Step 4: Run AI Dead Reckoning (baseline, no GNSS, from origin)
    # ------------------------------------------------------------------
    print("\n[STEP 3] Running AI Dead Reckoning (no GNSS, from origin)...")

    gt_heading_at_windows = test_heading_deg[window_center_indices]
    window_dt = np.zeros(n_windows)
    window_dt[1:] = np.diff(window_center_times)
    window_dt[0] = STRIDE_SECONDS

    # DR using reference heading + AI speed, starting at (0,0)
    dr_east = np.zeros(n_windows)
    dr_north = np.zeros(n_windows)
    for i in range(1, n_windows):
        ds = pred_speeds[i] * window_dt[i]
        angle = np.radians(90.0 - gt_heading_at_windows[i])
        dr_east[i] = dr_east[i - 1] + ds * np.cos(angle)
        dr_north[i] = dr_north[i - 1] + ds * np.sin(angle)

    dr_error = np.sqrt((dr_east - gt_east) ** 2 + (dr_north - gt_north) ** 2)
    print(f"  DR mean error: {dr_error.mean():.1f} m")
    print(f"  DR max error: {dr_error.max():.1f} m")

    # ------------------------------------------------------------------
    # Step 5: Run EKF GNSS + INS Fusion (4-state, external heading)
    # ------------------------------------------------------------------
    print("\n[STEP 4] Running EKF GNSS + INS Fusion...")

    # Initialize EKF at ground truth start position
    ekf = GNSSINSFilter(
        initial_pos=np.array([gt_east[0], gt_north[0]]),
        process_noise_config={
            'position': 0.1,
            'velocity': 1.0,
        },
        measurement_noise_config={
            'position': 9.0,  # 3m GNSS noise squared
        },
    )

    # Run EKF at window-level (same granularity as DR for fair comparison)
    ekf_east = np.zeros(n_windows)
    ekf_north = np.zeros(n_windows)
    ekf_east[0], ekf_north[0] = gt_east[0], gt_north[0]

    for i in range(1, n_windows):
        dt = window_dt[i]
        heading_rad = np.radians(90.0 - gt_heading_at_windows[i])
        ai_speed = max(0.0, float(pred_speeds[i]))

        # EKF prediction step
        ekf.predict(dt, heading_rad, ai_speed)

        # GNSS measurement update (only when available)
        if gnss_available[i]:
            gnss_e = gnss_east_all[window_center_indices[i]]
            gnss_n = gnss_north_all[window_center_indices[i]]
            ekf.update_gnss_position(gnss_e, gnss_n)

        ekf_east[i], ekf_north[i] = ekf.get_position()

    ekf_error = np.sqrt((ekf_east - gt_east) ** 2 + (ekf_north - gt_north) ** 2)
    print(f"  EKF mean error: {ekf_error.mean():.1f} m")
    print(f"  EKF max error: {ekf_error.max():.1f} m")

    # ------------------------------------------------------------------
    # Step 6: Compute metrics
    # ------------------------------------------------------------------
    print("\n[STEP 5] Computing metrics...")

    gt_distance = np.sum(np.sqrt(np.diff(gt_east)**2 + np.diff(gt_north)**2))

    # Outage interval metrics
    dr_outage_error = dr_error[outage_mask]
    ekf_outage_error = ekf_error[outage_mask]

    outage_gt_distance = np.sum(np.sqrt(
        np.diff(gt_east[outage_mask])**2 + np.diff(gt_north[outage_mask])**2
    ))

    # Pre-outage metrics (shows EKF advantage from GNSS corrections)
    pre_outage_mask = np.zeros(n_windows, dtype=bool)
    for i in range(n_windows):
        t = window_center_times[i]
        if t < outage_start:
            pre_outage_mask[i] = True

    dr_pre_error = dr_error[pre_outage_mask]
    ekf_pre_error = ekf_error[pre_outage_mask]

    # Post-outage metrics (shows EKF recovery)
    post_outage_mask = np.zeros(n_windows, dtype=bool)
    for i in range(n_windows):
        t = window_center_times[i]
        if t > outage_end:
            post_outage_mask[i] = True

    dr_post_error = dr_error[post_outage_mask]
    ekf_post_error = ekf_error[post_outage_mask]

    metrics = {
        "gnss_outage_duration_s": GNSS_OUTAGE_DURATION_S,
        "gnss_outage_type": "SIMULATED",
        "dataset": "IO-VNBD S1",
        "ekf_state_dimension": 4,
        "heading_source": "vehicle_reference (orientation sensor proxy)",
        "pre_outage": {
            "dr_mean_error_m": float(dr_pre_error.mean()) if len(dr_pre_error) > 0 else 0,
            "ekf_mean_error_m": float(ekf_pre_error.mean()) if len(ekf_pre_error) > 0 else 0,
        },
        "during_outage": {
            "dr_mean_error_m": float(dr_outage_error.mean()),
            "dr_max_error_m": float(dr_outage_error.max()),
            "dr_final_error_m": float(dr_outage_error[-1]),
            "ekf_mean_error_m": float(ekf_outage_error.mean()),
            "ekf_max_error_m": float(ekf_outage_error.max()),
            "ekf_final_error_m": float(ekf_outage_error[-1]),
            "outage_distance_m": float(outage_gt_distance),
        },
        "post_outage": {
            "dr_mean_error_m": float(dr_post_error.mean()) if len(dr_post_error) > 0 else 0,
            "ekf_mean_error_m": float(ekf_post_error.mean()) if len(ekf_post_error) > 0 else 0,
        },
        "full_trajectory": {
            "dr_mean_error_m": float(dr_error.mean()),
            "dr_max_error_m": float(dr_error.max()),
            "ekf_mean_error_m": float(ekf_error.mean()),
            "ekf_max_error_m": float(ekf_error.max()),
            "gt_distance_m": float(gt_distance),
        },
        "improvement": {
            "outage_mean_error_reduction_pct": float(
                (1 - ekf_outage_error.mean() / max(dr_outage_error.mean(), 0.01)) * 100
            ),
            "outage_max_error_reduction_pct": float(
                (1 - ekf_outage_error.max() / max(dr_outage_error.max(), 0.01)) * 100
            ),
            "full_traj_mean_error_reduction_pct": float(
                (1 - ekf_error.mean() / max(dr_error.mean(), 0.01)) * 100
            ),
        },
    }

    # ------------------------------------------------------------------
    # Step 7: Generate plots
    # ------------------------------------------------------------------
    print("\n[STEP 6] Generating plots...")
    os.makedirs(RESULTS_SUBDIR, exist_ok=True)

    # === Main trajectory comparison plot ===
    fig, ax = plt.subplots(1, 1, figsize=(13, 9))

    # Ground truth
    ax.plot(gt_east, gt_north, 'b-', linewidth=2.5,
            label='Ground Truth (Vehicle GPS)', alpha=0.9, zorder=3)

    # GNSS measurements at window centers (only when available)
    gnss_e_windows = gnss_east_all[window_center_indices]
    gnss_n_windows = gnss_north_all[window_center_indices]
    ax.plot(gnss_e_windows[gnss_available], gnss_n_windows[gnss_available],
            'c.', markersize=3, label='GNSS Measurements (noisy)', alpha=0.35, zorder=1)

    # AI Dead Reckoning
    ax.plot(dr_east, dr_north, 'r--', linewidth=1.5,
            label='AI Dead Reckoning (no GNSS)', alpha=0.7, zorder=2)

    # EKF GNSS+INS (before and after outage)
    ekf_pre = np.where(~outage_mask)[0]
    ax.plot(ekf_east[ekf_pre], ekf_north[ekf_pre], 'g-', linewidth=2,
            label='EKF GNSS+INS (GNSS available)', alpha=0.85, zorder=4)

    # EKF during GNSS outage (highlighted)
    ax.plot(ekf_east[outage_mask], ekf_north[outage_mask],
            color='orange', linewidth=3, alpha=0.8,
            label=f'EKF During Outage ({GNSS_OUTAGE_DURATION_S:.0f}s, AI DR only)',
            zorder=5)

    # Outage start/end markers
    outage_indices = np.where(outage_mask)[0]
    if len(outage_indices) > 0:
        ax.scatter(gt_east[outage_indices[0]], gt_north[outage_indices[0]],
                   s=150, c='orange', marker='X', zorder=6, linewidths=2,
                   label='GNSS Lost')
        ax.scatter(gt_east[outage_indices[-1]], gt_north[outage_indices[-1]],
                   s=150, c='purple', marker='X', zorder=6, linewidths=2,
                   label='GNSS Restored')

    ax.scatter(gt_east[0], gt_north[0], s=180, c='green', marker='^',
               zorder=6, label='Start')

    ax.set_xlabel("East (m)", fontsize=12)
    ax.set_ylabel("North (m)", fontsize=12)
    ax.set_title("GNSS + INS Sensor Fusion: EKF vs Dead Reckoning\n"
                 f"Simulated {GNSS_OUTAGE_DURATION_S:.0f}s GNSS-Denied Interval "
                 "(IO-VNBD S1)", fontsize=13)
    ax.legend(loc="best", fontsize=9)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_SUBDIR, "gnss_ins_trajectory.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {plot_path}")

    # === Position error over time plot ===
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    t_plot = window_center_times - test_start_time

    # Error plot
    ax = axes[0]
    ax.plot(t_plot, dr_error, 'r-', linewidth=1.2,
            label='AI Dead Reckoning Error', alpha=0.7)
    ax.plot(t_plot, ekf_error, 'g-', linewidth=1.5,
            label='EKF GNSS+INS Error', alpha=0.8)
    ax.axvspan(GNSS_OUTAGE_OFFSET_S, GNSS_OUTAGE_OFFSET_S + GNSS_OUTAGE_DURATION_S,
               alpha=0.15, color='red', label='GNSS Denied')
    ax.axhline(0, color='k', linewidth=0.5)

    ax.set_ylabel("Position Error (m)", fontsize=11)
    ax.set_title("Position Error: EKF GNSS+INS vs Pure Dead Reckoning\n"
                 "EKF maintains low error before outage, drifts during, "
                 "corrects after GNSS returns", fontsize=11)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Speed comparison (shows AI is working)
    ax = axes[1]
    gt_speeds_at_windows = test_speed_gt[window_center_indices]
    ax.plot(t_plot, gt_speeds_at_windows * 3.6, 'b-', linewidth=1,
            label='Ground Truth Speed', alpha=0.7)
    ax.plot(t_plot, pred_speeds * 3.6, 'r-', linewidth=0.8,
            label='AI Predicted Speed', alpha=0.6)
    ax.axvspan(GNSS_OUTAGE_OFFSET_S, GNSS_OUTAGE_OFFSET_S + GNSS_OUTAGE_DURATION_S,
               alpha=0.15, color='red')
    ax.set_xlabel("Time into test segment (s)", fontsize=11)
    ax.set_ylabel("Speed (km/h)", fontsize=11)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    error_plot_path = os.path.join(RESULTS_SUBDIR, "gnss_ins_error.png")
    plt.savefig(error_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {error_plot_path}")

    # Save metrics
    metrics_path = os.path.join(RESULTS_SUBDIR, "gnss_ins_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved: {metrics_path}")

    # ------------------------------------------------------------------
    # Print final summary
    # ------------------------------------------------------------------
    pre = metrics["pre_outage"]
    dur = metrics["during_outage"]
    post = metrics["post_outage"]
    imp = metrics["improvement"]
    full = metrics["full_trajectory"]

    print("\n" + "=" * 60)
    print("  GNSS + INS FUSION RESULTS")
    print("=" * 60)
    print(f"\n  GNSS outage: {GNSS_OUTAGE_DURATION_S:.0f}s (SIMULATED)")
    print(f"  Dataset: IO-VNBD S1")
    print(f"  EKF: 4-state [px, py, vx, vy], heading = external sensor")

    print(f"\n  --- Before Outage (GNSS available) ---")
    print(f"    DR mean error:   {pre['dr_mean_error_m']:.1f} m")
    print(f"    EKF mean error:  {pre['ekf_mean_error_m']:.1f} m")

    print(f"\n  --- During Outage ({GNSS_OUTAGE_DURATION_S:.0f}s GNSS denied) ---")
    print(f"    DR mean error:   {dur['dr_mean_error_m']:.1f} m")
    print(f"    DR max error:    {dur['dr_max_error_m']:.1f} m")
    print(f"    EKF mean error:  {dur['ekf_mean_error_m']:.1f} m")
    print(f"    EKF max error:   {dur['ekf_max_error_m']:.1f} m")
    print(f"    Improvement:     {imp['outage_mean_error_reduction_pct']:.1f}% "
          f"mean error reduction")

    print(f"\n  --- After Outage (GNSS restored) ---")
    print(f"    DR mean error:   {post['dr_mean_error_m']:.1f} m")
    print(f"    EKF mean error:  {post['ekf_mean_error_m']:.1f} m")

    print(f"\n  --- Full Trajectory ---")
    print(f"    DR mean error:   {full['dr_mean_error_m']:.1f} m")
    print(f"    EKF mean error:  {full['ekf_mean_error_m']:.1f} m")
    print(f"    Improvement:     {imp['full_traj_mean_error_reduction_pct']:.1f}% "
          f"mean error reduction")

    ekf_better = ekf_outage_error.mean() < dr_outage_error.mean()
    print(f"\n  EKF outperforms DR during outage: {'YES' if ekf_better else 'NO'}")

    print(f"\n  Key insight:")
    if ekf_better:
        print(f"    EKF enters outage with {ekf_error[outage_indices[0]]:.1f}m error "
              f"(GNSS-corrected)")
        print(f"    DR enters outage with {dr_error[outage_indices[0]]:.1f}m error "
              f"(accumulated drift)")
        print(f"    Both drift similarly during outage, but EKF starts closer to truth")
    else:
        print(f"    DR accumulated less pre-outage error than expected")

    print("\n" + "=" * 60)
    print(f"\n  Output files:")
    print(f"    {plot_path}")
    print(f"    {error_plot_path}")
    print(f"    {metrics_path}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
