"""
Trajectory Visualization and Metrics for IDR Prototype.

Generates publication-quality plots showing:
  - Ground-truth vs estimated trajectory
  - GNSS-denied interval highlighting
  - Speed estimation accuracy
  - Error metrics
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from src.config import (
    RESULTS_DIR, GNSS_OUTAGE_START_S, GNSS_OUTAGE_END_S,
)


def compute_trajectory_metrics(results):
    """
    Compute error metrics between ground-truth and dead-reckoned trajectory.

    Returns dict of metrics.
    """
    metrics = {}

    if results["gt_speed"] is not None:
        speed_error = results["pred_speeds"] - results["gt_speed"]
        metrics["speed_mae_ms"] = float(np.mean(np.abs(speed_error)))
        metrics["speed_rmse_ms"] = float(np.sqrt(np.mean(speed_error ** 2)))

    if results["gt_x"] is not None:
        pos_error = np.sqrt(
            (results["dr_x"] - results["gt_x"]) ** 2 +
            (results["dr_y"] - results["gt_y"]) ** 2
        )
        metrics["position_mae_m"] = float(np.mean(pos_error))
        metrics["position_rmse_m"] = float(np.sqrt(np.mean(pos_error ** 2)))
        metrics["max_position_error_m"] = float(np.max(pos_error))
        metrics["final_position_error_m"] = float(pos_error[-1])

        # Total distance comparison
        gt_dist = np.sum(np.sqrt(
            np.diff(results["gt_x"]) ** 2 + np.diff(results["gt_y"]) ** 2
        ))
        dr_dist = results["dr_distance"][-1]
        metrics["gt_total_distance_m"] = float(gt_dist)
        metrics["dr_total_distance_m"] = float(dr_dist)
        metrics["distance_error_pct"] = float(
            abs(dr_dist - gt_dist) / max(gt_dist, 1e-6) * 100
        )

    # GNSS-denied interval metrics
    if results["gt_x"] is not None:
        mask = (results["window_times"] >= GNSS_OUTAGE_START_S) & \
               (results["window_times"] <= GNSS_OUTAGE_END_S)
        if np.any(mask):
            gnss_denied_error = np.sqrt(
                (results["gnss_x"][mask] - results["gt_x"][mask]) ** 2 +
                (results["gnss_y"][mask] - results["gt_y"][mask]) ** 2
            )
            metrics["gnss_denied_mae_m"] = float(np.mean(gnss_denied_error))
            metrics["gnss_denied_max_error_m"] = float(np.max(gnss_denied_error))

    return metrics


def plot_trajectory(results, save_path=None):
    """
    Plot 2D trajectory comparison: ground truth vs dead reckoning vs GNSS fusion.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Ground truth
    if results["gt_x"] is not None:
        ax.plot(results["gt_x"], results["gt_y"],
                "b-", linewidth=2, label="Ground Truth", alpha=0.8)

    # Dead reckoning (full)
    ax.plot(results["dr_x"], results["dr_y"],
            "r--", linewidth=1.5, label="AI Dead Reckoning", alpha=0.7)

    # GNSS fusion result
    ax.plot(results["gnss_x"], results["gnss_y"],
            "g-", linewidth=1.5, label="GNSS + DR Fusion", alpha=0.7)

    # Highlight GNSS-denied region
    mask = (results["window_times"] >= GNSS_OUTAGE_START_S) & \
           (results["window_times"] <= GNSS_OUTAGE_END_S)
    if np.any(mask) and results["gt_x"] is not None:
        ax.scatter(results["gt_x"][mask][0], results["gt_y"][mask][0],
                   s=100, c="orange", marker="x", zorder=5,
                   label="GNSS Denied Start")
        ax.scatter(results["gt_x"][mask][-1], results["gt_y"][mask][-1],
                   s=100, c="purple", marker="x", zorder=5,
                   label="GNSS Denied End")
        # Shade the DR path during outage
        ax.plot(results["gnss_x"][mask], results["gnss_y"][mask],
                "orange", linewidth=3, alpha=0.6, label="DR During Outage")

    # Start/end markers
    ax.scatter(results["dr_x"][0], results["dr_y"][0],
               s=150, c="green", marker="^", zorder=5, label="Start")
    ax.scatter(results["dr_x"][-1], results["dr_y"][-1],
               s=150, c="red", marker="v", zorder=5, label="End")

    ax.set_xlabel("X position (m) [Forward]", fontsize=12)
    ax.set_ylabel("Y position (m) [Left]", fontsize=12)
    ax.set_title("IDR Prototype: AI Dead Reckoning Trajectory\n"
                 "(SYNTHETIC DEMONSTRATION DATA)", fontsize=13)
    ax.legend(loc="best", fontsize=9)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Trajectory plot saved: {save_path}")
    plt.close()


def plot_speed_comparison(results, save_path=None):
    """Plot predicted vs ground-truth speed over time."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    t = results["window_times"]

    # Speed comparison
    ax = axes[0]
    if results["gt_speed"] is not None:
        ax.plot(t, results["gt_speed"], "b-", linewidth=1.5,
                label="Ground Truth Speed", alpha=0.8)
    ax.plot(t, results["pred_speeds"], "r-", linewidth=1,
            label="Predicted Speed", alpha=0.7)

    # Shade GNSS outage
    ax.axvspan(GNSS_OUTAGE_START_S, GNSS_OUTAGE_END_S,
               alpha=0.1, color="red", label="GNSS Denied")
    ax.set_ylabel("Speed (m/s)", fontsize=11)
    ax.set_title("Speed Estimation: AI Prediction vs Ground Truth\n"
                 "(SYNTHETIC DEMONSTRATION DATA)", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Speed error
    ax = axes[1]
    if results["gt_speed"] is not None:
        error = results["pred_speeds"] - results["gt_speed"]
        ax.plot(t, error, "r-", linewidth=0.8, alpha=0.7)
        ax.axhline(0, color="k", linewidth=0.5)
        ax.axvspan(GNSS_OUTAGE_START_S, GNSS_OUTAGE_END_S,
                   alpha=0.1, color="red")
        ax.set_ylabel("Speed Error (m/s)", fontsize=11)

    ax.set_xlabel("Time (s)", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Speed plot saved: {save_path}")
    plt.close()


def plot_position_error(results, save_path=None):
    """Plot position error over time."""
    if results["gt_x"] is None:
        return

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))

    t = results["window_times"]
    pos_error = np.sqrt(
        (results["dr_x"] - results["gt_x"]) ** 2 +
        (results["dr_y"] - results["gt_y"]) ** 2
    )

    ax.plot(t, pos_error, "r-", linewidth=1.2, label="Position Error (DR)")
    ax.axvspan(GNSS_OUTAGE_START_S, GNSS_OUTAGE_END_S,
               alpha=0.1, color="red", label="GNSS Denied")
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Position Error (m)", fontsize=11)
    ax.set_title("Dead Reckoning Position Error Over Time\n"
                 "(SYNTHETIC DEMONSTRATION DATA)", fontsize=12)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Error plot saved: {save_path}")
    plt.close()


def generate_all_plots(results, prefix="demo"):
    """Generate and save all visualization plots."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    plot_trajectory(
        results,
        save_path=os.path.join(RESULTS_DIR, f"{prefix}_trajectory.png")
    )
    plot_speed_comparison(
        results,
        save_path=os.path.join(RESULTS_DIR, f"{prefix}_speed.png")
    )
    plot_position_error(
        results,
        save_path=os.path.join(RESULTS_DIR, f"{prefix}_error.png")
    )


def print_metrics_report(metrics):
    """Print a formatted metrics report."""
    print("\n" + "=" * 60)
    print("  IDR PROTOTYPE - PERFORMANCE METRICS")
    print("  (SYNTHETIC DEMONSTRATION DATA - NOT REAL-WORLD ACCURACY)")
    print("=" * 60)

    if "speed_mae_ms" in metrics:
        print(f"\n  Speed Estimation:")
        print(f"    MAE:  {metrics['speed_mae_ms']:.4f} m/s")
        print(f"    RMSE: {metrics['speed_rmse_ms']:.4f} m/s")

    if "position_mae_m" in metrics:
        print(f"\n  Position (Dead Reckoning):")
        print(f"    MAE:              {metrics['position_mae_m']:.2f} m")
        print(f"    RMSE:             {metrics['position_rmse_m']:.2f} m")
        print(f"    Max error:        {metrics['max_position_error_m']:.2f} m")
        print(f"    Final error:      {metrics['final_position_error_m']:.2f} m")

    if "gt_total_distance_m" in metrics:
        print(f"\n  Distance:")
        print(f"    Ground truth:     {metrics['gt_total_distance_m']:.1f} m")
        print(f"    Dead reckoned:    {metrics['dr_total_distance_m']:.1f} m")
        print(f"    Error:            {metrics['distance_error_pct']:.1f}%")

    if "gnss_denied_mae_m" in metrics:
        print(f"\n  GNSS-Denied Interval ({GNSS_OUTAGE_START_S}s - {GNSS_OUTAGE_END_S}s):")
        print(f"    Mean error:       {metrics['gnss_denied_mae_m']:.2f} m")
        print(f"    Max error:        {metrics['gnss_denied_max_error_m']:.2f} m")

    print("\n" + "=" * 60)
