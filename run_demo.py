"""
IDR Prototype - Complete Demo Pipeline

Runs the full end-to-end demonstration:
  1. Generate synthetic IMU data
  2. Train the CNN-LSTM speed estimation model
  3. Run inference on test sequence
  4. Simulate GNSS-denied interval
  5. Generate trajectory visualization
  6. Report metrics
  7. Export TFLite model

Usage:
    python run_demo.py

NOTE: Results are from SYNTHETIC DEMONSTRATION DATA.
      They do NOT represent real-world accuracy.
"""
import os
import sys
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    DATA_DIR, MODEL_DIR, RESULTS_DIR,
    GNSS_OUTAGE_START_S, GNSS_OUTAGE_END_S,
)


def main():
    print("=" * 60)
    print("  AI-Enhanced Intelligent Dead Reckoning (IDR)")
    print("  Navigation System for GNSS-Denied Environments")
    print("  --- PROTOTYPE DEMONSTRATION ---")
    print("=" * 60)
    print("\n  NOTE: Using SYNTHETIC data for demonstration.")
    print("  Results do NOT represent real-world accuracy.\n")

    # ----------------------------------------------------------------
    # Step 1: Generate synthetic IMU data
    # ----------------------------------------------------------------
    print("\n[STEP 1] Generating synthetic IMU dataset...")
    from src.data_generator import generate_dataset
    data_files = generate_dataset()
    print(f"  Generated {len(data_files)} sequences")

    # ----------------------------------------------------------------
    # Step 2: Train the model
    # ----------------------------------------------------------------
    print("\n[STEP 2] Training CNN-LSTM speed estimation model...")
    from src.train import train_model

    # Use first 4 sequences for training, last 1 for inference demo
    train_files = data_files[:-1]
    test_file = data_files[-1]

    model, history, train_metrics, (mean, std) = train_model(train_files)

    # ----------------------------------------------------------------
    # Step 3: Run inference on test sequence
    # ----------------------------------------------------------------
    print("\n[STEP 3] Running inference on test sequence...")
    from src.inference import run_inference_on_sequence

    results = run_inference_on_sequence(test_file, model, mean, std)
    print(f"  Predicted {len(results['pred_speeds'])} speed windows")
    print(f"  Trajectory: ({results['dr_x'][-1]:.1f}, {results['dr_y'][-1]:.1f}) m")
    print(f"  Total DR distance: {results['dr_distance'][-1]:.1f} m")

    # ----------------------------------------------------------------
    # Step 4: Compute metrics
    # ----------------------------------------------------------------
    print("\n[STEP 4] Computing metrics...")
    from src.trajectory import (
        compute_trajectory_metrics, print_metrics_report,
        generate_all_plots,
    )

    metrics = compute_trajectory_metrics(results)
    print_metrics_report(metrics)

    # Save metrics
    os.makedirs(RESULTS_DIR, exist_ok=True)
    metrics_path = os.path.join(RESULTS_DIR, "inference_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics saved: {metrics_path}")

    # ----------------------------------------------------------------
    # Step 5: Generate trajectory visualizations
    # ----------------------------------------------------------------
    print("\n[STEP 5] Generating trajectory plots...")
    generate_all_plots(results, prefix="demo")

    # ----------------------------------------------------------------
    # Step 6: Export TFLite model
    # ----------------------------------------------------------------
    print("\n[STEP 6] Exporting TFLite model...")
    from src.export_tflite import export_to_tflite
    tflite_path = export_to_tflite()

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  DEMO COMPLETE")
    print("=" * 60)
    print(f"\n  Artifacts produced:")
    print(f"    - Keras model:     models/idr_speed_model.keras")
    print(f"    - TFLite model:    models/idr_speed_model.tflite")
    print(f"    - Trajectory plot: results/demo_trajectory.png")
    print(f"    - Speed plot:      results/demo_speed.png")
    print(f"    - Error plot:      results/demo_error.png")
    print(f"    - Metrics:         results/inference_metrics.json")
    print(f"\n  GNSS outage simulated: {GNSS_OUTAGE_START_S}s - {GNSS_OUTAGE_END_S}s")
    print(f"\n  REMINDER: These results are from SYNTHETIC data.")
    print(f"  Replace with real IMU data for actual performance evaluation.")
    print("=" * 60)


if __name__ == "__main__":
    main()
