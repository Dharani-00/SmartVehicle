# AI-Enhanced Intelligent Dead Reckoning (IDR) - Prototype

**Navigation System for GNSS-Denied Environments**

## Overview

This prototype demonstrates AI-based dead reckoning using smartphone IMU data.
A CNN-LSTM neural network predicts forward speed from accelerometer and gyroscope
readings, which is then integrated with gyro-derived heading to produce a 2D
trajectory estimate.

**Current Status:** Baseline prototype with synthetic demonstration data.

## Quick Start

```bash
cd idr-prototype
pip install -r requirements.txt
python run_demo.py
```

This will:
1. Generate synthetic IMU data (vehicle motion simulation)
2. Train the CNN-LSTM speed estimation model (~20 epochs)
3. Run inference and produce dead-reckoned trajectory
4. Simulate a 100-second GNSS outage (100s-200s)
5. Generate trajectory plots and error metrics
6. Export a TFLite model for mobile deployment

## Architecture

```
IMU (ax,ay,az,gx,gy,gz) @ 200Hz
        │
        ▼
┌─────────────────────┐
│  Preprocessing      │  Normalization + Sliding Window
│  (window=200, 6ch)  │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  CNN-LSTM Model     │  Conv1D → Conv1D → Conv1D → LSTM → Dense
│  (~150K params)     │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Speed Estimate     │  Forward velocity (m/s)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Dead Reckoning     │  Gyro heading + speed → x,y trajectory
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  GNSS Fusion Demo   │  Reference when available, DR when denied
└─────────────────────┘
```

## Project Structure

```
idr-prototype/
├── data/
│   ├── synthetic/          # Generated demo data (CSV)
│   └── README.md           # Data format documentation
├── models/                 # Saved models (.keras, .tflite)
├── results/                # Plots and metrics
├── src/
│   ├── config.py           # All configuration parameters
│   ├── data_generator.py   # Synthetic IMU data generation
│   ├── preprocessing.py    # Normalization, windowing, splitting
│   ├── model.py            # CNN-LSTM architecture definition
│   ├── train.py            # Training pipeline
│   ├── inference.py        # Inference + dead reckoning
│   ├── trajectory.py       # Visualization + metrics
│   └── export_tflite.py    # TFLite conversion
├── requirements.txt
├── run_demo.py             # End-to-end demo script
└── README.md
```

## Coordinate Convention

- **X** = forward (direction of travel)
- **Y** = left
- **Z** = up
- **Heading** = yaw angle from +X axis, counter-clockwise positive

The IMU is assumed aligned with the vehicle/body frame. Phone-to-vehicle
alignment calibration will be added in a future iteration.

## Configuration

All parameters are in `src/config.py`:
- Sensor: sample rate, window size, stride
- Model: filters, kernel size, LSTM units
- Training: learning rate, batch size, epochs, early stopping
- Synthetic data: noise, bias, duration

## Future Extensions

This prototype is designed as a foundation for:

1. **Real IMU Data (IO-VNBD)** - Replace synthetic generator with real dataset loader
2. **GNSS+INS Fusion (ESKF)** - Error-State Kalman Filter combining GPS + IMU
3. **Map Matching** - Constrain trajectory to road network
4. **Non-Holonomic Constraints** - Vehicle motion model (no lateral slip)
5. **Phone-to-Vehicle Alignment** - Automatic IMU orientation calibration
6. **Android Deployment** - TFLite model in mobile app with real-time inference

## Important Notes

- Results shown are from **SYNTHETIC** demonstration data
- Synthetic results do **NOT** represent real-world accuracy
- The GNSS fusion is a **demonstration** of architecture, not a complete implementation
- Model is intentionally lightweight for smartphone inference
