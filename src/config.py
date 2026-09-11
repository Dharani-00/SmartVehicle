"""
Configuration for IDR (Intelligent Dead Reckoning) Prototype.

All tunable parameters in one place.
"""
import os

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "synthetic")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# ----------------------------------------------------------------------
# IMU / Sensor Configuration
# ----------------------------------------------------------------------
SAMPLE_RATE_HZ = 200          # IMU sampling frequency (Hz)
WINDOW_SIZE = 200             # Samples per prediction window (1 second at 200Hz)
STRIDE = 50                   # Stride between consecutive windows (0.25s)

# Coordinate convention:
#   X = forward (direction of travel)
#   Y = left
#   Z = up
#   Heading theta = yaw from +X axis, counter-clockwise positive

# ----------------------------------------------------------------------
# Model Architecture
# ----------------------------------------------------------------------
CONV_FILTERS = [64, 64, 128]  # Filters per Conv1D layer
KERNEL_SIZE = 5               # Conv1D kernel size
LSTM_UNITS = 64               # LSTM hidden units
DENSE_UNITS = 32              # Dense layer units before output
DROPOUT_RATE = 0.2            # Dropout for regularization

# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 20
EARLY_STOP_PATIENCE = 5
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# ----------------------------------------------------------------------
# Synthetic Data Generation
# ----------------------------------------------------------------------
SYNTH_DURATION_S = 300        # Total duration of synthetic route (seconds)
SYNTH_NUM_SEQUENCES = 8       # Number of synthetic sequences to generate

# IMU noise parameters (realistic smartphone-grade MEMS)
ACC_NOISE_STD = 0.05          # Accelerometer noise std (m/s^2)
GYR_NOISE_STD = 0.005         # Gyroscope noise std (rad/s)
ACC_BIAS = 0.02               # Accelerometer bias (m/s^2)
GYR_BIAS = 0.001              # Gyroscope bias (rad/s)

# ----------------------------------------------------------------------
# GNSS Simulation
# ----------------------------------------------------------------------
GNSS_OUTAGE_START_S = 100.0   # Start of simulated GNSS denial (seconds)
GNSS_OUTAGE_END_S = 200.0     # End of simulated GNSS denial (seconds)
