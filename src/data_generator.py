"""
Synthetic IMU Data Generator for IDR Prototype.

Generates realistic 6-axis IMU data simulating vehicle/pedestrian motion
with ground-truth speed and trajectory for training and evaluation.

WARNING: This is DEMONSTRATION DATA for prototype development.
Results from synthetic data do NOT represent real-world accuracy.
"""
import numpy as np
import pandas as pd
import os
from src.config import (
    SAMPLE_RATE_HZ, DATA_DIR, RANDOM_SEED,
    ACC_NOISE_STD, GYR_NOISE_STD, ACC_BIAS, GYR_BIAS,
    SYNTH_DURATION_S, SYNTH_NUM_SEQUENCES,
)


def generate_motion_profile(duration_s, sample_rate, rng):
    """
    Generate a realistic motion profile with varied phases.

    Returns arrays of ground-truth speed (m/s) and yaw_rate (rad/s)
    at each timestep.
    """
    n_samples = int(duration_s * sample_rate)
    dt = 1.0 / sample_rate

    speed = np.zeros(n_samples)
    yaw_rate = np.zeros(n_samples)

    # Define motion phases: (type, duration_seconds)
    phases = []
    t_remaining = duration_s
    while t_remaining > 1.0:
        phase_type = rng.choice(
            ["stationary", "accelerate", "constant", "decelerate", "turn"],
            p=[0.10, 0.20, 0.35, 0.15, 0.20],
        )
        max_dur = min(20.0, t_remaining)
        min_dur = min(3.0, max_dur)
        phase_dur = rng.uniform(min_dur, max_dur)
        phases.append((phase_type, phase_dur))
        t_remaining -= phase_dur

    idx = 0
    current_speed = 0.0
    current_yaw_rate = 0.0

    for phase_type, phase_dur in phases:
        n_phase = int(phase_dur * sample_rate)
        if idx + n_phase > n_samples:
            n_phase = n_samples - idx
        if n_phase <= 0:
            break

        if phase_type == "stationary":
            target_speed = 0.0
            target_yaw = 0.0
            for i in range(n_phase):
                current_speed += (target_speed - current_speed) * 0.05
                current_yaw_rate *= 0.95
                speed[idx + i] = max(0.0, current_speed)
                yaw_rate[idx + i] = current_yaw_rate
        elif phase_type == "accelerate":
            target_speed = rng.uniform(2.0, 12.0)  # m/s (7-43 km/h)
            for i in range(n_phase):
                accel_rate = rng.uniform(0.3, 1.5) / sample_rate
                current_speed = min(current_speed + accel_rate, target_speed)
                current_yaw_rate *= 0.98
                speed[idx + i] = current_speed
                yaw_rate[idx + i] = current_yaw_rate
        elif phase_type == "constant":
            target_yaw = 0.0
            for i in range(n_phase):
                # Small speed fluctuations
                current_speed += rng.normal(0, 0.01)
                current_speed = max(0.5, current_speed)
                current_yaw_rate *= 0.95
                speed[idx + i] = current_speed
                yaw_rate[idx + i] = current_yaw_rate
        elif phase_type == "decelerate":
            target_speed = rng.uniform(0.0, current_speed * 0.3)
            for i in range(n_phase):
                decel_rate = rng.uniform(0.5, 2.0) / sample_rate
                current_speed = max(target_speed, current_speed - decel_rate)
                current_yaw_rate *= 0.98
                speed[idx + i] = current_speed
                yaw_rate[idx + i] = current_yaw_rate
        elif phase_type == "turn":
            turn_direction = rng.choice([-1, 1])
            turn_magnitude = rng.uniform(0.1, 0.5)  # rad/s
            for i in range(n_phase):
                # Smooth turn entry/exit
                progress = i / n_phase
                envelope = np.sin(progress * np.pi)
                current_yaw_rate = turn_direction * turn_magnitude * envelope
                # Slow down slightly during turn
                current_speed *= (1.0 - 0.001 * abs(current_yaw_rate))
                current_speed = max(0.0, current_speed)
                speed[idx + i] = current_speed
                yaw_rate[idx + i] = current_yaw_rate

        idx += n_phase

    return speed[:n_samples], yaw_rate[:n_samples]


def compute_ground_truth_trajectory(speed, yaw_rate, sample_rate):
    """
    Integrate speed and yaw_rate to produce ground-truth (x, y) trajectory.

    Coordinate frame:
        X = forward (initial heading direction)
        Y = left
        Heading theta measured from +X, CCW positive
    """
    n = len(speed)
    dt = 1.0 / sample_rate
    x = np.zeros(n)
    y = np.zeros(n)
    heading = np.zeros(n)

    for i in range(1, n):
        heading[i] = heading[i - 1] + yaw_rate[i] * dt
        x[i] = x[i - 1] + speed[i] * np.cos(heading[i]) * dt
        y[i] = y[i - 1] + speed[i] * np.sin(heading[i]) * dt

    return x, y, heading


def generate_imu_from_motion(speed, yaw_rate, sample_rate, rng):
    """
    Generate synthetic 6-axis IMU readings from ground-truth motion.

    Physics model:
        ax = d(speed)/dt + vibration(speed) + noise + bias
        ay = speed * yaw_rate + vibration(speed) + noise + bias
        az = gravity (9.81) + vibration(speed) + noise + bias
        gx = vibration(speed) + noise + bias  (roll rate ~ 0)
        gy = vibration(speed) + noise + bias  (pitch rate ~ 0)
        gz = yaw_rate + noise + bias

    Speed-dependent vibration models real-world phenomena:
    tire rotation, engine vibration, road roughness all scale with speed.
    This gives the neural network a learnable signal for speed estimation
    even during constant-velocity phases.
    """
    n = len(speed)
    dt = 1.0 / sample_rate

    # Ground-truth accelerations
    forward_accel = np.gradient(speed, dt)
    # Clip extreme values from numerical differentiation
    forward_accel = np.clip(forward_accel, -5.0, 5.0)

    # Centripetal acceleration (felt as lateral force)
    lateral_accel = speed * yaw_rate

    # Sensor biases (constant per sequence, random per sensor)
    acc_bias_x = rng.uniform(-ACC_BIAS, ACC_BIAS)
    acc_bias_y = rng.uniform(-ACC_BIAS, ACC_BIAS)
    acc_bias_z = rng.uniform(-ACC_BIAS, ACC_BIAS)
    gyr_bias_x = rng.uniform(-GYR_BIAS, GYR_BIAS)
    gyr_bias_y = rng.uniform(-GYR_BIAS, GYR_BIAS)
    gyr_bias_z = rng.uniform(-GYR_BIAS, GYR_BIAS)

    # Speed-dependent vibration: higher speed → more vibration
    # Models road roughness, tire rotation, engine harmonics
    vib_freq_base = rng.uniform(15.0, 40.0)  # Hz, varies per sequence
    t = np.arange(n) * dt
    speed_factor = speed / 10.0  # Normalize: 10 m/s → factor 1.0

    acc_vib_amplitude = 0.08 * speed_factor  # m/s^2
    gyr_vib_amplitude = 0.003 * speed_factor  # rad/s

    # Multi-frequency vibration (simulates multiple harmonics)
    vib_acc = (
        acc_vib_amplitude * np.sin(2 * np.pi * vib_freq_base * t) +
        0.5 * acc_vib_amplitude * np.sin(2 * np.pi * 2.3 * vib_freq_base * t) +
        0.3 * acc_vib_amplitude * rng.normal(0, 1, n)
    )
    vib_gyr = (
        gyr_vib_amplitude * np.sin(2 * np.pi * vib_freq_base * 0.8 * t) +
        0.4 * gyr_vib_amplitude * rng.normal(0, 1, n)
    )

    # IMU readings = true value + vibration + bias + noise
    ax = forward_accel + vib_acc + acc_bias_x + rng.normal(0, ACC_NOISE_STD, n)
    ay = lateral_accel + vib_acc * 0.6 + acc_bias_y + rng.normal(0, ACC_NOISE_STD, n)
    az = 9.81 + vib_acc * 0.8 + acc_bias_z + rng.normal(0, ACC_NOISE_STD * 0.5, n)
    gx = vib_gyr + gyr_bias_x + rng.normal(0, GYR_NOISE_STD, n)
    gy = vib_gyr * 0.7 + gyr_bias_y + rng.normal(0, GYR_NOISE_STD, n)
    gz = yaw_rate + vib_gyr * 0.3 + gyr_bias_z + rng.normal(0, GYR_NOISE_STD, n)

    return ax, ay, az, gx, gy, gz


def generate_dataset(num_sequences=None, duration_s=None, sample_rate=None):
    """
    Generate the full synthetic dataset and save to CSV.

    Returns list of file paths for generated sequences.
    """
    if num_sequences is None:
        num_sequences = SYNTH_NUM_SEQUENCES
    if duration_s is None:
        duration_s = SYNTH_DURATION_S
    if sample_rate is None:
        sample_rate = SAMPLE_RATE_HZ

    rng = np.random.default_rng(RANDOM_SEED)
    os.makedirs(DATA_DIR, exist_ok=True)

    generated_files = []

    for seq_idx in range(num_sequences):
        speed, yaw_rate = generate_motion_profile(duration_s, sample_rate, rng)
        x, y, heading = compute_ground_truth_trajectory(speed, yaw_rate, sample_rate)
        ax, ay, az, gx, gy, gz = generate_imu_from_motion(
            speed, yaw_rate, sample_rate, rng
        )

        n_samples = len(speed)
        timestamps = np.arange(n_samples) / sample_rate

        df = pd.DataFrame({
            "timestamp": timestamps,
            "ax": ax,
            "ay": ay,
            "az": az,
            "gx": gx,
            "gy": gy,
            "gz": gz,
            "gt_speed": speed,
            "gt_yaw_rate": yaw_rate,
            "gt_x": x,
            "gt_y": y,
            "gt_heading": heading,
        })

        filename = f"synthetic_sequence_{seq_idx:03d}.csv"
        filepath = os.path.join(DATA_DIR, filename)
        df.to_csv(filepath, index=False)
        generated_files.append(filepath)
        print(f"  Generated: {filename} ({n_samples} samples, {duration_s}s)")

    return generated_files


if __name__ == "__main__":
    print("Generating synthetic IMU dataset...")
    files = generate_dataset()
    print(f"\nGenerated {len(files)} sequences in {DATA_DIR}")
