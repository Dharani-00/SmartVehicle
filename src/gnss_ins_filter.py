"""
GNSS + INS Extended Kalman Filter for IDR Prototype.

Fuses:
  - Orientation sensor heading (from IMU orientation filter)
  - AI-predicted forward velocity (from CNN-LSTM)
  - GNSS position measurements (when available)

State vector [4]:
  x[0] = px      (position east, meters)
  x[1] = py      (position north, meters)
  x[2] = vx      (velocity east, m/s)
  x[3] = vy      (velocity north, m/s)

Heading is provided externally from an orientation sensor/filter.
This avoids gyro integration drift issues in a prototype.
A future ESKF implementation will estimate heading in-state.

Coordinate frame:
  East-North-Up (ENU) local tangent plane

The AI model does NOT receive GNSS input.
GNSS is used ONLY as a measurement update in this filter.
"""
import numpy as np


class GNSSINSFilter:
    """
    Extended Kalman Filter for GNSS + INS fusion.

    Prediction: AI-predicted speed + external heading → velocity → position
    Update: GNSS position (when available)
    """

    def __init__(self, initial_pos, process_noise_config=None,
                 measurement_noise_config=None):
        """
        Initialize the EKF.

        Args:
            initial_pos: (2,) initial [east, north] in meters
            process_noise_config: dict of process noise parameters
            measurement_noise_config: dict of measurement noise parameters
        """
        # State vector: [px, py, vx, vy]
        self.n_states = 4
        self.x = np.zeros(self.n_states)
        self.x[0] = initial_pos[0]
        self.x[1] = initial_pos[1]

        # State covariance
        self.P = np.diag([
            4.0,    # px uncertainty (m^2)
            4.0,    # py uncertainty (m^2)
            2.0,    # vx uncertainty (m/s)^2
            2.0,    # vy uncertainty (m/s)^2
        ])

        # Process noise parameters
        pn = process_noise_config or {}
        self.q_pos = pn.get('position', 0.5)         # m^2/s
        self.q_vel = pn.get('velocity', 8.0)         # (m/s)^2/s — accounts for
                                                      # AI speed uncertainty

        # Measurement noise parameters
        mn = measurement_noise_config or {}
        self.r_pos = mn.get('position', 9.0)         # m^2 (GNSS ~3m accuracy)

    def predict(self, dt, heading_rad, ai_speed):
        """
        EKF prediction step.

        Uses external heading + AI-predicted forward speed.

        Args:
            dt: time step (seconds)
            heading_rad: current heading from orientation sensor (radians, math convention)
            ai_speed: AI-predicted forward speed (m/s, non-negative)
        """
        if dt <= 0 or dt > 1.0:
            return

        # Decompose forward speed into ENU velocity
        vx_pred = ai_speed * np.cos(heading_rad)
        vy_pred = ai_speed * np.sin(heading_rad)

        # State transition: position += velocity * dt, velocity = from AI
        self.x[0] += vx_pred * dt
        self.x[1] += vy_pred * dt
        self.x[2] = vx_pred
        self.x[3] = vy_pred

        # State transition Jacobian
        F = np.eye(self.n_states)
        F[0, 2] = dt
        F[1, 3] = dt

        # Process noise — higher for velocity (AI prediction uncertainty)
        Q = np.diag([
            self.q_pos * dt,
            self.q_pos * dt,
            self.q_vel * dt,
            self.q_vel * dt,
        ])

        # Propagate covariance
        self.P = F @ self.P @ F.T + Q

    def update_gnss_position(self, gnss_east, gnss_north):
        """
        EKF measurement update with GNSS position.

        Measurement: z = [east, north]
        """
        z = np.array([gnss_east, gnss_north])

        # Measurement matrix: observe px, py directly
        H = np.zeros((2, self.n_states))
        H[0, 0] = 1.0
        H[1, 1] = 1.0

        # Measurement noise
        R = np.diag([self.r_pos, self.r_pos])

        # Innovation
        y = z - H @ self.x

        # Innovation covariance
        S = H @ self.P @ H.T + R

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y

        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(self.n_states) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

    def get_position(self):
        """Return current estimated position (east, north)."""
        return self.x[0], self.x[1]

    def get_velocity(self):
        """Return current estimated velocity (vx, vy)."""
        return self.x[2], self.x[3]

    def get_state(self):
        """Return full state vector copy."""
        return self.x.copy()


