"""Extended Kalman Filter for IMU sensor fusion.

Two modes of operation:
1. **Full-state EKF** (n=10): position + velocity + quaternion.
   Uses proper analytical Jacobians for accel/mag measurement updates.
   Consolidated from imu_ekf_simulation.py (the most mathematically rigorous version).

2. **AHRS-aided EKF** (n=9): position + velocity + accel_bias.
   Attitude handled by the AHRS complementary filter; EKF estimates translational
   states and accelerometer bias.  Supports adaptive noise from innovation monitoring.
   Consolidated from imu_ekf_fusion_enhanced.py.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from numpy.typing import NDArray

from ..math_utils import (
    quat_derivative,
    quat_angular_velocity_jacobian,
    quat_normalize,
    quat_to_rotation_matrix,
)
from .ahrs import AHRS


# ---------------------------------------------------------------------------
# Full-state EKF (10-state) — from imu_ekf_simulation.py
# ---------------------------------------------------------------------------

class ExtendedKalmanFilter:
    """10-state EKF: [pos(3), vel(3), quat(4)].

    Uses analytically-derived Jacobians for accelerometer and magnetometer
    measurement models.
    """

    def __init__(
        self,
        dt: float,
        initial_state: NDArray | None = None,
        gravity: NDArray | None = None,
        mag_ref: NDArray | None = None,
    ):
        self.n = 10
        self.dt = dt

        if initial_state is not None:
            self.x = initial_state.copy()
        else:
            self.x = np.zeros(self.n)
            self.x[6] = 1.0  # identity quaternion

        self.gravity = gravity if gravity is not None else np.array([0.0, 0.0, 9.81])
        self.mag_ref = mag_ref if mag_ref is not None else np.array([20.0, 0.0, -40.0])

        # Covariance
        self.P = np.eye(self.n) * 0.01
        self.P[0:3, 0:3] *= 0.01
        self.P[3:6, 3:6] *= 0.1
        self.P[6:10, 6:10] *= 0.001

        # Process noise
        self.Q = np.eye(self.n) * 0.001
        self.Q[0:3, 0:3] *= 0.001
        self.Q[3:6, 3:6] *= 0.01
        self.Q[6:10, 6:10] *= 0.001

        # Measurement noise
        self.R_accel = np.eye(3) * 0.05
        self.R_mag = np.eye(3) * 0.5

        # Bias estimates (slowly updated)
        self.accel_bias = np.zeros(3)
        self.gyro_bias = np.zeros(3)
        self.mag_bias = np.zeros(3)

    # -- prediction --------------------------------------------------------

    def predict(self, gyro: NDArray) -> None:
        pos, vel, quat = self.x[:3], self.x[3:6], self.x[6:10]
        gyro_c = gyro - self.gyro_bias

        q_dot = quat_derivative(quat, gyro_c)
        new_quat = quat_normalize(quat + q_dot * self.dt)

        self.x[:3] = pos + vel * self.dt
        self.x[6:10] = new_quat

        F = np.eye(self.n)
        F[0:3, 3:6] = np.eye(3) * self.dt
        F[6:10, 6:10] = np.eye(4) + quat_angular_velocity_jacobian(gyro_c) * self.dt

        self.P = F @ self.P @ F.T + self.Q

    # -- accelerometer correction ------------------------------------------

    def correct_accel(self, accel: NDArray) -> None:
        quat = self.x[6:10]
        R = quat_to_rotation_matrix(quat)
        expected = -R.T @ self.gravity
        residual = (accel - self.accel_bias) - expected

        H = self._accel_jacobian(quat)
        S = H @ self.P @ H.T + self.R_accel
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x += K @ residual
        self.x[6:10] = quat_normalize(self.x[6:10])
        self.P = (np.eye(self.n) - K @ H) @ self.P

        # velocity update from accelerometer
        accel_world = R @ (accel - self.accel_bias) + self.gravity
        alpha = 0.1
        self.x[3:6] = (1 - alpha) * self.x[3:6] + alpha * (self.x[3:6] + accel_world * self.dt)

    # -- magnetometer correction -------------------------------------------

    def correct_mag(self, mag: NDArray) -> None:
        quat = self.x[6:10]
        R = quat_to_rotation_matrix(quat)
        expected = R.T @ self.mag_ref
        residual = (mag - self.mag_bias) - expected

        H = self._mag_jacobian(quat)
        S = H @ self.P @ H.T + self.R_mag
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x += K @ residual
        self.x[6:10] = quat_normalize(self.x[6:10])
        self.P = (np.eye(self.n) - K @ H) @ self.P

    # -- state access ------------------------------------------------------

    def get_state(self) -> dict:
        return {
            "position": self.x[:3].copy(),
            "velocity": self.x[3:6].copy(),
            "quaternion": self.x[6:10].copy(),
        }

    # -- Jacobians (analytical) --------------------------------------------

    def _accel_jacobian(self, q: NDArray) -> NDArray:
        """d(R^T g)/d(quat) — analytical Jacobian of accel measurement model."""
        w, x, y, z = q
        gx, gy, gz = self.gravity
        H = np.zeros((3, self.n))

        # Partial derivatives of R^T @ gravity w.r.t. each quaternion component
        # Negated because accelerometer measures reaction force
        H[0, 6] = -2 * (2 * w * gz - 2 * gy * y + 2 * gx * z)
        H[0, 7] = -2 * (2 * gz * w - 2 * gy * z)
        H[0, 8] = -2 * (-2 * gx * z - 2 * gz * x)
        H[0, 9] = -2 * (2 * gx * y - 2 * gy * x)

        H[1, 6] = -2 * (2 * gz * y - 2 * gx * z)
        H[1, 7] = -2 * (2 * gy * w + 2 * gz * z)
        H[1, 8] = -2 * (2 * w * gz - 2 * gx * y)
        H[1, 9] = -2 * (2 * gz * x + 2 * gy * w)

        H[2, 6] = -2 * (-2 * gy * w - 2 * gx * x)
        H[2, 7] = -2 * (-2 * gx * w - 2 * gy * x)
        H[2, 8] = -2 * (-2 * gy * y - 2 * gz * z)
        H[2, 9] = -2 * (2 * gx * w - 2 * gy * z)

        return H

    def _mag_jacobian(self, q: NDArray) -> NDArray:
        """d(R^T m)/d(quat) — analytical Jacobian of magnetometer measurement model."""
        w, x, y, z = q
        mx, my, mz = self.mag_ref
        H = np.zeros((3, self.n))

        H[0, 6] = 2 * (2 * w * mz - 2 * my * y + 2 * mx * z)
        H[0, 7] = 2 * (2 * mz * w - 2 * my * z)
        H[0, 8] = 2 * (-2 * mx * z - 2 * mz * x)
        H[0, 9] = 2 * (2 * mx * y - 2 * my * x)

        H[1, 6] = 2 * (2 * mz * y - 2 * mx * z)
        H[1, 7] = 2 * (2 * my * w + 2 * mz * z)
        H[1, 8] = 2 * (2 * w * mz - 2 * mx * y)
        H[1, 9] = 2 * (2 * mz * x + 2 * my * w)

        H[2, 6] = 2 * (-2 * my * w - 2 * mx * x)
        H[2, 7] = 2 * (-2 * mx * w - 2 * my * x)
        H[2, 8] = 2 * (-2 * my * y - 2 * mz * z)
        H[2, 9] = 2 * (2 * mx * w - 2 * my * z)

        return H


# ---------------------------------------------------------------------------
# AHRS-aided adaptive EKF (9-state) — from imu_ekf_fusion_enhanced.py
# ---------------------------------------------------------------------------

class AdaptiveEKF:
    """9-state EKF: [pos(3), vel(3), accel_bias(3)].

    Attitude is estimated separately by an AHRS complementary filter.
    Process and measurement noise are adapted based on innovation statistics.
    """

    def __init__(
        self,
        dt: float,
        init_pos: NDArray | None = None,
        init_vel: NDArray | None = None,
        innovation_window: int = 20,
        gravity: NDArray | None = None,
        mag_ref: NDArray | None = None,
    ):
        self.n = 9
        self.dt = dt

        self.x = np.zeros(self.n)
        if init_pos is not None:
            self.x[:3] = init_pos
        if init_vel is not None:
            self.x[3:6] = init_vel

        self.gravity = gravity if gravity is not None else np.array([0.0, 0.0, 9.81])

        self.P = np.eye(self.n)
        self.P[:3, :3] *= 0.01
        self.P[3:6, 3:6] *= 0.1
        self.P[6:9, 6:9] *= 0.01

        self.base_Q = np.eye(self.n)
        self.base_Q[:3, :3] *= 0.01
        self.base_Q[3:6, 3:6] *= 0.1
        self.base_Q[6:9, 6:9] *= 0.001
        self.Q = self.base_Q.copy()

        self.base_R_accel = np.eye(3) * 0.1
        self.R_accel = self.base_R_accel.copy()

        self._innovations: deque[NDArray] = deque(maxlen=innovation_window)

        self.ahrs = AHRS(
            dt,
            accel_weight=0.02,
            mag_weight=0.01,
            gravity=self.gravity,
            mag_ref=mag_ref,
        )
        self.orientation = np.array([1.0, 0.0, 0.0, 0.0])
        self.gyro_bias = np.zeros(3)

    # -- prediction --------------------------------------------------------

    def predict(self, gyro: NDArray, accel: NDArray, mag: NDArray, adaptive_factor: float = 1.0) -> None:
        self.orientation, self.gyro_bias = self.ahrs.update(gyro, accel, mag)
        R = quat_to_rotation_matrix(self.orientation)

        accel_bias = self.x[6:9]
        accel_world = R @ (accel - accel_bias) + self.gravity

        pos, vel = self.x[:3], self.x[3:6]
        self.x[:3] = pos + vel * self.dt + 0.5 * accel_world * self.dt**2
        self.x[3:6] = vel + accel_world * self.dt

        F = np.eye(self.n)
        F[:3, 3:6] = np.eye(3) * self.dt
        F[3:6, 6:9] = -R * self.dt

        self.Q = self.base_Q * adaptive_factor
        self.P = F @ self.P @ F.T + self.Q

    # -- measurement update ------------------------------------------------

    def correct(self, accel: NDArray, adaptive_factor: float = 1.0) -> None:
        R = quat_to_rotation_matrix(self.orientation)
        accel_bias = self.x[6:9]
        expected = R.T @ (-self.gravity) + accel_bias
        innovation = accel - expected

        self._innovations.append(innovation)

        # Adaptive measurement noise
        if len(self._innovations) >= self._innovations.maxlen:
            cov = np.zeros((3, 3))
            for inn in self._innovations:
                cov += np.outer(inn, inn)
            cov /= len(self._innovations)
            self.R_accel = (self.base_R_accel + np.diag(np.diag(cov))) * adaptive_factor
        else:
            self.R_accel = self.base_R_accel * adaptive_factor

        H = np.zeros((3, self.n))
        H[:, 6:9] = np.eye(3)

        S = H @ self.P @ H.T + self.R_accel
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x += K @ innovation

        # Joseph form for numerical stability
        I = np.eye(self.n)
        IKH = I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ self.R_accel @ K.T

    # -- state access ------------------------------------------------------

    def get_state(self) -> dict:
        return {
            "position": self.x[:3].copy(),
            "velocity": self.x[3:6].copy(),
            "accel_bias": self.x[6:9].copy(),
            "quaternion": self.orientation.copy(),
            "gyro_bias": self.gyro_bias.copy(),
        }
