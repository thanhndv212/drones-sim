"""Attitude and Heading Reference System (AHRS) using complementary filtering.

Fuses accelerometer, gyroscope, and magnetometer data for attitude estimation.
Consolidated from imu_ekf_fusion_enhanced.py.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..math_utils import quat_normalize, quat_multiply, quat_to_rotation_matrix


class AHRS:
    """Complementary-filter AHRS with gyro bias learning.

    Uses accelerometer and magnetometer error feedback to correct
    gyroscope-integrated orientation.
    """

    def __init__(
        self,
        dt: float,
        accel_weight: float = 0.02,
        mag_weight: float = 0.01,
        bias_learn_rate: float = 0.001,
        gravity: NDArray | None = None,
        mag_ref: NDArray | None = None,
    ):
        self.dt = dt
        self.accel_weight = accel_weight
        self.mag_weight = mag_weight
        self.bias_learn_rate = bias_learn_rate

        self.gravity = gravity if gravity is not None else np.array([0.0, 0.0, 9.81])
        self.mag_ref = mag_ref if mag_ref is not None else np.array([25.0, 5.0, -40.0])

        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        self.gyro_bias = np.zeros(3)

    def _normalize(self, v: NDArray) -> NDArray:
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def update(self, gyro: NDArray, accel: NDArray, mag: NDArray) -> tuple[NDArray, NDArray]:
        """Process one time-step of sensor data.

        Returns (quaternion, gyro_bias).
        """
        gyro_corrected = gyro - self.gyro_bias

        accel_norm = self._normalize(accel)
        mag_norm = self._normalize(mag)

        R = quat_to_rotation_matrix(self.quaternion)

        expected_gravity = -R.T @ self._normalize(self.gravity)
        expected_mag = R.T @ self._normalize(self.mag_ref)

        accel_error = np.cross(accel_norm, expected_gravity)
        mag_error = np.cross(mag_norm, expected_mag)

        error = accel_error * self.accel_weight + mag_error * self.mag_weight
        self.gyro_bias += error * self.bias_learn_rate
        gyro_corrected = gyro_corrected + error

        w, x, y, z = self.quaternion
        wx, wy, wz = gyro_corrected

        q_dot = 0.5 * np.array([
            -x * wx - y * wy - z * wz,
             w * wx + y * wz - z * wy,
             w * wy + z * wx - x * wz,
             w * wz + x * wy - y * wx,
        ])

        self.quaternion = quat_normalize(self.quaternion + q_dot * self.dt)
        return self.quaternion.copy(), self.gyro_bias.copy()
