"""IMU sensor simulator with realistic noise, bias, scale factor, and optional temperature effects.

Consolidated from imu_ekf_simulation.py, imu_ekf_fusion_enhanced.py,
imu_ekf_fusion_final.py, and imu_ekf_fusion_simplified.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

from ..trajectory import TrajectoryData
from .models import SensorNoiseModel, TemperatureModel


@dataclass
class IMUConfig:
    """Configuration for IMU sensor characteristics."""
    accel_noise_std: float = 0.05       # m/s^2
    accel_bias_range: float = 0.1       # m/s^2
    accel_scale: tuple[float, float] = (0.98, 1.02)

    gyro_noise_std: float = 0.01        # rad/s
    gyro_bias_range: float = 0.005      # rad/s
    gyro_scale: tuple[float, float] = (0.99, 1.01)

    mag_noise_std: float = 0.5          # uT
    mag_bias_range: float = 1.0         # uT
    mag_scale: tuple[float, float] = (0.97, 1.03)

    # Earth references
    gravity: NDArray = field(default_factory=lambda: np.array([0.0, 0.0, 9.81]))
    mag_field_ref: NDArray = field(default_factory=lambda: np.array([25.0, 5.0, -40.0]))

    enable_temperature: bool = False


@dataclass
class IMUData:
    """Container for simulated IMU readings."""
    t: NDArray
    accel: NDArray          # (N, 3)
    gyro: NDArray           # (N, 3)
    mag: NDArray            # (N, 3)
    temperature: Optional[NDArray] = None  # (N,) if temperature model enabled

    # Ground-truth biases for validation
    accel_bias: NDArray = field(default_factory=lambda: np.zeros(3))
    gyro_bias: NDArray = field(default_factory=lambda: np.zeros(3))
    mag_bias: NDArray = field(default_factory=lambda: np.zeros(3))


class IMUSimulator:
    """Simulate 9-axis IMU readings from a known trajectory.

    Supports:
    - Additive white Gaussian noise
    - Constant bias per axis
    - Scale-factor errors
    - Optional temperature-dependent drift
    """

    def __init__(self, config: IMUConfig | None = None, seed: int | None = 42):
        if seed is not None:
            np.random.seed(seed)

        self.cfg = config or IMUConfig()

        self.accel_model = SensorNoiseModel(
            noise_std=self.cfg.accel_noise_std,
            bias_range=self.cfg.accel_bias_range,
            scale_factor_range=self.cfg.accel_scale,
        )
        self.gyro_model = SensorNoiseModel(
            noise_std=self.cfg.gyro_noise_std,
            bias_range=self.cfg.gyro_bias_range,
            scale_factor_range=self.cfg.gyro_scale,
        )
        self.mag_model = SensorNoiseModel(
            noise_std=self.cfg.mag_noise_std,
            bias_range=self.cfg.mag_bias_range,
            scale_factor_range=self.cfg.mag_scale,
        )
        self.temp_model = TemperatureModel() if self.cfg.enable_temperature else None

    def simulate(self, traj: TrajectoryData) -> IMUData:
        """Generate sensor readings from a trajectory."""
        n = len(traj.t)
        dt = traj.t[1] - traj.t[0] if n > 1 else 0.01
        duration = traj.t[-1] - traj.t[0]

        accel_out = np.zeros((n, 3))
        gyro_out = np.zeros((n, 3))
        mag_out = np.zeros((n, 3))
        temperature = np.zeros(n) if self.temp_model else None

        for i in range(n):
            # Rotation matrix from world to body
            q_xyzw = np.roll(traj.orientation_quat[i], -1)  # wxyz -> xyzw
            rot = R.from_quat(q_xyzw).as_matrix()

            # --- Accelerometer: gravity in body + linear accel reaction ---
            gravity_body = rot.T @ self.cfg.gravity
            lin_accel_body = rot.T @ (-traj.acceleration[i])
            true_accel = gravity_body + lin_accel_body

            # --- Gyroscope: angular velocity in body frame ---
            true_gyro = traj.angular_velocity[i]

            # --- Magnetometer: Earth field in body frame ---
            true_mag = rot.T @ self.cfg.mag_field_ref

            # Apply temperature effects if enabled
            temp_factor = 1.0
            if self.temp_model is not None:
                temp = self.temp_model.temperature_at(traj.t[i], duration)
                temperature[i] = temp
                temp_factor = self.temp_model.noise_scale(temp)
                true_accel = true_accel + self.temp_model.accel_offset(temp)
                true_gyro = true_gyro + self.temp_model.gyro_offset(temp)
                true_mag = true_mag + self.temp_model.mag_offset(temp)

            accel_out[i] = self.accel_model.apply(true_accel, temp_factor)
            gyro_out[i] = self.gyro_model.apply(true_gyro, temp_factor)
            mag_out[i] = self.mag_model.apply(true_mag, temp_factor)

        return IMUData(
            t=traj.t,
            accel=accel_out,
            gyro=gyro_out,
            mag=mag_out,
            temperature=temperature,
            accel_bias=self.accel_model.bias.copy(),
            gyro_bias=self.gyro_model.bias.copy(),
            mag_bias=self.mag_model.bias.copy(),
        )
