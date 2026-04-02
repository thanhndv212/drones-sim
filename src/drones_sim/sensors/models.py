"""Sensor noise and temperature models for IMU simulation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class SensorNoiseModel:
    """White noise + constant bias model for a 3-axis sensor."""
    noise_std: float
    bias_range: float
    scale_factor_range: tuple[float, float] = (1.0, 1.0)

    # Populated on init
    bias: NDArray = field(init=False)
    scale_factor: NDArray = field(init=False)

    def __post_init__(self):
        self.bias = np.random.uniform(-self.bias_range, self.bias_range, 3)
        lo, hi = self.scale_factor_range
        self.scale_factor = np.random.uniform(lo, hi, 3)

    def apply(self, true_value: NDArray, temp_factor: float = 1.0) -> NDArray:
        """Apply scale factor, bias, and additive noise."""
        noisy = true_value * self.scale_factor + self.bias
        noisy += np.random.normal(0, self.noise_std * temp_factor, 3)
        return noisy


@dataclass
class TemperatureModel:
    """Sinusoidal temperature profile with per-axis sensitivity coefficients."""
    base_temp: float = 25.0
    amplitude: float = 10.0

    # Per-axis temperature coefficients (populated randomly)
    accel_coef: NDArray = field(init=False)
    gyro_coef: NDArray = field(init=False)
    mag_coef: NDArray = field(init=False)

    def __post_init__(self):
        self.accel_coef = np.random.uniform(-0.002, 0.002, 3)
        self.gyro_coef = np.random.uniform(-0.0005, 0.0005, 3)
        self.mag_coef = np.random.uniform(-0.05, 0.05, 3)

    def temperature_at(self, t: float, duration: float) -> float:
        return self.base_temp + self.amplitude * np.sin(2 * np.pi * t / duration)

    def noise_scale(self, temp: float) -> float:
        """Noise increases slightly with temperature deviation."""
        return 1.0 + 0.01 * (temp - self.base_temp) / 10.0

    def accel_offset(self, temp: float) -> NDArray:
        return self.accel_coef * (temp - self.base_temp)

    def gyro_offset(self, temp: float) -> NDArray:
        return self.gyro_coef * (temp - self.base_temp)

    def mag_offset(self, temp: float) -> NDArray:
        return self.mag_coef * (temp - self.base_temp)
