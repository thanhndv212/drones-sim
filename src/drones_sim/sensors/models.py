"""Sensor noise and temperature models for IMU simulation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class SensorNoiseModel:
    """White noise + bias model for a 3-axis sensor.

    Bias model
    ----------
    When ``bias_time_constant`` is finite and ``bias_random_walk_std > 0``,
    the bias evolves each call to ``apply(dt=...)`` as a discrete
    first-order Gauss-Markov process:

        b_{k+1} = exp(-dt/tau) * b_k + sigma_b * sqrt(1 - exp(-2dt/tau)) * eta

    where ``tau = bias_time_constant`` and ``sigma_b = bias_random_walk_std``.

    The default ``bias_time_constant=inf`` reproduces the original constant-bias
    behaviour (backward compatible).  Set ``bias_random_walk_std=0`` to disable
    the stochastic drive while keeping the exponential decay.
    """
    noise_std: float
    bias_range: float
    scale_factor_range: tuple[float, float] = (1.0, 1.0)
    bias_time_constant: float = float('inf')   # tau_b  [s]; inf = constant bias
    bias_random_walk_std: float = 0.0          # sigma_b [units/s^0.5]

    # Populated on init
    bias: NDArray = field(init=False)
    scale_factor: NDArray = field(init=False)

    def __post_init__(self):
        self.bias = np.random.uniform(-self.bias_range, self.bias_range, 3)
        lo, hi = self.scale_factor_range
        self.scale_factor = np.random.uniform(lo, hi, 3)

    def apply(
        self,
        true_value: NDArray,
        temp_factor: float = 1.0,
        dt: float | None = None,
    ) -> NDArray:
        """Apply scale factor, bias, and additive noise.

        When ``dt`` is provided and bias dynamics are configured the bias
        state is updated via the Gauss-Markov recursion before measurement
        corruption.
        """
        # Bias update (Gauss-Markov random walk)
        if (
            dt is not None
            and not np.isinf(self.bias_time_constant)
            and self.bias_time_constant > 0.0
        ):
            decay = np.exp(-dt / self.bias_time_constant)
            if self.bias_random_walk_std > 0.0:
                noise_amp = self.bias_random_walk_std * np.sqrt(1.0 - decay ** 2)
                drive = np.random.normal(0.0, noise_amp, 3)
            else:
                drive = np.zeros(3)
            self.bias = decay * self.bias + drive

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
