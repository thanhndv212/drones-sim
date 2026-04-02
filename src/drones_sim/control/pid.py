"""Generic PID controller with anti-windup and output saturation.

Consolidated from quadcopter_simulation.py.
"""

from __future__ import annotations

import numpy as np


class PIDController:
    """Scalar PID with integral anti-windup and output clamping."""

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        output_limits: tuple[float, float] | None = None,
        windup_limits: tuple[float, float] | None = None,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        self.windup_limits = windup_limits

        self._prev_error = 0.0
        self._integral = 0.0

    def reset(self) -> None:
        self._prev_error = 0.0
        self._integral = 0.0

    def update(self, setpoint: float, measurement: float, dt: float) -> float:
        error = setpoint - measurement
        dt = max(dt, 1e-6)

        self._integral += error * dt
        if self.windup_limits is not None:
            self._integral = np.clip(self._integral, *self.windup_limits)

        derivative = (error - self._prev_error) / dt
        self._prev_error = error

        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        if self.output_limits is not None:
            output = np.clip(output, *self.output_limits)

        return float(output)
