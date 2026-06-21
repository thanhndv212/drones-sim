"""Action parameterizations for the quadcopter RL environment.

Each parameterization defines ``low``, ``high``, ``dim`` and implements
``to_motors(action) -> (4,)`` — converting a policy action vector into
four motor speed commands (rad/s).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class MotorSpeedAction:
    """Lowest level: raw motor speeds for each of 4 rotors.

    Hardest to learn, suitable for acrobatic maneuvers.
    """

    dim = 4
    low = np.zeros(4, dtype=np.float32)
    high = np.full(4, 4000.0, dtype=np.float32)

    def to_motors(self, quad, action: NDArray) -> NDArray:
        return np.clip(action, self.low, self.high)


class ThrustBodyRatesAction:
    """Mid-level: collective thrust + 3 body-rate commands.

    Action: [thrust (N), omega_x, omega_y, omega_z (rad/s)].

    The thrust is allocated equally across 4 motors (hover thrust added),
    and the body-rate error is mapped to differential motor commands via
    the quadcopter allocation matrix inverted as a proportional controller.
    """

    dim = 4
    _thrust_low = 0.2 * 9.81
    _thrust_high = 2.0 * 9.81
    _rate_low = -5.0
    _rate_high = 5.0

    @property
    def low(self) -> np.ndarray:
        return np.array([self._thrust_low, self._rate_low, self._rate_low, self._rate_low], dtype=np.float32)

    @property
    def high(self) -> np.ndarray:
        return np.array([self._thrust_high, self._rate_high, self._rate_high, self._rate_high], dtype=np.float32)

    def to_motors(self, quad, action: NDArray) -> NDArray:
        thrust, wx, wy, wz = action
        # Hover feedforward
        hover_thrust = quad.mass * quad.g
        total_thrust = thrust + hover_thrust

        # Rate errors → torque commands (proportional rate controller)
        omega = quad.get_angular_velocity()
        tau = np.array([wx - omega[0], wy - omega[1], wz - omega[2]])

        # Build wrench and invert allocation
        wrench = np.array([
            np.clip(total_thrust, 0.2 * hover_thrust, 2.0 * hover_thrust),
            tau[0] * 0.05,
            tau[1] * 0.05,
            tau[2] * 0.02,
        ])

        alloc = quad.allocation_matrix()
        try:
            w_sq = np.linalg.solve(alloc, wrench)
        except np.linalg.LinAlgError:
            hover_w = np.sqrt(hover_thrust / (4 * quad.k_f))
            return np.full(4, hover_w)

        motor_speeds = np.sqrt(np.maximum(w_sq, 0.0))
        return np.clip(motor_speeds, 0.0, 4000.0)
