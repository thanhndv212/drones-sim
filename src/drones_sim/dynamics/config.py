"""Physical configuration for the quadcopter plant."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class QuadcopterConfig:
    """Validated physical and actuator parameters.

    The default rotor positions form a ``+`` layout compatible with the
    historical model: front, right, rear, left. Positive rotor directions
    produce positive yaw moments.
    """

    mass: float = 1.0
    inertia: NDArray = field(
        default_factory=lambda: np.diag([0.01, 0.01, 0.018])
    )
    arm_length: float = 0.2
    thrust_coefficient: float = 1.0e-6
    moment_coefficient: float = 1.0e-7
    linear_drag: NDArray = field(default_factory=lambda: np.full(3, 0.1))
    quadratic_drag: NDArray = field(default_factory=lambda: np.zeros(3))
    gravity: float = 9.81
    motor_time_constant: float = 0.04
    min_motor_speed: float = 0.0
    max_motor_speed: float = 4000.0
    rotor_directions: NDArray = field(
        default_factory=lambda: np.array([1.0, -1.0, 1.0, -1.0])
    )

    def __post_init__(self) -> None:
        self.mass = float(self.mass)
        self.arm_length = float(self.arm_length)
        self.thrust_coefficient = float(self.thrust_coefficient)
        self.moment_coefficient = float(self.moment_coefficient)
        self.gravity = float(self.gravity)
        self.motor_time_constant = float(self.motor_time_constant)
        self.min_motor_speed = float(self.min_motor_speed)
        self.max_motor_speed = float(self.max_motor_speed)
        self.inertia = np.asarray(self.inertia, dtype=float).copy()
        if self.inertia.shape == (3,):
            self.inertia = np.diag(self.inertia)
        self.linear_drag = self._axis_value(self.linear_drag, "linear_drag")
        self.quadratic_drag = self._axis_value(
            self.quadratic_drag, "quadratic_drag"
        )
        self.rotor_directions = np.asarray(
            self.rotor_directions, dtype=float
        ).copy()
        self.validate()

    @staticmethod
    def _axis_value(value: NDArray | float, name: str) -> NDArray:
        array = np.asarray(value, dtype=float)
        if array.ndim == 0:
            array = np.full(3, float(array))
        if array.shape != (3,):
            raise ValueError(f"{name} must be scalar or shape (3,), got {array.shape}")
        return array.copy()

    def validate(self) -> None:
        if self.mass <= 0.0:
            raise ValueError("mass must be positive")
        if self.arm_length <= 0.0:
            raise ValueError("arm_length must be positive")
        if self.thrust_coefficient <= 0.0 or self.moment_coefficient <= 0.0:
            raise ValueError("rotor coefficients must be positive")
        if self.inertia.shape != (3, 3):
            raise ValueError("inertia must have shape (3, 3)")
        if not np.allclose(self.inertia, self.inertia.T):
            raise ValueError("inertia must be symmetric")
        if np.min(np.linalg.eigvalsh(self.inertia)) <= 0.0:
            raise ValueError("inertia must be positive definite")
        if np.any(self.linear_drag < 0.0) or np.any(self.quadratic_drag < 0.0):
            raise ValueError("drag coefficients must be non-negative")
        if self.gravity <= 0.0:
            raise ValueError("gravity must be positive")
        if self.motor_time_constant < 0.0:
            raise ValueError("motor_time_constant cannot be negative")
        if not 0.0 <= self.min_motor_speed < self.max_motor_speed:
            raise ValueError("motor speed limits are invalid")
        if self.rotor_directions.shape != (4,):
            raise ValueError("rotor_directions must have shape (4,)")

    @property
    def rotor_positions(self) -> NDArray:
        length = self.arm_length
        return np.array(
            [[length, 0.0, 0.0], [0.0, length, 0.0],
             [-length, 0.0, 0.0], [0.0, -length, 0.0]]
        )

    def allocation_matrix(self, efficiencies: NDArray | None = None) -> NDArray:
        """Map squared rotor speeds to ``[thrust, tau_x, tau_y, tau_z]``."""
        efficiency = (
            np.ones(4)
            if efficiencies is None
            else np.asarray(efficiencies, dtype=float)
        )
        if efficiency.shape != (4,):
            raise ValueError("efficiencies must have shape (4,)")
        force_gain = self.thrust_coefficient * efficiency
        positions = self.rotor_positions
        return np.vstack(
            [
                force_gain,
                positions[:, 1] * force_gain,
                -positions[:, 0] * force_gain,
                self.rotor_directions * self.moment_coefficient * efficiency,
            ]
        )
