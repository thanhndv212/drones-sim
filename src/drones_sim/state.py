"""Typed data exchanged between the simulation, controllers, and estimators.

The original package passed partially documented NumPy arrays between modules.
These small value objects make frame conventions and units explicit while still
offering vector conversions for numerical algorithms and legacy callers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .math_utils import quat_normalize, quat_to_euler, quat_to_rotation_matrix


def _vector(value: NDArray, size: int, name: str) -> NDArray:
    array = np.asarray(value, dtype=float).copy()
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


@dataclass
class VehicleState:
    """Complete quadcopter state using ENU/world and FRU/body conventions.

    ``quaternion`` is Hamilton ``[w, x, y, z]`` and rotates body vectors into
    the world frame. ``body_rates`` are expressed in the body frame [rad/s].
    """

    position: NDArray = field(default_factory=lambda: np.zeros(3))
    velocity: NDArray = field(default_factory=lambda: np.zeros(3))
    quaternion: NDArray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0])
    )
    body_rates: NDArray = field(default_factory=lambda: np.zeros(3))
    motor_speeds: NDArray = field(default_factory=lambda: np.zeros(4))
    time: float = 0.0

    def __post_init__(self) -> None:
        self.position = _vector(self.position, 3, "position")
        self.velocity = _vector(self.velocity, 3, "velocity")
        self.quaternion = quat_normalize(_vector(self.quaternion, 4, "quaternion"))
        if np.linalg.norm(self.quaternion) < 1e-12:
            raise ValueError("quaternion must have non-zero norm")
        self.body_rates = _vector(self.body_rates, 3, "body_rates")
        self.motor_speeds = _vector(self.motor_speeds, 4, "motor_speeds")
        self.time = float(self.time)

    @classmethod
    def from_vector(
        cls,
        vector: NDArray,
        *,
        motor_speeds: NDArray | None = None,
        time: float = 0.0,
    ) -> VehicleState:
        """Construct from ``[position, velocity, quaternion, body_rates]``."""
        value = _vector(vector, 13, "state vector")
        return cls(
            position=value[0:3],
            velocity=value[3:6],
            quaternion=value[6:10],
            body_rates=value[10:13],
            motor_speeds=np.zeros(4) if motor_speeds is None else motor_speeds,
            time=time,
        )

    def as_vector(self) -> NDArray:
        return np.concatenate(
            [self.position, self.velocity, self.quaternion, self.body_rates]
        )

    @property
    def rotation_matrix(self) -> NDArray:
        """World-from-body rotation matrix."""
        return quat_to_rotation_matrix(self.quaternion)

    @property
    def euler(self) -> NDArray:
        """Roll, pitch, yaw view for display and compatibility only."""
        return quat_to_euler(self.quaternion)

    def copy(self) -> VehicleState:
        return VehicleState(
            self.position,
            self.velocity,
            self.quaternion,
            self.body_rates,
            self.motor_speeds,
            self.time,
        )


@dataclass
class TrajectorySetpoint:
    """Flat-output reference consumed by position controllers."""

    position: NDArray = field(default_factory=lambda: np.zeros(3))
    velocity: NDArray = field(default_factory=lambda: np.zeros(3))
    acceleration: NDArray = field(default_factory=lambda: np.zeros(3))
    yaw: float = 0.0
    yaw_rate: float = 0.0

    def __post_init__(self) -> None:
        self.position = _vector(self.position, 3, "setpoint.position")
        self.velocity = _vector(self.velocity, 3, "setpoint.velocity")
        self.acceleration = _vector(self.acceleration, 3, "setpoint.acceleration")
        self.yaw = float(self.yaw)
        self.yaw_rate = float(self.yaw_rate)


@dataclass
class ControlOutput:
    """Controller output and allocator diagnostics."""

    thrust: float
    torque: NDArray
    motor_speeds: NDArray
    achieved_wrench: NDArray
    saturated: bool = False

    def __post_init__(self) -> None:
        self.thrust = float(self.thrust)
        self.torque = _vector(self.torque, 3, "torque")
        self.motor_speeds = _vector(self.motor_speeds, 4, "motor_speeds")
        self.achieved_wrench = _vector(
            self.achieved_wrench, 4, "achieved_wrench"
        )
