"""Coordinate-free SE(3) position and attitude controller."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ..state import ControlOutput, TrajectorySetpoint, VehicleState
from .allocation import ControlAllocator


@dataclass
class GeometricControllerConfig:
    position_gain: NDArray = field(
        default_factory=lambda: np.array([3.0, 3.0, 6.0])
    )
    velocity_gain: NDArray = field(
        default_factory=lambda: np.array([3.0, 3.0, 4.0])
    )
    integral_gain: NDArray = field(
        default_factory=lambda: np.array([0.02, 0.02, 0.10])
    )
    attitude_gain: NDArray = field(
        default_factory=lambda: np.array([0.22, 0.22, 0.08])
    )
    rate_gain: NDArray = field(
        default_factory=lambda: np.array([0.06, 0.06, 0.025])
    )
    integral_limit: NDArray = field(
        default_factory=lambda: np.array([2.0, 2.0, 1.0])
    )
    max_tilt: float = np.deg2rad(45.0)
    min_thrust_ratio: float = 0.05
    max_thrust_ratio: float = 2.5


def _vee(skew: NDArray) -> NDArray:
    return np.array([skew[2, 1], skew[0, 2], skew[1, 0]])


class GeometricController:
    """Nonlinear trajectory controller on SE(3).

    Position feedback produces a desired world-frame force. Its direction and
    the requested yaw define the full desired attitude, avoiding Euler-angle
    singularities and the small-angle approximation used by the legacy PID.
    """

    def __init__(self, quad, config: GeometricControllerConfig | None = None):
        self.quad = quad
        self.config = config or GeometricControllerConfig()
        self._integral_error = np.zeros(3)
        self._previous_target: NDArray | None = None

    def reset(self) -> None:
        self._integral_error.fill(0.0)
        self._previous_target = None

    def compute_output(
        self,
        setpoint: TrajectorySetpoint,
        dt: float,
        state: VehicleState | None = None,
    ) -> ControlOutput:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        current = self.quad.get_state() if state is None else state
        cfg = self.config
        mass = self.quad.mass
        gravity = self.quad.g

        position_error = setpoint.position - current.position
        velocity_error = setpoint.velocity - current.velocity
        self._integral_error += position_error * dt
        self._integral_error = np.clip(
            self._integral_error, -cfg.integral_limit, cfg.integral_limit
        )

        desired_accel = (
            setpoint.acceleration
            + cfg.position_gain * position_error
            + cfg.velocity_gain * velocity_error
            + cfg.integral_gain * self._integral_error
        )
        desired_force = mass * (desired_accel + np.array([0.0, 0.0, gravity]))

        # Enforce a tilt cone by limiting the horizontal/vertical force ratio.
        vertical = max(float(desired_force[2]), 1e-6)
        horizontal = desired_force[:2]
        horizontal_norm = float(np.linalg.norm(horizontal))
        max_horizontal = vertical * np.tan(cfg.max_tilt)
        if horizontal_norm > max_horizontal:
            desired_force[:2] *= max_horizontal / horizontal_norm

        force_norm = float(np.linalg.norm(desired_force))
        if force_norm < 1e-9:
            desired_b3 = np.array([0.0, 0.0, 1.0])
        else:
            desired_b3 = desired_force / force_norm
        heading = np.array([np.cos(setpoint.yaw), np.sin(setpoint.yaw), 0.0])
        desired_b2 = np.cross(desired_b3, heading)
        if np.linalg.norm(desired_b2) < 1e-8:
            desired_b2 = np.array([-np.sin(setpoint.yaw), np.cos(setpoint.yaw), 0.0])
        desired_b2 /= np.linalg.norm(desired_b2)
        desired_b1 = np.cross(desired_b2, desired_b3)
        desired_rotation = np.column_stack([desired_b1, desired_b2, desired_b3])

        rotation = current.rotation_matrix
        attitude_error = 0.5 * _vee(
            desired_rotation.T @ rotation - rotation.T @ desired_rotation
        )
        desired_rates = desired_rotation.T @ np.array(
            [0.0, 0.0, setpoint.yaw_rate]
        )
        rate_error = current.body_rates - rotation.T @ desired_rotation @ desired_rates

        thrust = float(np.dot(desired_force, rotation[:, 2]))
        hover = mass * gravity
        thrust = float(
            np.clip(
                thrust,
                cfg.min_thrust_ratio * hover,
                cfg.max_thrust_ratio * hover,
            )
        )
        inertia = self.quad.I
        torque = (
            -cfg.attitude_gain * attitude_error
            - cfg.rate_gain * rate_error
            + np.cross(current.body_rates, inertia @ current.body_rates)
        )
        requested_wrench = np.concatenate([[thrust], torque])
        allocator = ControlAllocator(
            self.quad.allocation_matrix(),
            min_speed=self.quad.min_motor_speed,
            max_speed=self.quad.max_motor_speed,
            weights=np.array([1.0, 5.0, 5.0, 2.0]),
        )
        allocation = allocator.allocate(requested_wrench)
        if allocation.saturated:
            # Back-calculation prevents integral accumulation when thrust is limited.
            self._integral_error *= 0.995
        return ControlOutput(
            thrust=thrust,
            torque=torque,
            motor_speeds=allocation.motor_speeds,
            achieved_wrench=allocation.achieved_wrench,
            saturated=allocation.saturated,
        )

    def compute(
        self,
        target_pos: NDArray,
        target_yaw: float,
        dt: float,
        prev_target_pos: NDArray | None = None,
    ) -> NDArray:
        """Compatibility interface shared with the legacy controllers."""
        velocity = np.zeros(3)
        if prev_target_pos is not None:
            velocity = (np.asarray(target_pos) - prev_target_pos) / dt
        output = self.compute_output(
            TrajectorySetpoint(
                position=np.asarray(target_pos), velocity=velocity, yaw=target_yaw
            ),
            dt,
        )
        return output.motor_speeds
