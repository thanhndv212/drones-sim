"""Cascaded PID controller for quadcopter position→velocity→attitude control.

Consolidated from quadcopter_simulation.py.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..dynamics.quadcopter import QuadcopterDynamics
from .pid import PIDController


class QuadcopterController:
    """Three-loop cascaded controller with physically consistent outputs."""

    def __init__(self, quad: QuadcopterDynamics):
        self.quad = quad
        self.convergence_threshold = 0.05
        self.max_lateral_speed = 1.5
        self.max_vertical_speed = 1.0
        self.max_lateral_accel = 3.0
        self.max_vertical_accel = 4.0
        self.max_tilt = np.deg2rad(17.0)
        self.max_torque = np.array([0.25, 0.25, 0.08])
        self.rate_damping = np.array([0.03, 0.03, 0.015])
        self.min_thrust = 0.2 * self.quad.mass * self.quad.g
        self.max_thrust = 2.0 * self.quad.mass * self.quad.g
        self.max_motor_speed = 4000.0

        # -- Position (outer) --
        self.x_ctrl = PIDController(1.2, 0.1, 0.35, (-1.5, 1.5), (-1.0, 1.0))
        self.y_ctrl = PIDController(1.2, 0.1, 0.35, (-1.5, 1.5), (-1.0, 1.0))
        self.z_ctrl = PIDController(1.6, 0.25, 0.45, (-1.0, 1.0), (-0.8, 0.8))

        # -- Velocity (middle) --
        self.vx_ctrl = PIDController(2.2, 0.3, 0.2, (-3.0, 3.0), (-1.0, 1.0))
        self.vy_ctrl = PIDController(2.2, 0.3, 0.2, (-3.0, 3.0), (-1.0, 1.0))
        self.vz_ctrl = PIDController(5.0, 1.5, 0.3, (-4.0, 4.0), (-1.0, 1.0))

        # -- Attitude (inner, torque outputs in N·m) --
        self.roll_ctrl = PIDController(
            0.12,
            0.02,
            0.0,
            (-0.25, 0.25),
            (-0.15, 0.15),
        )
        self.pitch_ctrl = PIDController(
            0.12,
            0.02,
            0.0,
            (-0.25, 0.25),
            (-0.15, 0.15),
        )
        self.yaw_ctrl = PIDController(
            0.05,
            0.01,
            0.0,
            (-0.08, 0.08),
            (-0.08, 0.08),
        )

        self._prev_target_vel = np.zeros(3)

    def reset(self) -> None:
        for ctrl in [
            self.x_ctrl, self.y_ctrl, self.z_ctrl,
            self.vx_ctrl, self.vy_ctrl, self.vz_ctrl,
            self.roll_ctrl, self.pitch_ctrl, self.yaw_ctrl,
        ]:
            ctrl.reset()
        self._prev_target_vel = np.zeros(3)

    def compute(
        self,
        target_pos: NDArray,
        target_yaw: float,
        dt: float,
        prev_target_pos: NDArray | None = None,
    ) -> NDArray:
        """Compute 4 motor speeds for position + yaw tracking.

        Returns motor speeds as a (4,) array.
        """
        pos = self.quad.get_position()
        vel = self.quad.get_velocity()
        att = self.quad.get_attitude()
        omega = self.quad.get_angular_velocity()

        target_vel_ff = np.zeros(3)
        if prev_target_pos is not None:
            target_vel_ff = (target_pos - prev_target_pos) / max(dt, 1e-6)
        target_vel_ff = np.clip(
            target_vel_ff,
            [
                -self.max_lateral_speed,
                -self.max_lateral_speed,
                -self.max_vertical_speed,
            ],
            [
                self.max_lateral_speed,
                self.max_lateral_speed,
                self.max_vertical_speed,
            ],
        )

        # Position -> desired velocity (m/s)
        target_vel = np.array([
            self.x_ctrl.update(target_pos[0], pos[0], dt),
            self.y_ctrl.update(target_pos[1], pos[1], dt),
            self.z_ctrl.update(target_pos[2], pos[2], dt),
        ])
        target_vel += np.array([0.25, 0.25, 0.15]) * target_vel_ff
        target_vel = np.clip(
            target_vel,
            [
                -self.max_lateral_speed,
                -self.max_lateral_speed,
                -self.max_vertical_speed,
            ],
            [
                self.max_lateral_speed,
                self.max_lateral_speed,
                self.max_vertical_speed,
            ],
        )

        # Velocity -> desired acceleration (m/s^2)
        desired_accel = np.array([
            self.vx_ctrl.update(target_vel[0], vel[0], dt),
            self.vy_ctrl.update(target_vel[1], vel[1], dt),
            self.vz_ctrl.update(target_vel[2], vel[2], dt),
        ])
        target_acc_ff = (target_vel - self._prev_target_vel) / max(dt, 1e-6)
        self._prev_target_vel = target_vel.copy()
        desired_accel += np.array([0.1, 0.1, 0.05]) * target_acc_ff
        desired_accel[:2] = np.clip(
            desired_accel[:2],
            -self.max_lateral_accel,
            self.max_lateral_accel,
        )
        desired_accel[2] = np.clip(
            desired_accel[2],
            -self.max_vertical_accel,
            self.max_vertical_accel,
        )

        # Small-angle mapping with package rotation convention:
        # positive pitch -> +x acceleration, positive roll -> -y acceleration.
        des_pitch = np.clip(
            desired_accel[0] / self.quad.g,
            -self.max_tilt,
            self.max_tilt,
        )
        des_roll = np.clip(
            -desired_accel[1] / self.quad.g,
            -self.max_tilt,
            self.max_tilt,
        )

        thrust = self.quad.mass * (self.quad.g + desired_accel[2])
        thrust /= max(np.cos(att[0]) * np.cos(att[1]), 0.7)
        thrust = float(np.clip(thrust, self.min_thrust, self.max_thrust))

        # Attitude -> torques (N·m) with angular-rate damping.
        roll_tau = self.roll_ctrl.update(
            des_roll,
            att[0],
            dt,
        ) - self.rate_damping[0] * omega[0]
        pitch_tau = self.pitch_ctrl.update(
            des_pitch,
            att[1],
            dt,
        ) - self.rate_damping[1] * omega[1]
        yaw_tau = self.yaw_ctrl.update(
            target_yaw,
            att[2],
            dt,
        ) - self.rate_damping[2] * omega[2]
        torques = np.clip(
            np.array([roll_tau, pitch_tau, yaw_tau]),
            -self.max_torque,
            self.max_torque,
        )

        desired_wrench = np.array([thrust, torques[0], torques[1], torques[2]])

        try:
            allocation = self.quad.allocation_matrix()
            w_sq = np.linalg.solve(allocation, desired_wrench)
        except np.linalg.LinAlgError:
            hover_w = np.sqrt(
                self.quad.mass * self.quad.g / (4 * self.quad.k_f)
            )
            return np.full(4, hover_w)

        motor_speeds = np.sqrt(np.maximum(w_sq, 0.0))
        return np.clip(motor_speeds, 0.0, self.max_motor_speed)
