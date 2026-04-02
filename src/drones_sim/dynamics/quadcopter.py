"""Quadcopter rigid-body dynamics based on Newton-Euler equations.

Consolidated from quadcopter_simulation.py.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..math_utils import euler_to_rotation_matrix, angular_vel_to_euler_rates


class QuadcopterDynamics:
    """12-state quadcopter model: [pos(3), vel(3), euler(3), omega_body(3)].

    Motor layout (X-configuration, looking from above):
        Motor 1: +x   (front)
        Motor 2: +y   (right)
        Motor 3: -x   (back)
        Motor 4: -y   (left)

    Motor dynamics: first-order lag  tau_m * d(omega)/dt + omega = omega_cmd.
    Set motor_time_constant=0.0 (default) for instantaneous motor response.
    Typical small UAV value: 0.03–0.08 s.
    """

    def __init__(
        self,
        mass: float = 1.0,
        arm_length: float = 0.2,
        inertia: NDArray | None = None,
        k_f: float = 1.0e-6,
        k_m: float = 1.0e-7,
        k_d: float = 0.1,
        g: float = 9.81,
        motor_time_constant: float = 0.0,
    ):
        self.mass = mass
        self.arm_length = arm_length
        self.I = inertia if inertia is not None else np.diag([0.01, 0.01, 0.018])
        self.k_f = k_f
        self.k_m = k_m
        self.k_d = k_d
        self.g = g
        self.motor_time_constant = motor_time_constant

        # state: [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
        self.state = np.zeros(12)
        # actual motor speeds (lag state); equals commanded speeds when tau_m=0
        self.motor_states = np.zeros(4)

    def reset(self, position: NDArray | None = None, attitude: NDArray | None = None) -> None:
        self.state = np.zeros(12)
        self.motor_states = np.zeros(4)
        if position is not None:
            self.state[:3] = position
        if attitude is not None:
            self.state[6:9] = attitude

    # -- state accessors ---------------------------------------------------

    def get_position(self) -> NDArray:
        return self.state[:3].copy()

    def get_velocity(self) -> NDArray:
        return self.state[3:6].copy()

    def get_attitude(self) -> NDArray:
        return self.state[6:9].copy()

    def get_angular_velocity(self) -> NDArray:
        return self.state[9:12].copy()

    # -- dynamics ----------------------------------------------------------

    def rotation_matrix(self) -> NDArray:
        phi, theta, psi = self.state[6:9]
        return euler_to_rotation_matrix(phi, theta, psi)

    def _derivatives(self, state: NDArray, motor_speeds: NDArray) -> NDArray:
        """Compute state derivatives for a given state and motor speeds."""
        T = self.k_f * np.sum(motor_speeds**2)
        tau_phi = self.k_f * self.arm_length * (motor_speeds[1] ** 2 - motor_speeds[3] ** 2)
        tau_theta = self.k_f * self.arm_length * (motor_speeds[2] ** 2 - motor_speeds[0] ** 2)
        tau_psi = self.k_m * (
            motor_speeds[0] ** 2 - motor_speeds[1] ** 2
            + motor_speeds[2] ** 2 - motor_speeds[3] ** 2
        )
        torques = np.array([tau_phi, tau_theta, tau_psi])

        vx, vy, vz = state[3:6]
        phi, theta, psi = state[6:9]
        p, q, r = state[9:12]

        R = euler_to_rotation_matrix(phi, theta, psi)
        F_thrust = R @ np.array([0, 0, T])
        F_drag = -self.k_d * np.array([vx, vy, vz])
        F_gravity = np.array([0, 0, -self.mass * self.g])
        accel = (F_thrust + F_drag + F_gravity) / self.mass

        omega = np.array([p, q, r])
        angular_accel = np.linalg.solve(self.I, torques - np.cross(omega, self.I @ omega))
        euler_rates = angular_vel_to_euler_rates(phi, theta, omega)

        deriv = np.zeros(12)
        deriv[:3] = [vx, vy, vz]
        deriv[3:6] = accel
        deriv[6:9] = euler_rates
        deriv[9:12] = angular_accel
        return deriv

    def get_motor_speeds(self) -> NDArray:
        """Return actual motor speeds (after lag filter if enabled)."""
        return self.motor_states.copy()

    def update(self, dt: float, motor_speeds: NDArray) -> NDArray:
        """Advance one time-step given 4 commanded motor speeds (rad/s) using RK4.

        When motor_time_constant > 0 the actual rotor speeds follow a first-order
        lag:  tau_m * d(omega)/dt + omega = omega_cmd.
        """
        motor_cmds = np.maximum(motor_speeds, 0.0)

        # --- motor lag integration (discrete first-order filter) ----------
        if self.motor_time_constant > 0.0:
            alpha = dt / self.motor_time_constant
            self.motor_states += alpha * (motor_cmds - self.motor_states)
            self.motor_states = np.maximum(self.motor_states, 0.0)
            actual_motors = self.motor_states
        else:
            self.motor_states = motor_cmds
            actual_motors = motor_cmds

        # --- RK4 plant integration ----------------------------------------
        k1 = self._derivatives(self.state, actual_motors)
        k2 = self._derivatives(self.state + 0.5 * dt * k1, actual_motors)
        k3 = self._derivatives(self.state + 0.5 * dt * k2, actual_motors)
        k4 = self._derivatives(self.state + dt * k3, actual_motors)

        self.state += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return self.state.copy()

    # -- motor allocation helpers ------------------------------------------

    def allocation_matrix(self) -> NDArray:
        """Return the 4x4 matrix A mapping [w1^2, w2^2, w3^2, w4^2] to [T, tau_phi, tau_theta, tau_psi]."""
        kf, km, L = self.k_f, self.k_m, self.arm_length
        return np.array([
            [kf, kf, kf, kf],
            [0, kf * L, 0, -kf * L],
            [-kf * L, 0, kf * L, 0],
            [km, -km, km, -km],
        ])
