"""Quadcopter rigid-body dynamics based on Newton–Euler equations.

The internal state is quaternion-based (13-state)::

    state[0:3]   position      [m]       ENU world frame (z-up)
    state[3:6]   velocity      [m/s]     ENU world frame
    state[6:10]  quaternion    [-]       Hamilton [w, x, y, z], body→world
    state[10:13] omega_body    [rad/s]   body-frame angular velocity [p, q, r]

Quaternions avoid the Euler-angle singularity at 90° pitch (gimbal lock) and
match the convention used by ``ExtendedKalmanFilter`` (also ``[w,x,y,z]``).

Public accessors preserve the original Euler-based API so that controllers,
examples, and existing tests keep working without changes:

    get_position()        -> (3,)
    get_velocity()        -> (3,)
    get_attitude()        -> (3,)   Euler [roll, pitch, yaw] derived from the quat
    get_quaternion()      -> (4,)   [w, x, y, z]   (new, idiomatic)
    get_angular_velocity()-> (3,)   [p, q, r] body frame

Motor layout (X-configuration, looking from above):
    Motor 1: +x   (front)
    Motor 2: +y   (right)
    Motor 3: -x   (back)
    Motor 4: -y   (left)

Motor dynamics: first-order lag  tau_m * d(omega)/dt + omega = omega_cmd.
Set motor_time_constant=0.0 (default) for instantaneous motor response.
Typical small-UAV value: 0.03–0.08 s.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..math_utils import (
    quat_derivative,
    quat_from_euler,
    quat_normalize,
    quat_to_euler,
    quat_to_rotation_matrix,
)


class QuadcopterDynamics:
    """13-state quaternion quadcopter model.

    See module docstring for the state layout and public API.  The class keeps
    the original Euler-based accessors alive (``get_attitude``) while storing
    attitude as a unit quaternion internally — eliminating gimbal lock while
    remaining a drop-in replacement for the previous 12-state Euler plant.
    """

    # State size changed from 12 (Euler) to 13 (quaternion).
    STATE_SIZE = 13

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

        # state: [pos(3), vel(3), quat(4), omega_body(3)]
        self.state = np.zeros(self.STATE_SIZE)
        self.state[6] = 1.0  # identity quaternion [w, x, y, z]
        # actual motor speeds (lag state); equals commanded speeds when tau_m=0
        self.motor_states = np.zeros(4)

    def reset(self, position: NDArray | None = None, attitude: NDArray | None = None) -> None:
        """Reset state to rest at origin with identity orientation.

        ``attitude`` (if given) is interpreted as Euler [roll, pitch, yaw] and
        converted to a quaternion — preserving the legacy call signature.
        """
        self.state = np.zeros(self.STATE_SIZE)
        self.state[6] = 1.0  # identity quaternion
        self.motor_states = np.zeros(4)
        if position is not None:
            self.state[:3] = position
        if attitude is not None:
            # Backward-compatible: caller passes Euler angles.
            self.state[6:10] = quat_from_euler(attitude[0], attitude[1], attitude[2])

    # -- state accessors ---------------------------------------------------

    def get_position(self) -> NDArray:
        return self.state[:3].copy()

    def get_velocity(self) -> NDArray:
        return self.state[3:6].copy()

    def get_attitude(self) -> NDArray:
        """Euler angles [roll, pitch, yaw] derived from the internal quaternion.

        Backward-compatible with the previous 12-state plant so controllers and
        examples that consume Euler angles continue to work unmodified.
        """
        return quat_to_euler(self.state[6:10])

    def get_quaternion(self) -> NDArray:
        """Unit quaternion [w, x, y, z] (body→world)."""
        return self.state[6:10].copy()

    def get_angular_velocity(self) -> NDArray:
        return self.state[10:13].copy()

    # -- dynamics ----------------------------------------------------------

    def rotation_matrix(self) -> NDArray:
        """World←body rotation matrix from the internal quaternion."""
        return quat_to_rotation_matrix(self.state[6:10])

    def _derivatives(self, state: NDArray, motor_speeds: NDArray) -> NDArray:
        """Compute the 13-state derivative for a given state and motor speeds.

        Quaternion kinematics are integrated as

            q_dot = 0.5 * q ⊗ [0, omega]

        which preserves attitude information through arbitrary rotations
        (no gimbal-lock singularity).  Renormalisation happens once per
        ``update()`` step (see ``update``).
        """
        T = self.k_f * np.sum(motor_speeds**2)
        tau_phi = self.k_f * self.arm_length * (motor_speeds[1] ** 2 - motor_speeds[3] ** 2)
        tau_theta = self.k_f * self.arm_length * (motor_speeds[2] ** 2 - motor_speeds[0] ** 2)
        tau_psi = self.k_m * (
            motor_speeds[0] ** 2 - motor_speeds[1] ** 2
            + motor_speeds[2] ** 2 - motor_speeds[3] ** 2
        )
        torques = np.array([tau_phi, tau_theta, tau_psi])

        vel = state[3:6]
        quat = state[6:10]
        omega = state[10:13]

        R = quat_to_rotation_matrix(quat)
        F_thrust = R @ np.array([0.0, 0.0, T])
        F_drag = -self.k_d * vel
        F_gravity = np.array([0.0, 0.0, -self.mass * self.g])
        accel = (F_thrust + F_drag + F_gravity) / self.mass

        angular_accel = np.linalg.solve(self.I, torques - np.cross(omega, self.I @ omega))
        quat_dot = quat_derivative(quat, omega)

        deriv = np.zeros(self.STATE_SIZE)
        deriv[:3] = vel
        deriv[3:6] = accel
        deriv[6:10] = quat_dot
        deriv[10:13] = angular_accel
        return deriv

    def get_motor_speeds(self) -> NDArray:
        """Return actual motor speeds (after lag filter if enabled)."""
        return self.motor_states.copy()

    def update(self, dt: float, motor_speeds: NDArray) -> NDArray:
        """Advance one time-step given 4 commanded motor speeds (rad/s) using RK4.

        When motor_time_constant > 0 the actual rotor speeds follow a first-order
        lag:  tau_m * d(omega)/dt + omega = omega_cmd.

        The quaternion part of the state is renormalised after the RK4 step to
        counter numerical drift (quaternions slowly lose unit norm under
        fixed-step RK4, which would otherwise corrupt the rotation matrix).
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
        # Renormalise quaternion to suppress RK4 norm drift.
        self.state[6:10] = quat_normalize(self.state[6:10])
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
