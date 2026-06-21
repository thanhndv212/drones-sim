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
    # Thrust is defined as delta from hover (mass * g).
    # Negative → descend, positive → ascend, zero → exactly hover.
    # Total thrust is clipped to [0.2*hover, 2.0*hover] in to_motors.
    _thrust_low = -0.8 * 9.81    # total = 0.2 * hover (minimum)
    _thrust_high = 1.0 * 9.81    # total = 2.0 * hover (maximum)
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


class LQRResidualAction:
    """LQR baseline + learned residual.

    Action: 4-dim delta in [-1, 1] applied as a multiplicative offset
    to the LQR-computed motor speeds.

    The built-in ``LQRController`` provides a stable hover baseline;
    the policy only needs to learn corrections to tighten tracking.
    This makes the learning problem dramatically easier — the drone
    never crashes during exploration because LQR always keeps it stable.
    """

    dim = 4
    _scale: float = 500.0  # max motor speed delta (rad/s) from LQR baseline

    def __init__(self, target_yaw: float = 0.0, **kwargs):
        self._target_yaw = target_yaw
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def low(self) -> np.ndarray:
        return np.full(self.dim, -1.0, dtype=np.float32)

    @property
    def high(self) -> np.ndarray:
        return np.full(self.dim, 1.0, dtype=np.float32)

    def to_motors(self, quad, action: NDArray) -> NDArray:
        from ..control.lqr import LQRController

        # Ensure we have a cached LQR instance keyed on the quad object
        lqr = getattr(self, "_lqr", None)
        if lqr is None or lqr.quad is not quad:
            lqr = LQRController(quad)
            self._lqr = lqr

        # Use the stored task target (set by env before calling to_motors)
        target = getattr(self, "_task_target", quad.get_position())
        lqr_motors = lqr.compute(target, self._target_yaw, dt=0.01)

        # Apply learned delta
        delta = action * self._scale
        motors = lqr_motors + delta
        return np.clip(motors, 0.0, 4000.0)


class VelocityLevelAction:
    """High-level: desired world-frame velocity + yaw rate.

    Action: [vx, vy, vz (m/s world), yaw_rate (rad/s)].

    A built-in cascaded P-controller converts the desired velocity into
    motor speeds::

        vel_error → desired accel → desired attitude + thrust
        attitude_error → angular velocity → torque
        yaw_rate_error → torque

    This avoids the accumulating-attitude problem of body-rate commands
    and makes the policy's job much easier — the drone is inherently
    stable and the policy only needs to decide *where* to go.

    Gains are tuned for the default quadcopter parameters (m=1.0 kg,
    I=diag(0.01,0.01,0.018), k_f=1e-6, arm=0.2).
    """

    dim = 4
    _vel_low = -3.0
    _vel_high = 3.0
    _vz_low = -2.0
    _vz_high = 2.0
    _yaw_rate_low = -1.0
    _yaw_rate_high = 1.0

    # ── controller gains (tunable) ──
    kp_vel: float = 2.0      # velocity → acceleration
    kp_att: float = 6.0      # attitude error → angular velocity
    kp_rate: float = 4.0     # angular-velocity error → torque
    kp_yaw: float = 2.0      # yaw-rate error → torque

    def __init__(self, **gains):
        for k, v in gains.items():
            setattr(self, k, v)

    @property
    def low(self) -> np.ndarray:
        return np.array([self._vel_low, self._vel_low, self._vz_low, self._yaw_rate_low], dtype=np.float32)

    @property
    def high(self) -> np.ndarray:
        return np.array([self._vel_high, self._vel_high, self._vz_high, self._yaw_rate_high], dtype=np.float32)

    def to_motors(self, quad, action: NDArray) -> NDArray:
        vx_des, vy_des, vz_des, yaw_rate_des = action

        m = quad.mass
        g = quad.g
        hover = m * g

        # ── 1. velocity → desired acceleration (world frame) ──
        vel = quad.get_velocity()
        ax_des = self.kp_vel * (vx_des - vel[0])
        ay_des = self.kp_vel * (vy_des - vel[1])
        az_des = self.kp_vel * (vz_des - vel[2])

        # ── 2. desired accel → thrust + desired attitude ──
        total_thrust = np.clip(m * (g + az_des), 0.2 * hover, 2.0 * hover)

        # Small-angle mapping: ax_des ≈ g * tan(theta), ay_des ≈ -g * tan(phi)
        phi_des = np.arctan2(-ay_des, g)
        theta_des = np.arctan2(ax_des, g)
        phi_des = np.clip(phi_des, -0.8, 0.8)       # ~45° limit
        theta_des = np.clip(theta_des, -0.8, 0.8)

        # ── 3. attitude error → angular velocity command ──
        att = quad.get_attitude()  # [roll, pitch, yaw]
        e_phi = phi_des - att[0]
        e_theta = theta_des - att[1]
        # Wrap to [-pi, pi]
        e_phi = (e_phi + np.pi) % (2 * np.pi) - np.pi
        e_theta = (e_theta + np.pi) % (2 * np.pi) - np.pi

        p_des = self.kp_att * e_phi
        q_des = self.kp_att * e_theta
        r_des = yaw_rate_des

        # ── 4. angular-velocity error → torque ──
        omega = quad.get_angular_velocity()  # [p, q, r] body frame
        tau_phi = self.kp_rate * (p_des - omega[0])
        tau_theta = self.kp_rate * (q_des - omega[1])
        tau_psi = self.kp_yaw * (r_des - omega[2])

        # ── 5. wrench → motor speeds ──
        wrench = np.array([total_thrust, tau_phi, tau_theta, tau_psi])
        alloc = quad.allocation_matrix()
        try:
            w_sq = np.linalg.solve(alloc, wrench)
        except np.linalg.LinAlgError:
            hover_w = np.sqrt(hover / (4 * quad.k_f))
            return np.full(4, hover_w)

        motor_speeds = np.sqrt(np.maximum(w_sq, 0.0))
        return np.clip(motor_speeds, 0.0, 4000.0)
