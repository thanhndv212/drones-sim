"""LQR controller for quadcopter stabilization.

Linearizes the plant around hover and solves the continuous-time Algebraic
Riccati Equation to obtain the full-state feedback gain K.

Control law:  u_delta = -K @ x_error
Wrench:       [T, tau_phi, tau_theta, tau_psi] = hover_wrench + u_delta
Motors:       allocation_matrix inverse maps wrench -> motor speeds
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_continuous_are

from ..dynamics.quadcopter import QuadcopterDynamics


class LQRController:
    """Full-state feedback LQR for quadcopter position and attitude control.

    The linearization is computed once at construction around the hover trim
    point (phi=theta=0, T=mg).  The resulting gain K is constant (offline
    optimal).

    Interface matches QuadcopterController.compute() so both can be used
    interchangeably in simulation loops:

        motors = lqr.compute(target_pos, target_yaw, dt, prev_target_pos)

    Parameters
    ----------
    quad:
        Live QuadcopterDynamics instance (reads state each step).
    Q:
        12x12 state-error penalty matrix.  Defaults to a diagonal matrix
        that penalizes position and attitude tracking most.
    R:
        4x4 control-effort penalty matrix for [T, tau_phi, tau_theta, tau_psi].
    """

    def __init__(
        self,
        quad: QuadcopterDynamics,
        Q: NDArray | None = None,
        R: NDArray | None = None,
    ) -> None:
        self.quad = quad
        self.K = self._compute_gain(Q, R)

    # ------------------------------------------------------------------
    # Hover linearization
    # ------------------------------------------------------------------

    def _linearize_hover(self) -> tuple[NDArray, NDArray]:
        """Return (A, B) continuous-time hover linearization matrices."""
        m = self.quad.mass
        g = self.quad.g
        k_d = self.quad.k_d
        Ix, Iy, Iz = np.diag(self.quad.I)

        # State: [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
        A = np.zeros((12, 12))
        # Position kinematics: dp/dt = v
        A[0, 3] = 1.0
        A[1, 4] = 1.0
        A[2, 5] = 1.0
        # Velocity: aero drag + gravity coupling through small-angle attitude
        A[3, 3] = -k_d / m
        A[3, 7] = g          # d(vx)/d(theta)  (pitch -> forward accel)
        A[4, 4] = -k_d / m
        A[4, 6] = -g         # d(vy)/d(phi)    (roll  -> lateral accel)
        A[5, 5] = -k_d / m
        # Euler angle kinematics (linearized, small angles)
        A[6, 9]  = 1.0       # dphi/dt   = p
        A[7, 10] = 1.0       # dtheta/dt = q
        A[8, 11] = 1.0       # dpsi/dt   = r

        # Input: [T, tau_phi, tau_theta, tau_psi]
        B = np.zeros((12, 4))
        B[5, 0]  = 1.0 / m   # d(vz)/dT
        B[9, 1]  = 1.0 / Ix  # d(p)/d(tau_phi)
        B[10, 2] = 1.0 / Iy  # d(q)/d(tau_theta)
        B[11, 3] = 1.0 / Iz  # d(r)/d(tau_psi)

        return A, B

    def _compute_gain(self, Q: NDArray | None, R: NDArray | None) -> NDArray:
        """Solve ARE and return gain K = R^{-1} B^T P."""
        A, B = self._linearize_hover()

        if Q is None:
            q_diag = np.array([
                10.0, 10.0, 12.0,  # position  x, y, z
                2.0,  2.0,  4.0,   # velocity  vx, vy, vz
                5.0,  5.0,  3.0,   # attitude  phi, theta, psi
                0.5,  0.5,  0.3,   # body rate p, q, r
            ])
            Q = np.diag(q_diag)

        if R is None:
            R = np.diag([0.01, 50.0, 50.0, 20.0])  # [T, tau_phi, tau_theta, tau_psi]

        P = solve_continuous_are(A, B, Q, R)
        return np.linalg.inv(R) @ B.T @ P

    # ------------------------------------------------------------------
    # Runtime interface
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """No integrators; nothing to reset."""

    def compute(
        self,
        target_pos: NDArray,
        target_yaw: float,
        dt: float,
        prev_target_pos: NDArray | None = None,
    ) -> NDArray:
        """Compute motor speeds for a given position and yaw target.

        Parameters
        ----------
        target_pos:
            Desired 3-D position [x, y, z] in world frame.
        target_yaw:
            Desired yaw angle (rad).
        dt:
            Time step (unused by LQR; kept for interface compatibility).
        prev_target_pos:
            Previous target (unused; kept for interface compatibility).

        Returns
        -------
        motor_speeds: NDArray shape (4,)
            Commanded rotor speeds in rad/s.
        """
        m  = self.quad.mass
        g  = self.quad.g
        kf = self.quad.k_f

        pos   = self.quad.get_position()
        vel   = self.quad.get_velocity()
        att   = self.quad.get_attitude()    # [phi, theta, psi]
        omega = self.quad.get_angular_velocity()  # [p, q, r]

        # Error w.r.t. target (target velocity / rate = 0)
        x_err = np.concatenate([
            pos - target_pos,
            vel,
            att - np.array([0.0, 0.0, target_yaw]),
            omega,
        ])
        # Wrap yaw error to [-pi, pi]
        x_err[8] = (x_err[8] + np.pi) % (2.0 * np.pi) - np.pi

        # LQR feedback: delta wrench
        u_delta = -self.K @ x_err  # [delta_T, tau_phi, tau_theta, tau_psi]

        # Absolute thrust with hover compensation
        T = float(np.clip(m * g + u_delta[0], 0.2 * m * g, 2.0 * m * g))
        tau_max = np.array([0.25, 0.25, 0.08])
        torques = np.clip(u_delta[1:], -tau_max, tau_max)

        wrench = np.array([T, torques[0], torques[1], torques[2]])

        # Invert allocation matrix -> squared motor speeds
        A_alloc = self.quad.allocation_matrix()
        try:
            omega_sq = np.linalg.solve(A_alloc, wrench)
        except np.linalg.LinAlgError:
            hover_w = float(np.sqrt(m * g / (4.0 * kf)))
            return np.full(4, hover_w)

        motor_speeds = np.sqrt(np.maximum(omega_sq, 0.0))
        return np.clip(motor_speeds, 0.0, 4000.0)
