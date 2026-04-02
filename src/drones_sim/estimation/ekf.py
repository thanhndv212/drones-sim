"""Extended Kalman Filter for IMU sensor fusion.

Two modes of operation:
1. **Full-state EKF** (n=10): position + velocity + quaternion.
   Uses proper analytical Jacobians for accel/mag measurement updates.
   Consolidated from imu_ekf_simulation.py (the most mathematically rigorous version).

2. **AHRS-aided EKF** (n=9): position + velocity + accel_bias.
   Attitude handled by the AHRS complementary filter; EKF estimates translational
   states and accelerometer bias.  Supports adaptive noise from innovation monitoring.
   Consolidated from imu_ekf_fusion_enhanced.py.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from numpy.typing import NDArray

from ..math_utils import (
    quat_derivative,
    quat_angular_velocity_jacobian,
    quat_normalize,
    quat_to_rotation_matrix,
)
from .ahrs import AHRS


# ---------------------------------------------------------------------------
# Full-state EKF (10-state) — from imu_ekf_simulation.py
# ---------------------------------------------------------------------------

class ExtendedKalmanFilter:
    """16-state EKF for quadcopter navigation.

    State vector x[16]::

        x[0:3]   position      [m]      ENU world frame (z-up)
        x[3:6]   velocity      [m/s]    ENU world frame
        x[6:10]  quaternion    [-]      Hamilton [w, x, y, z], body→world
        x[10:13] accel_bias    [m/s²]   body-frame accelerometer bias (random walk)
        x[13:16] gyro_bias     [rad/s]  body-frame gyroscope bias (random walk)

    Gravity convention (ENU, z-up)::

        self.gravity = [0, 0, 9.81]   # upward reaction direction

        At rest  : IMU reads  Rᵀ @ [0,0,9.81]  in body frame  (positive body-z)
        Velocity : a_world = R @ f_body - self.gravity
        Correct  : expected  = Rᵀ @ self.gravity   (NOT -Rᵀ @ g)

    Key identity used throughout::

        f_body = Rᵀ (a_linear + g_up)   # specific force = what IMU measures
        a_linear = R @ f_body - g_up     # recover linear accel for integration

    Backward compatibility:
        ``initial_state`` may be a 10-element vector (legacy).  It is padded
        with zeros for the six bias states automatically.

    Covariance updates use the Joseph form
        ``P = (I-KH) P (I-KH)ᵀ + K R Kᵀ``
    for numerical stability (avoids asymmetric drift in P).
    """

    def __init__(
        self,
        dt: float,
        initial_state: NDArray | None = None,
        gravity: NDArray | None = None,
        mag_ref: NDArray | None = None,
    ):
        self.n = 16
        self.dt = dt

        # Accept legacy 10-element initial_state and pad with zero biases.
        if initial_state is not None:
            if len(initial_state) == 10:
                s = np.zeros(16)
                s[:10] = initial_state
                self.x = s
            else:
                self.x = initial_state.copy()
        else:
            self.x = np.zeros(self.n)
            self.x[6] = 1.0  # identity quaternion

        self.gravity = gravity if gravity is not None else np.array([0.0, 0.0, 9.81])
        self.mag_ref = mag_ref if mag_ref is not None else np.array([20.0, 0.0, -40.0])

        # Covariance
        self.P = np.eye(self.n) * 0.01
        self.P[0:3, 0:3] *= 0.01
        self.P[3:6, 3:6] *= 0.1
        self.P[6:10, 6:10] *= 0.001
        self.P[10:13, 10:13] = np.eye(3) * 0.01    # accel bias init uncertainty
        self.P[13:16, 13:16] = np.eye(3) * 0.0001  # gyro  bias init uncertainty

        # Process noise — diagonal, tuned to sensor noise spectral densities.
        # Bias states model slow random walks; their Q entries are tiny.
        self.Q = np.zeros((self.n, self.n))
        self.Q[0:3, 0:3]   = np.eye(3) * 1e-6    # position (driven by velocity)
        self.Q[3:6, 3:6]   = np.eye(3) * 0.01    # velocity  (accel noise ×dt)
        self.Q[6:10, 6:10] = np.eye(4) * 0.002   # quaternion (gyro noise ×dt)
        self.Q[10:13, 10:13] = np.eye(3) * 2.5e-5  # accel bias random walk
        self.Q[13:16, 13:16] = np.eye(3) * 1e-7    # gyro  bias random walk

        # Measurement noise
        self.R_accel = np.eye(3) * 0.003   # σ≈0.05 m/s² → var≈0.0025
        self.R_mag   = np.eye(3) * 0.5

    # -- prediction --------------------------------------------------------

    def predict(self, gyro: NDArray, accel: NDArray | None = None) -> None:
        """Propagate state forward by dt.

        Args:
            gyro:  Body-frame angular velocity (rad/s), 3-vector.
            accel: Body-frame specific force (m/s²), 3-vector.
                   When provided, linear acceleration is integrated into velocity
                   and position (midpoint scheme: pos += v*dt + ½*a*dt²).
                   If None the velocity/position states are held constant.
        """
        pos  = self.x[:3]
        vel  = self.x[3:6]
        quat = self.x[6:10]
        accel_bias = self.x[10:13]
        gyro_bias  = self.x[13:16]

        # Bias-corrected measurements
        gyro_c  = gyro - gyro_bias
        accel_c = (accel - accel_bias) if accel is not None else None

        # Quaternion kinematics
        q_dot    = quat_derivative(quat, gyro_c)
        new_quat = quat_normalize(quat + q_dot * self.dt)

        # Velocity + position propagation from specific force
        if accel_c is not None:
            R = quat_to_rotation_matrix(quat)
            a_world = R @ accel_c - self.gravity   # linear acceleration in world
            new_vel = vel + a_world * self.dt
            # Midpoint scheme: pos += v*dt + ½*a*dt²
            new_pos = pos + vel * self.dt + 0.5 * a_world * self.dt ** 2
        else:
            new_vel = vel
            new_pos = pos + vel * self.dt

        self.x[:3]   = new_pos
        self.x[3:6]  = new_vel
        self.x[6:10] = new_quat
        # Bias states integrate as a random walk (no deterministic dynamics)

        # --- State transition Jacobian F (16×16) ---------------------------
        F = np.eye(self.n)

        # Position → velocity coupling
        F[0:3, 3:6] = np.eye(3) * self.dt

        # Quaternion kinematics Jacobian (∂new_q / ∂q)
        F[6:10, 6:10] = np.eye(4) + quat_angular_velocity_jacobian(gyro_c) * self.dt

        if accel_c is not None:
            R = quat_to_rotation_matrix(quat)

            # Velocity sensitivity to accel_bias: ∂vel_new/∂b_a = -R*dt
            F[3:6, 10:13] = -R * self.dt

            # Position sensitivity to accel_bias (from midpoint term)
            F[0:3, 10:13] = -0.5 * R * self.dt ** 2

        # Quaternion sensitivity to gyro_bias: ∂new_q/∂b_g = -∂q_dot/∂ω * dt
        # q_dot = 0.5 * quat_multiply(q, [0, ω]), so ∂q_dot/∂ω is the 4×3 matrix Xi(q):
        #   Xi(q) = 0.5 * [[-x,-y,-z], [w,-z,y], [z,w,-x], [-y,x,w]]
        w, x, y, z = quat
        Xi = 0.5 * np.array([
            [-x, -y, -z],
            [ w, -z,  y],
            [ z,  w, -x],
            [-y,  x,  w],
        ])
        F[6:10, 13:16] = -Xi * self.dt

        self.P = F @ self.P @ F.T + self.Q

    # -- accelerometer correction ------------------------------------------

    def correct_accel(self, accel: NDArray) -> None:
        """Attitude correction from accelerometer (gravity direction).

        Only corrects quaternion — velocity is propagated in predict() from
        the same accel measurement.  The attitude update columns of H are
        non-zero; position/velocity/bias columns are zero.
        """
        quat      = self.x[6:10]
        accel_bias = self.x[10:13]
        R         = quat_to_rotation_matrix(quat)
        expected  = R.T @ self.gravity
        residual  = (accel - accel_bias) - expected

        H = self._accel_jacobian(quat)
        S = H @ self.P @ H.T + self.R_accel
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x += K @ residual
        self.x[6:10] = quat_normalize(self.x[6:10])
        assert abs(np.linalg.norm(self.x[6:10]) - 1.0) < 1e-3, (
            f"EKF quaternion norm={np.linalg.norm(self.x[6:10]):.6f} after correct_accel "
            "\u2014 filter diverging"
        )
        # Joseph form: P = (I-KH)P(I-KH)^T + K*R_accel*K^T
        IKH = np.eye(self.n) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ self.R_accel @ K.T

    # -- magnetometer correction -------------------------------------------

    def correct_mag(self, mag: NDArray) -> None:
        quat = self.x[6:10]
        R    = quat_to_rotation_matrix(quat)
        expected = R.T @ self.mag_ref
        residual = mag - expected    # mag has its own fixed bias; no state bias term

        H = self._mag_jacobian(quat)
        S = H @ self.P @ H.T + self.R_mag
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x += K @ residual
        self.x[6:10] = quat_normalize(self.x[6:10])
        assert abs(np.linalg.norm(self.x[6:10]) - 1.0) < 1e-3, (
            f"EKF quaternion norm={np.linalg.norm(self.x[6:10]):.6f} after correct_mag "
            "\u2014 filter diverging"
        )
        IKH = np.eye(self.n) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ self.R_mag @ K.T

    # -- GPS / barometer position correction -------------------------------

    def correct_position(
        self, pos_meas: NDArray, R_pos: NDArray | None = None
    ) -> None:
        """Full 3-D GPS position update.  pos_meas shape: (3,)."""
        m = len(pos_meas)
        if R_pos is None:
            R_pos = np.eye(m) * 0.1
        H = np.zeros((m, self.n))
        H[:m, :m] = np.eye(m)
        residual = pos_meas - self.x[:m]
        S = H @ self.P @ H.T + R_pos
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x += K @ residual
        self.x[6:10] = quat_normalize(self.x[6:10])
        IKH = np.eye(self.n) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R_pos @ K.T

    def correct_velocity(
        self, vel_meas: NDArray, R_vel: NDArray | None = None
    ) -> None:
        """3-D velocity update from GPS Doppler / DVL.  vel_meas shape: (3,)."""
        if R_vel is None:
            R_vel = np.eye(3) * 0.01   # GPS vel std~0.1 m/s → var=0.01
        H = np.zeros((3, self.n))
        H[0:3, 3:6] = np.eye(3)
        residual = vel_meas - self.x[3:6]
        S = H @ self.P @ H.T + R_vel
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x += K @ residual
        self.x[6:10] = quat_normalize(self.x[6:10])
        IKH = np.eye(self.n) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R_vel @ K.T

    def correct_altitude(self, z_meas: float, r_z: float = 0.05) -> None:
        """1-D barometer altitude update.  Anchors vertical dead-reckoning."""
        H = np.zeros((1, self.n))
        H[0, 2] = 1.0
        residual = np.array([z_meas - self.x[2]])
        R_z = np.array([[r_z]])
        S = H @ self.P @ H.T + R_z
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x += (K @ residual).ravel()
        self.x[6:10] = quat_normalize(self.x[6:10])
        IKH = np.eye(self.n) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R_z @ K.T

    # -- state access ------------------------------------------------------

    def get_state(self) -> dict:
        return {
            "position":   self.x[:3].copy(),
            "velocity":   self.x[3:6].copy(),
            "quaternion": self.x[6:10].copy(),
            "accel_bias": self.x[10:13].copy(),
            "gyro_bias":  self.x[13:16].copy(),
        }

    # -- Jacobians (analytical) --------------------------------------------

    def _accel_jacobian(self, q: NDArray) -> NDArray:
        """Analytical Jacobian d(R(q/|q|)^T @ g)/dq for the accelerometer measurement model.

        Derived from the Hamilton quaternion rotation matrix::

            R[w,x,y,z] (body→world), so R^T maps world→body.

        Because quat_to_rotation_matrix() normalizes q internally, the true
        Jacobian w.r.t. the raw quaternion parameters is the tangent-space
        projection of the algebraic derivative::

            H_proj[:, 6:10] = H_raw[:, 6:10] - outer(H_raw[:, 6:10] @ q, q)

        This removes the radial (along-q) component that normalization kills.
        State layout (16-state): indices 6-9 = [qw, qx, qy, qz]; bias columns are zero.
        """
        w, x, y, z = q
        gx, gy, gz = self.gravity
        H = np.zeros((3, self.n))

        # Row 0:  d(R^T @ g)[0] / d[w, x, y, z]  (unnormalized)
        H[0, 6] =  2 * z * gy - 2 * y * gz
        H[0, 7] =  2 * y * gy + 2 * z * gz
        H[0, 8] = -4 * y * gx + 2 * x * gy - 2 * w * gz
        H[0, 9] = -4 * z * gx + 2 * w * gy + 2 * x * gz

        # Row 1:  d(R^T @ g)[1] / d[w, x, y, z]
        H[1, 6] = -2 * z * gx + 2 * x * gz
        H[1, 7] =  2 * y * gx - 4 * x * gy + 2 * w * gz
        H[1, 8] =  2 * x * gx + 2 * z * gz
        H[1, 9] = -2 * w * gx - 4 * z * gy + 2 * y * gz

        # Row 2:  d(R^T @ g)[2] / d[w, x, y, z]
        H[2, 6] =  2 * y * gx - 2 * x * gy
        H[2, 7] =  2 * z * gx - 2 * w * gy - 4 * x * gz
        H[2, 8] =  2 * w * gx + 2 * z * gy - 4 * y * gz
        H[2, 9] =  2 * x * gx + 2 * y * gy

        # Tangent-space projection: remove the component along q (radial direction)
        Hq = H[:, 6:10]           # (3, 4)
        H[:, 6:10] = Hq - np.outer(Hq @ q, q)

        return H

    def _mag_jacobian(self, q: NDArray) -> NDArray:
        """d(R(q/|q|)^T m)/dq — analytical Jacobian of magnetometer measurement model.

        Identical structure to _accel_jacobian: both compute d(R^T v)/dq for a
        constant reference vector v (gravity vs. mag_ref).
        Includes tangent-space projection (see _accel_jacobian docstring).
        State layout (16-state): indices 6-9 = [qw, qx, qy, qz]; bias columns are zero.
        """
        w, x, y, z = q
        mx, my, mz = self.mag_ref
        H = np.zeros((3, self.n))

        # Row 0: d(R^T m)[0] / d[w, x, y, z]
        H[0, 6] =  2 * z * my - 2 * y * mz
        H[0, 7] =  2 * y * my + 2 * z * mz
        H[0, 8] = -4 * y * mx + 2 * x * my - 2 * w * mz
        H[0, 9] = -4 * z * mx + 2 * w * my + 2 * x * mz

        # Row 1: d(R^T m)[1] / d[w, x, y, z]
        H[1, 6] = -2 * z * mx + 2 * x * mz
        H[1, 7] =  2 * y * mx - 4 * x * my + 2 * w * mz
        H[1, 8] =  2 * x * mx + 2 * z * mz
        H[1, 9] = -2 * w * mx - 4 * z * my + 2 * y * mz

        # Row 2: d(R^T m)[2] / d[w, x, y, z]
        H[2, 6] =  2 * y * mx - 2 * x * my
        H[2, 7] =  2 * z * mx - 2 * w * my - 4 * x * mz
        H[2, 8] =  2 * w * mx + 2 * z * my - 4 * y * mz
        H[2, 9] =  2 * x * mx + 2 * y * my

        # Tangent-space projection
        Hq = H[:, 6:10]
        H[:, 6:10] = Hq - np.outer(Hq @ q, q)

        return H


# ---------------------------------------------------------------------------
# AHRS-aided adaptive EKF (9-state) — from imu_ekf_fusion_enhanced.py
# ---------------------------------------------------------------------------

class AdaptiveEKF:
    """9-state EKF: [pos(3), vel(3), accel_bias(3)].

    Attitude is estimated separately by an AHRS complementary filter.
    Process and measurement noise are adapted based on innovation statistics.
    """

    def __init__(
        self,
        dt: float,
        init_pos: NDArray | None = None,
        init_vel: NDArray | None = None,
        innovation_window: int = 20,
        gravity: NDArray | None = None,
        mag_ref: NDArray | None = None,
    ):
        self.n = 9
        self.dt = dt

        self.x = np.zeros(self.n)
        if init_pos is not None:
            self.x[:3] = init_pos
        if init_vel is not None:
            self.x[3:6] = init_vel

        self.gravity = gravity if gravity is not None else np.array([0.0, 0.0, 9.81])

        self.P = np.eye(self.n)
        self.P[:3, :3] *= 0.01
        self.P[3:6, 3:6] *= 0.1
        self.P[6:9, 6:9] *= 0.01

        self.base_Q = np.eye(self.n)
        self.base_Q[:3, :3] *= 0.01
        self.base_Q[3:6, 3:6] *= 0.1
        self.base_Q[6:9, 6:9] *= 0.001
        self.Q = self.base_Q.copy()

        self.base_R_accel = np.eye(3) * 0.1
        self.R_accel = self.base_R_accel.copy()

        self._innovations: deque[NDArray] = deque(maxlen=innovation_window)

        self.ahrs = AHRS(
            dt,
            accel_weight=0.02,
            mag_weight=0.01,
            gravity=self.gravity,
            mag_ref=mag_ref,
        )
        self.orientation = np.array([1.0, 0.0, 0.0, 0.0])
        self.gyro_bias = np.zeros(3)

    # -- prediction --------------------------------------------------------

    def predict(self, gyro: NDArray, accel: NDArray, mag: NDArray, adaptive_factor: float = 1.0) -> None:
        self.orientation, self.gyro_bias = self.ahrs.update(gyro, accel, mag)
        R = quat_to_rotation_matrix(self.orientation)

        accel_bias = self.x[6:9]
        accel_world = R @ (accel - accel_bias) + self.gravity

        pos, vel = self.x[:3], self.x[3:6]
        self.x[:3] = pos + vel * self.dt + 0.5 * accel_world * self.dt**2
        self.x[3:6] = vel + accel_world * self.dt

        F = np.eye(self.n)
        F[:3, 3:6] = np.eye(3) * self.dt
        F[3:6, 6:9] = -R * self.dt

        self.Q = self.base_Q * adaptive_factor
        self.P = F @ self.P @ F.T + self.Q

    # -- measurement update ------------------------------------------------

    def correct(self, accel: NDArray, adaptive_factor: float = 1.0) -> None:
        R = quat_to_rotation_matrix(self.orientation)
        accel_bias = self.x[6:9]
        expected = R.T @ (-self.gravity) + accel_bias
        innovation = accel - expected

        self._innovations.append(innovation)

        # Adaptive measurement noise
        if len(self._innovations) >= self._innovations.maxlen:
            cov = np.zeros((3, 3))
            for inn in self._innovations:
                cov += np.outer(inn, inn)
            cov /= len(self._innovations)
            self.R_accel = (self.base_R_accel + np.diag(np.diag(cov))) * adaptive_factor
        else:
            self.R_accel = self.base_R_accel * adaptive_factor

        H = np.zeros((3, self.n))
        H[:, 6:9] = np.eye(3)

        S = H @ self.P @ H.T + self.R_accel
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x += K @ innovation

        # Joseph form for numerical stability
        I = np.eye(self.n)
        IKH = I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ self.R_accel @ K.T

    # -- state access ------------------------------------------------------

    def get_state(self) -> dict:
        return {
            "position": self.x[:3].copy(),
            "velocity": self.x[3:6].copy(),
            "accel_bias": self.x[6:9].copy(),
            "quaternion": self.orientation.copy(),
            "gyro_bias": self.gyro_bias.copy(),
        }
