"""Quaternion and rotation matrix utilities.

Convention: quaternions are stored as [w, x, y, z] throughout this package.
scipy.spatial.transform.Rotation uses [x, y, z, w], so conversions are needed
at the boundary.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Quaternion operations
# ---------------------------------------------------------------------------

def quat_normalize(q: NDArray) -> NDArray:
    """Normalize a quaternion to unit magnitude."""
    mag = np.linalg.norm(q)
    return q / mag if mag > 0 else q


def quat_multiply(q1: NDArray, q2: NDArray) -> NDArray:
    """Hamilton product of two quaternions [w,x,y,z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_conjugate(q: NDArray) -> NDArray:
    """Conjugate (inverse for unit quaternion)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_to_rotation_matrix(q: NDArray) -> NDArray:
    """Convert unit quaternion [w,x,y,z] to 3x3 rotation matrix."""
    q = quat_normalize(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)],
    ])


def quat_from_euler(roll: float, pitch: float, yaw: float) -> NDArray:
    """Convert Euler angles (xyz intrinsic) to quaternion [w,x,y,z].

    Uses scipy internally for correctness, then converts to our convention.
    """
    from scipy.spatial.transform import Rotation as R
    q_xyzw = R.from_euler("xyz", [roll, pitch, yaw]).as_quat()
    return np.roll(q_xyzw, 1)  # xyzw -> wxyz


def quat_to_euler(q: NDArray) -> NDArray:
    """Convert quaternion [w,x,y,z] to Euler angles [roll, pitch, yaw]."""
    from scipy.spatial.transform import Rotation as R
    q_xyzw = np.roll(q, -1)  # wxyz -> xyzw
    return R.from_quat(q_xyzw).as_euler("xyz")


def quat_derivative(q: NDArray, omega: NDArray) -> NDArray:
    """Quaternion time-derivative given body angular velocity omega."""
    omega_quat = np.array([0.0, omega[0], omega[1], omega[2]])
    return 0.5 * quat_multiply(q, omega_quat)


def quat_angular_velocity_jacobian(omega: NDArray) -> NDArray:
    """4x4 Jacobian of quaternion derivative w.r.t. quaternion components.

    Returns the Omega matrix such that q_dot = 0.5 * Omega @ q.
    """
    wx, wy, wz = omega
    return 0.5 * np.array([
        [0, -wx, -wy, -wz],
        [wx, 0, wz, -wy],
        [wy, -wz, 0, wx],
        [wz, wy, -wx, 0],
    ])


# ---------------------------------------------------------------------------
# Rotation matrix helpers (Euler-based, used by dynamics module)
# ---------------------------------------------------------------------------

def euler_to_rotation_matrix(phi: float, theta: float, psi: float) -> NDArray:
    """ZYX rotation matrix from Euler angles (roll=phi, pitch=theta, yaw=psi)."""
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi), np.cos(phi)],
    ])
    R_y = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)],
    ])
    R_z = np.array([
        [np.cos(psi), -np.sin(psi), 0],
        [np.sin(psi), np.cos(psi), 0],
        [0, 0, 1],
    ])
    return R_z @ R_y @ R_x


def angular_vel_to_euler_rates(phi: float, theta: float, omega_body: NDArray) -> NDArray:
    """Convert body angular velocity [p,q,r] to Euler angle rates."""
    T = np.array([
        [1, 0, -np.sin(theta)],
        [0, np.cos(phi), np.cos(theta) * np.sin(phi)],
        [0, -np.sin(phi), np.cos(theta) * np.cos(phi)],
    ])
    return np.linalg.solve(T, omega_body)
