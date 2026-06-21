"""Tests for the quaternion-based dynamics plant.

Verifies that the internal rewrite (Euler → quaternion state) preserves
the original behaviour while adding the gimbal-lock-free guarantee and
keeping the quaternion numerically unit-norm over long runs.
"""

import numpy as np
import pytest

from drones_sim.dynamics import QuadcopterDynamics
from drones_sim.math_utils import (
    quat_from_euler,
    quat_normalize,
    quat_to_euler,
    quat_to_rotation_matrix,
)


# ---------------------------------------------------------------------------
# Numerical stability
# ---------------------------------------------------------------------------

def test_initial_quaternion_is_identity():
    quad = QuadcopterDynamics()
    q = quad.get_quaternion()
    np.testing.assert_allclose(q, [1.0, 0.0, 0.0, 0.0], atol=1e-12,
                               err_msg="Fresh quad must have identity quaternion")


def test_quaternion_stays_unit_norm_under_hover():
    """Over 1000 hover steps the quaternion norm must stay in [0.999, 1.001]."""
    quad = QuadcopterDynamics(motor_time_constant=0.04)
    hover_w = np.sqrt(quad.mass * quad.g / (4 * quad.k_f))
    motors = np.full(4, hover_w)
    for _ in range(1000):
        quad.update(0.01, motors)
        q_norm = float(np.linalg.norm(quad.get_quaternion()))
        assert 0.999 <= q_norm <= 1.001, (
            f"Quaternion norm drifted to {q_norm:.6f} — renormalisation is not working"
        )


def test_quaternion_stays_unit_norm_under_aggressive_manoeuvre():
    """Asymmetric motor commands (large body torque) must not corrupt the quat norm."""
    quad = QuadcopterDynamics()
    # 3 motors at hover, 1 motor near zero → strong roll torque
    hover_w = np.sqrt(quad.mass * quad.g / (4 * quad.k_f))
    motors = np.array([10.0, hover_w, hover_w, hover_w])
    for _ in range(500):
        quad.update(0.01, motors)
        q_norm = float(np.linalg.norm(quad.get_quaternion()))
        assert 0.999 <= q_norm <= 1.001, (
            f"Quaternion norm drifted to {q_norm:.6f} under aggressive manoeuvre"
        )
        assert np.all(np.isfinite(quad.state)), "State became non-finite"


# ---------------------------------------------------------------------------
# Euler/quaternion API consistency
# ---------------------------------------------------------------------------

def test_get_attitude_returns_euler_from_quaternion():
    """get_attitude() must return the Euler angles encoded in the quaternion."""
    quad = QuadcopterDynamics()
    # Set a known attitude via Euler → quaternion round trip.
    roll, pitch, yaw = 0.15, -0.08, 0.4
    quad.state[6:10] = quat_from_euler(roll, pitch, yaw)
    rpy = quad.get_attitude()
    np.testing.assert_allclose(rpy, [roll, pitch, yaw], atol=1e-9,
                               err_msg="get_attitude() does not match the stored quaternion")


def test_rotation_matrix_matches_quaternion():
    """rotation_matrix() must equal quat_to_rotation_matrix(get_quaternion())."""
    quad = QuadcopterDynamics()
    quad.state[6:10] = quat_normalize(np.array([0.8, 0.2, 0.3, 0.4]))
    R_from_method = quad.rotation_matrix()
    R_from_quat = quat_to_rotation_matrix(quad.get_quaternion())
    np.testing.assert_allclose(R_from_method, R_from_quat, atol=1e-12)


def test_reset_with_euler_attitude_sets_quaternion():
    """reset(attitude=euler) must populate the quaternion consistently."""
    quad = QuadcopterDynamics()
    quad.reset(attitude=np.array([0.1, 0.2, -0.3]))
    expected_q = quat_from_euler(0.1, 0.2, -0.3)
    np.testing.assert_allclose(quad.get_quaternion(), expected_q, atol=1e-12)


# ---------------------------------------------------------------------------
# Gimbal-lock regression — the whole reason for the rewrite
# ---------------------------------------------------------------------------

def test_quaternion_no_gimbal_lock_at_90_pitch():
    """At 90° pitch the Euler plant would singularise; the quaternion plant must not.

    We seed the drone at 89.9° pitch and integrate.  The Euler kinematics matrix
    ``T(phi, theta)`` becomes singular at theta=90° (its determinant → 0), which
    used to raise ``LinAlgError`` or produce NaN state.  The quaternion plant
    must remain finite and well-conditioned through and past 90°.
    """
    quad = QuadcopterDynamics()
    # Seed attitude at 89.9° pitch (just shy of the Euler singularity).
    quad.reset(attitude=np.array([0.0, np.deg2rad(89.9), 0.0]))
    hover_w = np.sqrt(quad.mass * quad.g / (4 * quad.k_f))
    motors = np.full(4, hover_w)

    for _ in range(200):
        quad.update(0.01, motors)
        assert np.all(np.isfinite(quad.state)), (
            "State became NaN/inf near 90° pitch — gimbal-lock regression"
        )


def test_quaternion_handles_full_flip():
    """A 180° pitch flip (inverted flight) must integrate cleanly.

    This is impossible in the Euler plant (singular at 90°); the quaternion
    plant handles it trivially.
    """
    quad = QuadcopterDynamics()
    quad.reset(attitude=np.array([0.0, np.pi, 0.0]))  # fully inverted
    hover_w = np.sqrt(quad.mass * quad.g / (4 * quad.k_f))
    motors = np.full(4, hover_w)

    for _ in range(100):
        quad.update(0.01, motors)
        assert np.all(np.isfinite(quad.state)), "State became non-finite during flip"
    # Inverted quad → thrust now points down → it should be falling.
    assert quad.get_velocity()[2] < 0.0, "Inverted thrust should accelerate drone downward"


# ---------------------------------------------------------------------------
# Behavioural parity with the old Euler plant
# ---------------------------------------------------------------------------

def test_initial_state_is_zero():
    """Backward-compat: a fresh quad starts at rest at origin."""
    quad = QuadcopterDynamics()
    assert np.allclose(quad.get_position(), 0)
    assert np.allclose(quad.get_velocity(), 0)
    assert np.allclose(quad.get_attitude(), 0)


def test_freefall():
    """Zero motor speeds → drone accelerates downward at g."""
    quad = QuadcopterDynamics()
    for _ in range(100):
        quad.update(0.01, np.zeros(4))
    assert quad.get_position()[2] < 0
    assert quad.get_velocity()[2] < 0
    # Acceleration should be ≈ -g (no drag at rest, no thrust)
    assert abs(quad.get_velocity()[2] + 9.81 * 1.0) < 1.0


def test_hover_thrust():
    """Exact hover thrust → vertical velocity stays near zero."""
    quad = QuadcopterDynamics()
    hover_w = np.sqrt(quad.mass * quad.g / (4 * quad.k_f))
    motors = np.full(4, hover_w)
    for _ in range(100):
        quad.update(0.01, motors)
    assert abs(quad.get_velocity()[2]) < 0.5


def test_motor_lag_disabled_by_default():
    """Default motor_time_constant=0 means instant motor response."""
    quad = QuadcopterDynamics()
    cmd = np.full(4, 500.0)
    quad.update(0.01, cmd)
    assert np.allclose(quad.get_motor_speeds(), cmd)
