"""Tests for EKF state estimation."""

import numpy as np
import pytest

from drones_sim.estimation.ekf import ExtendedKalmanFilter
from drones_sim.math_utils import quat_to_rotation_matrix, quat_normalize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _identity_state() -> np.ndarray:
    s = np.zeros(10)
    s[6] = 1.0
    return s


def _fd_jacobian(ekf: ExtendedKalmanFilter, measurement_fn, eps: float = 1e-6) -> np.ndarray:
    """Finite-difference Jacobian of measurement_fn(ekf) w.r.t. ekf.x."""
    x0 = ekf.x.copy()
    h0 = measurement_fn(ekf)
    H = np.zeros((len(h0), len(x0)))
    for j in range(len(x0)):
        xp, xm = x0.copy(), x0.copy()
        xp[j] += eps
        xm[j] -= eps
        ekf.x = xp
        hp = measurement_fn(ekf)
        ekf.x = xm
        hm = measurement_fn(ekf)
        H[:, j] = (hp - hm) / (2 * eps)
    ekf.x = x0  # restore
    return H


def test_ekf_initial_state():
    ekf = ExtendedKalmanFilter(dt=0.01)
    state = ekf.get_state()
    assert np.allclose(state["position"], 0)
    assert np.allclose(state["velocity"], 0)
    assert np.allclose(state["quaternion"], [1, 0, 0, 0])


def test_ekf_predict_moves_position():
    init = np.zeros(10)
    init[3] = 1.0  # vx = 1 m/s
    init[6] = 1.0  # identity quat

    ekf = ExtendedKalmanFilter(dt=0.1, initial_state=init)
    ekf.predict(np.zeros(3))  # no gyro
    state = ekf.get_state()
    assert state["position"][0] > 0  # moved in x


def test_ekf_custom_initial_state():
    init = np.zeros(10)
    init[:3] = [5, 3, 1]
    init[6] = 1.0
    ekf = ExtendedKalmanFilter(dt=0.01, initial_state=init)
    assert np.allclose(ekf.get_state()["position"], [5, 3, 1])


# ---------------------------------------------------------------------------
# Phase 1 Rec-1: Numerical Jacobian validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("quat", [
    np.array([1.0, 0.0, 0.0, 0.0]),                            # identity (level)
    quat_normalize(np.array([0.9, 0.3, 0.1, 0.0])),            # rolled
    quat_normalize(np.array([0.9, 0.0, 0.3, 0.1])),            # pitched + yawed
    quat_normalize(np.array([0.7, 0.4, 0.4, 0.4])),            # large rotation
])
def test_accel_jacobian_matches_fd(quat):
    """Analytical _accel_jacobian must match finite-difference Jacobian to 1e-5."""
    ekf = ExtendedKalmanFilter(dt=0.01, initial_state=_identity_state())
    ekf.x[6:10] = quat

    def accel_h(e):
        R = quat_to_rotation_matrix(e.x[6:10])
        return R.T @ e.gravity

    H_ana = ekf._accel_jacobian(quat)
    H_fd  = _fd_jacobian(ekf, accel_h)
    np.testing.assert_allclose(H_ana, H_fd, atol=1e-5,
                               err_msg=f"_accel_jacobian wrong at q={quat}")


@pytest.mark.parametrize("quat", [
    np.array([1.0, 0.0, 0.0, 0.0]),
    quat_normalize(np.array([0.8, 0.4, 0.2, 0.1])),
])
def test_mag_jacobian_matches_fd(quat):
    """Analytical _mag_jacobian must match finite-difference Jacobian to 1e-5."""
    ekf = ExtendedKalmanFilter(dt=0.01, initial_state=_identity_state(),
                               mag_ref=np.array([25.0, 5.0, -40.0]))
    ekf.x[6:10] = quat

    def mag_h(e):
        R = quat_to_rotation_matrix(e.x[6:10])
        return R.T @ e.mag_ref

    H_ana = ekf._mag_jacobian(quat)
    H_fd  = _fd_jacobian(ekf, mag_h)
    np.testing.assert_allclose(H_ana, H_fd, atol=1e-5,
                               err_msg=f"_mag_jacobian wrong at q={quat}")


# ---------------------------------------------------------------------------
# Phase 1 Rec-2: At-rest IMU sign convention
# ---------------------------------------------------------------------------

def test_accel_expected_at_rest_identity():
    """Level drone (R=I): EKF expected accel = [0, 0, 9.81] body frame."""
    ekf = ExtendedKalmanFilter(dt=0.01, initial_state=_identity_state())
    R = quat_to_rotation_matrix(ekf.x[6:10])
    np.testing.assert_allclose(R, np.eye(3), atol=1e-9,
                               err_msg="Identity quat must give R=I")
    expected = R.T @ ekf.gravity
    np.testing.assert_allclose(expected, [0.0, 0.0, 9.81], atol=1e-9,
                               err_msg="Wrong sign or convention in accel expected")


def test_accel_expected_90deg_roll():
    """90° roll around x-axis: gravity projects entirely to +y body axis."""
    ekf = ExtendedKalmanFilter(dt=0.01, initial_state=_identity_state())
    # q = [cos(45°), sin(45°), 0, 0]  →  90° roll about body-x
    q90x = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])
    R = quat_to_rotation_matrix(q90x)
    expected = R.T @ ekf.gravity   # should be ≈ [0, ±9.81, 0]
    np.testing.assert_allclose(expected[0], 0.0, atol=1e-5,
                               err_msg="x-component should vanish at 90° roll")
    np.testing.assert_allclose(abs(expected[1]), 9.81, atol=0.01,
                               err_msg="y-component should equal g at 90° roll")
    np.testing.assert_allclose(expected[2], 0.0, atol=0.01,
                               err_msg="z-component should vanish at 90° roll")


# ---------------------------------------------------------------------------
# Phase 2 Rec-3: Filter consistency (NIS test)
# ---------------------------------------------------------------------------

def test_accel_nis_at_rest():
    """Median NIS for accel at rest must be below χ²(3, 99%) ≈ 11.3.

    A wrong Jacobian or wrong expected measurement inflates NIS far above this.
    """
    CHI2_99_3DOF = 11.345
    rng = np.random.default_rng(42)
    ekf = ExtendedKalmanFilter(dt=0.01, initial_state=_identity_state())
    g   = ekf.gravity

    nises = []
    for _ in range(200):
        accel_noisy = g + rng.normal(0, 0.05, 3)
        q = ekf.x[6:10]
        R = quat_to_rotation_matrix(q)
        expected = R.T @ g
        residual = accel_noisy - expected
        H = ekf._accel_jacobian(q)
        S = H @ ekf.P @ H.T + ekf.R_accel
        nis = float(residual @ np.linalg.inv(S) @ residual)
        nises.append(nis)

    median_nis = float(np.median(nises))
    assert median_nis < CHI2_99_3DOF, (
        f"Median NIS={median_nis:.2f} exceeds χ²(3,99%)={CHI2_99_3DOF} — "
        "filter inconsistent (wrong Jacobian or measurement model)"
    )
