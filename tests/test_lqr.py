"""Tests for the LQR controller."""

import numpy as np
import pytest

from drones_sim.control import LQRController
from drones_sim.dynamics import QuadcopterDynamics


def _simulate_lqr(target: np.ndarray, duration: float, dt: float = 0.01):
    quad = QuadcopterDynamics()
    quad.reset(position=np.array([0.0, 0.0, 0.0]))
    lqr = LQRController(quad)

    positions = []
    for _ in range(int(duration / dt)):
        motors = lqr.compute(target, 0.0, dt)
        quad.update(dt, motors)
        positions.append(quad.get_position())

    return np.array(positions)


def test_lqr_gain_shape():
    quad = QuadcopterDynamics()
    lqr = LQRController(quad)
    assert lqr.K.shape == (4, 12)


def test_lqr_hover_convergence():
    """LQR should converge to a hover target within 8 s."""
    target = np.array([0.0, 0.0, 1.0])
    positions = _simulate_lqr(target, duration=8.0)
    final_error = np.linalg.norm(positions[-1] - target)
    assert final_error < 0.2, f"LQR did not converge: error={final_error:.3f} m"


def test_lqr_lateral_waypoint():
    """LQR should reach a lateral waypoint within 10 s."""
    target = np.array([1.0, 1.0, 1.5])
    positions = _simulate_lqr(target, duration=10.0)
    min_dist = np.min(np.linalg.norm(positions - target, axis=1))
    assert min_dist < 0.3, f"LQR did not reach waypoint: min_dist={min_dist:.3f} m"


def test_lqr_custom_Q_R():
    """LQR should accept custom Q and R matrices without error."""
    quad = QuadcopterDynamics()
    Q = np.eye(12) * 5.0
    R = np.eye(4) * 0.1
    lqr = LQRController(quad, Q=Q, R=R)
    assert lqr.K.shape == (4, 12)
    # Should produce finite motor speeds
    motors = lqr.compute(np.array([0.0, 0.0, 1.0]), 0.0, 0.01)
    assert np.all(np.isfinite(motors))


def test_lqr_reset_is_noop():
    """reset() should not raise and be callable."""
    quad = QuadcopterDynamics()
    lqr = LQRController(quad)
    lqr.reset()  # should not raise
