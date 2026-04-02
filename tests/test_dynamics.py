"""Tests for quadcopter dynamics."""

import numpy as np
import pytest

from drones_sim.dynamics import QuadcopterDynamics


def test_initial_state_is_zero():
    quad = QuadcopterDynamics()
    assert np.allclose(quad.get_position(), 0)
    assert np.allclose(quad.get_velocity(), 0)
    assert np.allclose(quad.get_attitude(), 0)


def test_reset():
    quad = QuadcopterDynamics()
    quad.state = np.ones(12)
    quad.reset(position=np.array([1, 2, 3]))
    assert np.allclose(quad.get_position(), [1, 2, 3])
    assert np.allclose(quad.get_velocity(), 0)


def test_freefall():
    """With zero motor speeds, quad should accelerate downward."""
    quad = QuadcopterDynamics()
    for _ in range(100):
        quad.update(0.01, np.zeros(4))
    assert quad.get_position()[2] < 0  # fell down
    assert quad.get_velocity()[2] < 0


def test_hover_thrust():
    """With exact hover thrust, vertical velocity should stay near zero."""
    quad = QuadcopterDynamics()
    hover_w = np.sqrt(quad.mass * quad.g / (4 * quad.k_f))
    motors = np.full(4, hover_w)
    for _ in range(100):
        quad.update(0.01, motors)
    assert abs(quad.get_velocity()[2]) < 0.5  # roughly hovering


def test_motor_lag_delays_response():
    """With a motor time constant, actual motor speeds should lag the command."""
    tau_m = 0.05  # 50 ms
    dt = 0.01
    quad = QuadcopterDynamics(motor_time_constant=tau_m)
    cmd = np.full(4, 1000.0)

    # After one step the actual speed should be less than the command
    quad.update(dt, cmd)
    actual = quad.get_motor_speeds()
    assert np.all(actual < cmd), "Motor speeds should lag behind command"
    # After enough time (10× tau_m) the lag should have converged to within 1%
    for _ in range(int(10 * tau_m / dt)):
        quad.update(dt, cmd)
    actual_settled = quad.get_motor_speeds()
    assert np.allclose(actual_settled, cmd, rtol=0.01), "Motor speeds should settle to command"


def test_motor_lag_disabled_by_default():
    """Default motor_time_constant=0 means instant motor response."""
    quad = QuadcopterDynamics()
    cmd = np.full(4, 500.0)
    quad.update(0.01, cmd)
    assert np.allclose(quad.get_motor_speeds(), cmd)
