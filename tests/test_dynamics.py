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
