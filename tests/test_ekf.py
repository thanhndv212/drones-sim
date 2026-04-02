"""Tests for EKF state estimation."""

import numpy as np
import pytest

from drones_sim.estimation.ekf import ExtendedKalmanFilter


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
