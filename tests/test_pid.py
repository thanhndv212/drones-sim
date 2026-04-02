"""Tests for PID controller."""

import numpy as np
import pytest

from drones_sim.control.pid import PIDController


def test_proportional_only():
    pid = PIDController(kp=2.0, ki=0.0, kd=0.0)
    out = pid.update(setpoint=10.0, measurement=6.0, dt=0.01)
    assert np.isclose(out, 8.0)  # 2.0 * 4.0


def test_output_limits():
    pid = PIDController(kp=100.0, ki=0.0, kd=0.0, output_limits=(-5, 5))
    out = pid.update(setpoint=10.0, measurement=0.0, dt=0.01)
    assert out == 5.0


def test_windup_limits():
    pid = PIDController(kp=0.0, ki=10.0, kd=0.0, windup_limits=(-1, 1))
    for _ in range(100):
        pid.update(setpoint=100.0, measurement=0.0, dt=0.01)
    # Integral should be clamped to 1.0
    out = pid.update(setpoint=100.0, measurement=0.0, dt=0.01)
    assert out == 10.0  # ki * clamped_integral = 10 * 1.0


def test_reset():
    pid = PIDController(kp=1.0, ki=1.0, kd=1.0)
    pid.update(setpoint=5.0, measurement=0.0, dt=0.1)
    pid.reset()
    assert pid._prev_error == 0.0
    assert pid._integral == 0.0
