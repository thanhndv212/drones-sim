"""Regression tests for the cascaded quadcopter controller."""

import numpy as np
import pytest

from drones_sim.control import QuadcopterController
from drones_sim.dynamics import QuadcopterDynamics
from drones_sim.estimation import ExtendedKalmanFilter
from drones_sim.sensors.models import SensorNoiseModel
from drones_sim.math_utils import euler_to_rotation_matrix


def _simulate_targets(
    targets: list[np.ndarray],
    durations: list[float],
    dt: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    quad = QuadcopterDynamics()
    ctrl = QuadcopterController(quad)
    quad.reset(position=np.array([0.0, 0.0, 0.0]))
    ctrl.reset()

    positions: list[np.ndarray] = []
    errors: list[float] = []
    prev_target = targets[0].copy()

    for target, duration in zip(targets, durations):
        steps = int(duration / dt)
        for _ in range(steps):
            motors = ctrl.compute(target, 0.0, dt, prev_target)
            quad.update(dt, motors)
            pos = quad.get_position()
            positions.append(pos)
            errors.append(np.linalg.norm(pos - target))
            prev_target = target.copy()

    return np.array(positions), np.array(errors)


def test_controller_converges_to_hover_target():
    target = np.array([0.0, 0.0, 1.0])
    positions, errors = _simulate_targets([target], [6.0])

    assert np.linalg.norm(positions[-1] - target) < 0.12
    assert float(np.mean(errors[-100:])) < 0.12


def test_controller_reaches_each_waypoint_in_sequence():
    targets = [
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 1.0]),
        np.array([1.0, 1.0, 1.0]),
        np.array([0.0, 1.0, 1.5]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, 0.0]),
    ]
    durations = [2.0, 3.0, 3.0, 3.0, 3.0, 3.0]

    positions, _ = _simulate_targets(targets, durations)

    for target in targets[:-1]:
        distance = np.min(np.linalg.norm(positions - target, axis=1))
        assert distance < 0.22

    assert np.linalg.norm(positions[-1] - targets[-1]) < 0.6


# ---------------------------------------------------------------------------
# Phase 1 Rec-4: Bounded-error EKF integration gate
# ---------------------------------------------------------------------------

def _identity_state() -> np.ndarray:
    s = np.zeros(10)
    s[6] = 1.0
    return s


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ekf_position_error_bounded(seed):
    """Full closed-loop hover: mean 3-D position estimation error must stay < 2 m.

    Sensor stack: accel + gyro + barometer (every step) + GPS at 10 Hz.
    With GPS σ=0.5 m the theoretical floor is well below 2 m; a regression that
    re-introduces any of the six historical bugs will exceed this threshold.
    """
    rng = np.random.default_rng(seed)
    quad = QuadcopterDynamics()
    dt   = 0.01
    gravity = np.array([0.0, 0.0, 9.81])

    ekf = ExtendedKalmanFilter(dt=dt, initial_state=_identity_state(),
                               gravity=gravity)
    ekf.Q[3:6, 3:6]   = np.eye(3) * 0.02
    ekf.Q[6:10, 6:10]  = np.eye(4) * 0.002
    ekf.P[3:6, 3:6]    = np.eye(3) * 0.5
    ekf.R_accel        = np.eye(3) * 0.003

    accel_noise = SensorNoiseModel(noise_std=0.05, bias_range=0.1)
    gyro_noise  = SensorNoiseModel(noise_std=0.01, bias_range=0.005)
    baro_noise  = SensorNoiseModel(noise_std=0.02, bias_range=0.05)
    gps_noise   = SensorNoiseModel(noise_std=0.5,  bias_range=0.3)

    hover_speed = np.sqrt(quad.g * quad.mass / (4 * quad.k_f))
    motors = np.full(4, hover_speed)

    errors = []
    for i in range(500):   # 5 s at 100 Hz
        pos = quad.get_position()
        R   = euler_to_rotation_matrix(*quad.get_attitude())
        accel_meas = accel_noise.apply(R.T @ gravity)
        gyro_meas  = gyro_noise.apply(quad.get_angular_velocity())

        ekf.predict(gyro_meas, accel_meas)
        ekf.correct_accel(accel_meas)

        z_baro = float(baro_noise.apply(np.array([pos[2]]))[0])
        ekf.correct_altitude(z_baro, r_z=0.0004)

        if i % 10 == 0:
            gps_meas = gps_noise.apply(pos.copy())
            ekf.correct_position(gps_meas, R_pos=np.eye(3) * 0.25)

        errors.append(np.linalg.norm(pos - ekf.get_state()["position"]))
        quad.update(dt, motors)

    mean_err = float(np.mean(errors))
    assert mean_err < 2.0, (
        f"EKF mean position error {mean_err:.2f} m > 2.0 m (seed={seed}) — "
        "estimation diverged; check EKF Jacobians, sign conventions, or sensor models"
    )
