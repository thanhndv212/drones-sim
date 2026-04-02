"""Regression tests for the cascaded quadcopter controller."""

import numpy as np

from drones_sim.control import QuadcopterController
from drones_sim.dynamics import QuadcopterDynamics


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

    assert np.linalg.norm(positions[-1] - targets[-1]) < 0.45
