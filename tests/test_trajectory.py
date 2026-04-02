"""Tests for trajectory generators."""

import numpy as np
import pytest

from drones_sim.trajectory import (
    generate_minimum_snap,
    TrajectoryData,
)


# ---------------------------------------------------------------------------
# Minimum-snap tests
# ---------------------------------------------------------------------------

def test_minimum_snap_two_waypoints():
    """Single-segment: start and end positions must match waypoints exactly."""
    wp = [(0.0, 0.0, 0.0), (2.0, 1.0, 1.5)]
    traj = generate_minimum_snap(wp, times=[0.0, 2.0], dt=0.01)

    assert isinstance(traj, TrajectoryData)
    assert np.allclose(traj.position[0], wp[0], atol=1e-6)
    assert np.allclose(traj.position[-1], wp[1], atol=1e-6)


def test_minimum_snap_multi_waypoints_passes_through():
    """All waypoints must be visited (distance < 5 mm)."""
    wp = [(0, 0, 0), (1, 0, 1), (2, 1, 1), (2, 2, 0)]
    times = [0.0, 1.5, 3.0, 4.5]
    traj = generate_minimum_snap(wp, times=times, dt=0.01)

    for i, (t_k, w) in enumerate(zip(times, wp)):
        idx = int(np.argmin(np.abs(traj.t - t_k)))
        dist = np.linalg.norm(traj.position[idx] - np.array(w))
        assert dist < 5e-3, f"Waypoint {i}: dist={dist:.4f} m exceeds 5 mm"


def test_minimum_snap_zero_velocity_at_endpoints():
    """Velocity at start and end should be near zero."""
    wp = [(0, 0, 0), (3, 0, 1)]
    traj = generate_minimum_snap(wp, times=[0.0, 3.0], dt=0.01)

    assert np.linalg.norm(traj.velocity[0]) < 1e-4, "Start velocity not zero"
    assert np.linalg.norm(traj.velocity[-1]) < 1e-4, "End velocity not zero"


def test_minimum_snap_velocity_smoother_than_waypoint_trajectory():
    """Min-snap should have lower max-velocity-jerk than piecewise-linear."""
    from drones_sim.trajectory import generate_waypoint_trajectory

    wp_tuples = [(0, 0, 0), (1, 0, 1), (2, 1, 1)]
    times = [0.0, 1.5, 3.0]

    snap_traj = generate_minimum_snap(wp_tuples, times=times, dt=0.01)
    lin_traj  = generate_waypoint_trajectory(wp_tuples, times, dt=0.01)

    snap_jerk = float(np.max(np.abs(np.diff(snap_traj.velocity, axis=0))) / 0.01)
    lin_jerk  = float(np.max(np.abs(np.diff(lin_traj.velocity, axis=0))) / 0.01)

    assert snap_jerk < lin_jerk, (
        f"Min-snap jerk {snap_jerk:.2f} should be less than linear jerk {lin_jerk:.2f}"
    )


def test_minimum_snap_auto_time_allocation():
    """Auto time allocation should produce a valid trajectory."""
    wp = [(0, 0, 0), (1, 0, 0), (1, 1, 0)]
    traj = generate_minimum_snap(wp, times=None, dt=0.01)
    assert traj.position.shape[1] == 3
    assert np.all(np.isfinite(traj.position))
    assert np.all(np.isfinite(traj.velocity))


def test_minimum_snap_requires_two_waypoints():
    with pytest.raises(ValueError):
        generate_minimum_snap([(0, 0, 0)])
