"""System tests for the typed model/control/estimation/simulation architecture."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from drones_sim.control import ControlAllocator, GeometricController
from drones_sim.dynamics import QuadcopterConfig, QuadcopterDynamics
from drones_sim.estimation import ExtendedKalmanFilter
from drones_sim.simulation import ClosedLoopSimulator, SimulationConfig
from drones_sim.state import TrajectorySetpoint, VehicleState
from drones_sim.trajectory import generate_circular
from drones_sim.visualization import plot_simulation


def test_vehicle_state_round_trip_and_validation():
    state = VehicleState(
        position=np.array([1.0, 2.0, 3.0]),
        quaternion=np.array([2.0, 0.0, 0.0, 0.0]),
    )
    restored = VehicleState.from_vector(state.as_vector())
    np.testing.assert_allclose(restored.as_vector(), state.as_vector())
    assert np.isclose(np.linalg.norm(restored.quaternion), 1.0)
    with pytest.raises(ValueError, match="position"):
        VehicleState(position=np.zeros(2))


def test_config_rejects_nonphysical_inertia():
    with pytest.raises(ValueError, match="positive definite"):
        QuadcopterConfig(inertia=np.diag([0.01, -0.01, 0.02]))


def test_motor_lag_is_stable_when_step_exceeds_time_constant():
    quad = QuadcopterDynamics(motor_time_constant=0.01)
    quad.update(0.1, np.full(4, 1000.0))
    assert np.all(quad.get_motor_speeds() >= 0.0)
    assert np.all(quad.get_motor_speeds() <= 1000.0)


def test_bounded_allocator_reconstructs_feasible_wrench():
    quad = QuadcopterDynamics()
    allocator = ControlAllocator(quad.allocation_matrix(), max_speed=4000.0)
    desired_speeds = np.array([1200.0, 1300.0, 1250.0, 1100.0])
    desired_wrench = quad.allocation_matrix() @ desired_speeds**2
    result = allocator.allocate(desired_wrench)
    np.testing.assert_allclose(result.achieved_wrench, desired_wrench, rtol=1e-8, atol=1e-8)
    assert not result.saturated


def test_geometric_controller_hover_converges():
    quad = QuadcopterDynamics(motor_time_constant=0.03)
    controller = GeometricController(quad)
    target = TrajectorySetpoint(position=np.array([0.0, 0.0, 1.0]))
    quad.reset()
    for _ in range(600):
        output = controller.compute_output(target, 0.01)
        quad.update(0.01, output.motor_speeds)
    assert np.linalg.norm(quad.get_position() - target.position) < 0.08


def test_ekf_rejects_large_gps_outlier():
    ekf = ExtendedKalmanFilter(dt=0.01, innovation_gate=16.27)
    before = ekf.x.copy()
    accepted = ekf.correct_position(
        np.array([1000.0, -1000.0, 500.0]), R_pos=np.eye(3) * 0.25
    )
    assert not accepted
    np.testing.assert_array_equal(ekf.x, before)
    assert ekf.diagnostics.rejected["position"] == 1


def test_closed_loop_result_and_dashboard():
    trajectory = generate_circular(
        duration=3.0, sample_rate=100, radius=0.5, angular_vel=0.35
    )
    simulator = ClosedLoopSimulator(
        config=SimulationConfig(dt=0.01, use_estimator=False, seed=2)
    )
    result = simulator.run(trajectory)
    assert result.position.shape == trajectory.position.shape
    assert result.summary()["tracking_rmse_m"] < 0.15
    figure = plot_simulation(result)
    assert len(figure.axes) == 5


def test_estimator_in_loop_stays_bounded():
    trajectory = generate_circular(
        duration=6.0, sample_rate=100, radius=0.75, angular_vel=0.35
    )
    simulator = ClosedLoopSimulator(
        config=SimulationConfig(dt=0.01, use_estimator=True, seed=0)
    )
    result = simulator.run(trajectory)
    metrics = result.summary()
    assert metrics["tracking_rmse_m"] < 0.3
    assert metrics["estimation_rmse_m"] < 0.25

    # Re-running one configured simulator is deterministic and resets filter stats.
    repeated = simulator.run(trajectory)
    np.testing.assert_allclose(repeated.position, result.position, atol=1e-12)
