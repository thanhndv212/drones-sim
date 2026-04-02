"""Tests for GPS sensor simulator and EKF GPS fusion."""

import numpy as np
import pytest

from drones_sim.sensors.gps import GPSSimulator, GPSConfig, GPSData
from drones_sim.trajectory import generate_hover_accel_cruise
from drones_sim.estimation import ExtendedKalmanFilter


@pytest.fixture
def traj():
    return generate_hover_accel_cruise(duration=10.0, sample_rate=100)


# ---------------------------------------------------------------------------
# GPSSimulator unit tests
# ---------------------------------------------------------------------------

def test_gps_simulate_output_shape(traj):
    """simulate() should return GPS epochs at the configured update rate."""
    gps = GPSSimulator(GPSConfig(update_rate=5.0), seed=0)
    data = gps.simulate(traj)

    expected_epochs = int(traj.t[-1] * 5.0) + 1
    assert abs(len(data.t) - expected_epochs) <= 1
    assert data.position.shape == (len(data.t), 3)
    assert data.velocity.shape == (len(data.t), 3)
    assert data.valid.shape  == (len(data.t),)


def test_gps_position_noise_magnitude(traj):
    """Position noise RMS should be close to the configured std."""
    noise_std = 2.0
    gps = GPSSimulator(GPSConfig(position_noise_std=noise_std, update_rate=10.0), seed=42)
    data = gps.simulate(traj)

    # Interpolate true positions to GPS epoch times
    true_pos = np.array([
        traj.position[int(np.argmin(np.abs(traj.t - te)))]
        for te in data.t
    ])
    error_rms = float(np.sqrt(np.mean((data.position - true_pos) ** 2)))
    # Should be within factor 2 of the std
    assert error_rms < noise_std * 2.0


def test_gps_dropout(traj):
    """With 100% dropout probability all epochs should be invalid."""
    gps = GPSSimulator(GPSConfig(dropout_probability=1.0), seed=0)
    data = gps.simulate(traj)
    assert not np.any(data.valid)


def test_gps_no_dropout_by_default(traj):
    """Default config has no dropouts."""
    gps = GPSSimulator(seed=0)
    data = gps.simulate(traj)
    assert np.all(data.valid)


def test_gps_step():
    """step() should return a 3-tuple (pos_meas, vel_meas, valid)."""
    gps = GPSSimulator(seed=7)
    true_pos = np.array([1.0, 2.0, 3.0])
    true_vel = np.array([0.1, 0.0, -0.1])
    pos_m, vel_m, valid = gps.step(true_pos, true_vel)
    assert pos_m.shape == (3,)
    assert vel_m.shape == (3,)
    assert isinstance(valid, (bool, np.bool_))


# ---------------------------------------------------------------------------
# EKF GPS fusion integration test
# ---------------------------------------------------------------------------

def test_ekf_gps_fusion_reduces_position_error(traj):
    """EKF with GPS corrections should have lower final position error than IMU-only."""
    from drones_sim.sensors import IMUSimulator

    imu = IMUSimulator()
    imu_data = imu.simulate(traj)
    gps = GPSSimulator(GPSConfig(position_noise_std=0.5, update_rate=5.0), seed=0)
    gps_data = gps.simulate(traj)

    dt = float(np.mean(np.diff(traj.t)))
    gps_dt = 1.0 / 5.0
    gps_steps = max(1, int(round(gps_dt / dt)))

    # EKF without GPS
    ekf_no_gps = ExtendedKalmanFilter(dt=dt)
    for k in range(len(imu_data.t)):
        ekf_no_gps.predict(imu_data.gyro[k], imu_data.accel[k])
        ekf_no_gps.correct_accel(imu_data.accel[k])
        ekf_no_gps.correct_mag(imu_data.mag[k])
    err_no_gps = np.linalg.norm(
        ekf_no_gps.get_state()["position"] - traj.position[-1]
    )

    # EKF with GPS
    ekf_gps = ExtendedKalmanFilter(dt=dt)
    gps_idx = 0
    for k in range(len(imu_data.t)):
        ekf_gps.predict(imu_data.gyro[k], imu_data.accel[k])
        ekf_gps.correct_accel(imu_data.accel[k])
        ekf_gps.correct_mag(imu_data.mag[k])
        # Apply GPS at lower rate
        if gps_idx < len(gps_data.t) and imu_data.t[k] >= gps_data.t[gps_idx]:
            if gps_data.valid[gps_idx]:
                ekf_gps.correct_position(gps_data.position[gps_idx])
            gps_idx += 1
    err_gps = np.linalg.norm(
        ekf_gps.get_state()["position"] - traj.position[-1]
    )

    assert err_gps < err_no_gps, (
        f"GPS fusion did not help: err_gps={err_gps:.3f} >= err_no_gps={err_no_gps:.3f}"
    )
