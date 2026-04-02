#!/usr/bin/env python3
"""Example 01: Basic IMU simulation and EKF sensor fusion.

Generates a hover-accel-cruise trajectory, simulates IMU readings,
and runs the 10-state EKF to fuse accel/gyro/mag data.
"""

import numpy as np
import matplotlib.pyplot as plt

from drones_sim.trajectory import generate_hover_accel_cruise
from drones_sim.sensors import IMUSimulator
from drones_sim.sensors.imu import IMUConfig
from drones_sim.estimation import ExtendedKalmanFilter
from drones_sim.visualization.plots import plot_ekf_results


def main():
    # 1. Generate trajectory
    traj = generate_hover_accel_cruise(duration=10.0, sample_rate=100)

    # 2. Simulate IMU
    imu = IMUSimulator(IMUConfig(), seed=42)
    imu_data = imu.simulate(traj)

    # 3. Run EKF
    dt = traj.t[1] - traj.t[0]
    init_state = np.zeros(10)
    init_state[:3] = traj.position[0]
    init_state[3:6] = traj.velocity[0]
    init_state[6:10] = traj.orientation_quat[0]

    ekf = ExtendedKalmanFilter(dt=dt, initial_state=init_state)

    n = len(traj.t)
    filt_pos = np.zeros((n, 3))
    filt_vel = np.zeros((n, 3))
    filt_quat = np.zeros((n, 4))

    for i in range(n):
        ekf.predict(imu_data.gyro[i])
        ekf.correct_accel(imu_data.accel[i])
        ekf.correct_mag(imu_data.mag[i])

        state = ekf.get_state()
        filt_pos[i] = state["position"]
        filt_vel[i] = state["velocity"]
        filt_quat[i] = state["quaternion"]

    # 4. Plot
    fig = plot_ekf_results(
        traj.t,
        traj.position, traj.velocity, traj.orientation_quat,
        filt_pos, filt_vel, filt_quat,
        accel=imu_data.accel, gyro=imu_data.gyro,
    )
    plt.show()

    # 5. Error metrics
    pos_err = np.linalg.norm(traj.position - filt_pos, axis=1)
    vel_err = np.linalg.norm(traj.velocity - filt_vel, axis=1)
    print(f"Position error: mean={pos_err.mean():.3f}m, max={pos_err.max():.3f}m")
    print(f"Velocity error: mean={vel_err.mean():.3f}m/s, max={vel_err.max():.3f}m/s")


if __name__ == "__main__":
    main()
