#!/usr/bin/env python3
"""Example 02: Adaptive EKF with temperature effects.

Uses the AHRS-aided adaptive EKF (9-state) with temperature-dependent
sensor models.
"""

import numpy as np
import matplotlib.pyplot as plt

from drones_sim.trajectory import generate_hover_accel_cruise
from drones_sim.sensors import IMUSimulator
from drones_sim.sensors.imu import IMUConfig
from drones_sim.estimation.ekf import AdaptiveEKF
from drones_sim.visualization.plots import plot_ekf_results


def main():
    # 1. Trajectory
    traj = generate_hover_accel_cruise(duration=10.0, sample_rate=100)

    # 2. IMU with temperature effects
    cfg = IMUConfig(enable_temperature=True)
    imu = IMUSimulator(cfg, seed=42)
    imu_data = imu.simulate(traj)

    # 3. Adaptive EKF
    dt = traj.t[1] - traj.t[0]
    ekf = AdaptiveEKF(dt=dt, init_pos=traj.position[0], init_vel=traj.velocity[0])

    n = len(traj.t)
    filt_pos = np.zeros((n, 3))
    filt_vel = np.zeros((n, 3))
    filt_quat = np.zeros((n, 4))

    g_ref = np.array([0, 0, 9.81])
    for i in range(n):
        accel = imu_data.accel[i]
        gyro = imu_data.gyro[i]
        mag = imu_data.mag[i]

        # Adaptive factors from dynamics
        accel_mag = np.linalg.norm(accel - g_ref)
        gyro_mag = np.linalg.norm(gyro)
        proc_factor = 1.0 + min(4.0, accel_mag / 2.0 + gyro_mag * 5.0)
        meas_factor = 1.0 + min(4.0, accel_mag / 3.0)

        ekf.predict(gyro, accel, mag, adaptive_factor=proc_factor)
        ekf.correct(accel, adaptive_factor=meas_factor)

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


if __name__ == "__main__":
    main()
