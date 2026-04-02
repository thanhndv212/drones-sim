#!/usr/bin/env python3
"""Example 05: Full pipeline — EKF sensor fusion + quadcopter control.

This is the key integration example that wires together:
1. Quadcopter dynamics simulation
2. IMU sensor simulation from the dynamics state
3. EKF state estimation from noisy IMU readings
4. Cascaded PID control using EKF-estimated state

This closes the loop that was missing in the original codebase.
"""

import numpy as np
import matplotlib.pyplot as plt

from drones_sim.dynamics import QuadcopterDynamics
from drones_sim.control.pid import PIDController
from drones_sim.sensors.models import SensorNoiseModel
from drones_sim.estimation import ExtendedKalmanFilter
from drones_sim.math_utils import (
    euler_to_rotation_matrix,
    quat_from_euler,
    quat_to_rotation_matrix,
)


def main():
    # --- Setup ---
    quad = QuadcopterDynamics()
    dt = 0.01
    t_max = 10.0
    n = int(t_max / dt)
    t = np.linspace(0, t_max, n)

    gravity = np.array([0.0, 0.0, 9.81])
    mag_ref = np.array([25.0, 5.0, -40.0])

    # Sensor noise models
    accel_noise = SensorNoiseModel(noise_std=0.05, bias_range=0.1)
    gyro_noise = SensorNoiseModel(noise_std=0.01, bias_range=0.005)
    mag_noise = SensorNoiseModel(noise_std=0.5, bias_range=1.0)

    # EKF
    init_state = np.zeros(10)
    init_state[6] = 1.0  # identity quat
    ekf = ExtendedKalmanFilter(dt=dt, initial_state=init_state, gravity=gravity, mag_ref=mag_ref)

    # Simple hover controller (just z-axis for demonstration)
    z_ctrl = PIDController(kp=20, ki=5, kd=10, output_limits=(0, 30), windup_limits=(-15, 15))
    target_z = 2.0

    # Logging
    true_pos = np.zeros((n, 3))
    est_pos = np.zeros((n, 3))
    true_att = np.zeros((n, 3))
    est_quat = np.zeros((n, 4))
    motor_log = np.zeros((n, 4))

    quad.reset()

    for i in range(n):
        # --- True state ---
        pos = quad.get_position()
        vel = quad.get_velocity()
        att = quad.get_attitude()
        omega = quad.get_angular_velocity()
        R_true = euler_to_rotation_matrix(*att)

        # --- Simulate sensors ---
        # Accelerometer: gravity in body + linear accel (simplified)
        true_accel_body = R_true.T @ gravity
        accel_meas = accel_noise.apply(true_accel_body)

        gyro_meas = gyro_noise.apply(omega)
        mag_body = R_true.T @ mag_ref
        mag_meas = mag_noise.apply(mag_body)

        # --- EKF update ---
        ekf.predict(gyro_meas)
        ekf.correct_accel(accel_meas)
        ekf.correct_mag(mag_meas)
        est = ekf.get_state()

        # --- Control using EKF estimate ---
        est_z = est["position"][2]
        thrust_cmd = z_ctrl.update(target_z, est_z, dt)

        # Convert thrust to equal motor speeds
        hover_base = quad.g * quad.mass * quad.k_f
        total_thrust = thrust_cmd + hover_base
        motor_speed = np.sqrt(max(total_thrust / (4 * quad.k_f), 0))
        motors = np.full(4, motor_speed)

        # --- Step dynamics ---
        quad.update(dt, motors)

        # --- Log ---
        true_pos[i] = pos
        est_pos[i] = est["position"]
        true_att[i] = att
        est_quat[i] = est["quaternion"]
        motor_log[i] = motors

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(t, true_pos[:, 2], "b-", label="True Z")
    axes[0, 0].plot(t, est_pos[:, 2], "r--", label="EKF Z")
    axes[0, 0].axhline(target_z, color="g", ls=":", label="Target")
    axes[0, 0].set_ylabel("Z (m)")
    axes[0, 0].set_title("Height Control (EKF in the loop)")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(t, np.linalg.norm(true_pos - est_pos, axis=1), "k-")
    axes[0, 1].set_ylabel("Error (m)")
    axes[0, 1].set_title("Position Estimation Error")
    axes[0, 1].grid(True)

    axes[1, 0].plot(t, true_pos[:, 0], label="X")
    axes[1, 0].plot(t, true_pos[:, 1], label="Y")
    axes[1, 0].set_ylabel("Position (m)")
    axes[1, 0].set_title("XY Position")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].plot(t, motor_log[:, 0], label="Motor speed")
    axes[1, 1].set_ylabel("Motor speed")
    axes[1, 1].set_title("Motor Commands")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    for ax in axes.flat:
        ax.set_xlabel("Time (s)")

    fig.suptitle("Full Pipeline: Dynamics → IMU → EKF → PID Control")
    fig.tight_layout()
    plt.show()

    print(f"Final height: {true_pos[-1, 2]:.3f}m (target: {target_z}m)")
    print(f"Mean estimation error: {np.linalg.norm(true_pos - est_pos, axis=1).mean():.4f}m")


if __name__ == "__main__":
    main()
