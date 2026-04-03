#!/usr/bin/env python3
"""Example 06: Full pipeline — EKF-fused 3-D trajectory following.

Extends example 05 from single-altitude hover control to full 3-D trajectory
tracking.  A predefined circular trajectory is used as the reference;
a cascaded PID controller tracks it while an EKF fuses IMU + barometer +
GPS measurements in the background.

Architecture::

    generate_circular      →  QuadcopterController  →  QuadcopterDynamics
         (reference)              (cascaded PID)        (RK4 + motor lag)
                                                               ↓
                                                   IMU/baro/GPS simulation
                                                   (Gauss-Markov bias)
                                                               ↓
                                                   ExtendedKalmanFilter
                                                       (observer)
                                                               ↓
                                                  Plots + viser 3-D viewer
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from drones_sim.dynamics import QuadcopterDynamics
from drones_sim.control import QuadcopterController
from drones_sim.sensors.models import SensorNoiseModel
from drones_sim.sensors import GPSSimulator, GPSConfig
from drones_sim.estimation import ExtendedKalmanFilter
from drones_sim.trajectory import generate_circular
from drones_sim.math_utils import euler_to_rotation_matrix
from drones_sim.visualization.viewer import DroneViewer


def main():
    # ------------------------------------------------------------------ #
    # Predefined reference trajectory  (circular, R=3 m, ω=0.4 rad/s)
    # ------------------------------------------------------------------ #
    dt = 0.01
    traj = generate_circular(
        duration=30.0,
        sample_rate=int(1 / dt),
        radius=3.0,
        angular_vel=0.4,
    )
    n = len(traj.t)
    t = traj.t
    print(f"Trajectory: {n} steps over {t[-1]:.1f} s")

    # ------------------------------------------------------------------ #
    # Dynamics  (RK4 integration + 40 ms rotor lag)
    # ------------------------------------------------------------------ #
    quad = QuadcopterDynamics(motor_time_constant=0.04)
    ctrl = QuadcopterController(quad)

    # ------------------------------------------------------------------ #
    # Sensor models
    # ------------------------------------------------------------------ #
    gravity = np.array([0.0, 0.0, 9.81])
    mag_ref = np.array([25.0, 5.0, -40.0])

    # IMU noise with Gauss-Markov bias random walk
    accel_noise = SensorNoiseModel(
        noise_std=0.05, bias_range=0.1,
        bias_time_constant=20.0, bias_random_walk_std=0.005,
    )
    gyro_noise = SensorNoiseModel(
        noise_std=0.01, bias_range=0.005,
        bias_time_constant=60.0, bias_random_walk_std=0.001,
    )
    mag_noise  = SensorNoiseModel(noise_std=0.5, bias_range=1.0)
    baro_noise = SensorNoiseModel(noise_std=0.02, bias_range=0.05)

    # GPS at 10 Hz with realistic position / velocity noise
    gps = GPSSimulator(
        GPSConfig(position_noise_std=0.5, velocity_noise_std=0.1, update_rate=10.0)
    )

    # ------------------------------------------------------------------ #
    # EKF (runs as an observer alongside the true-state controller)
    # ------------------------------------------------------------------ #
    init_state = np.zeros(10)
    init_state[0:3] = traj.position[0]      # seed position with drone's start
    init_state[6] = 1.0                     # identity quaternion w=1
    ekf = ExtendedKalmanFilter(
        dt=dt, initial_state=init_state, gravity=gravity, mag_ref=mag_ref,
    )
    # Loosen process noise so the filter tracks fast dynamics
    ekf.Q[3:6, 3:6]   = np.eye(3) * 0.02
    ekf.Q[6:10, 6:10] = np.eye(4) * 0.002
    ekf.P[3:6, 3:6]   = np.eye(3) * 0.5
    ekf.R_accel        = np.eye(3) * 0.003   # trust accel closer to noise floor

    # ------------------------------------------------------------------ #
    # Logging arrays
    # ------------------------------------------------------------------ #
    true_pos  = np.zeros((n, 3))
    est_pos   = np.zeros((n, 3))
    ref_pos   = np.zeros((n, 3))
    motor_log = np.zeros((n, 4))
    rotations = np.zeros((n, 3, 3))

    # ------------------------------------------------------------------ #
    # Closed-loop simulation
    # ------------------------------------------------------------------ #
    # Start the drone at the trajectory's initial position so tracking error
    # begins at zero rather than spanning the distance to the first point.
    quad.reset(position=traj.position[0].copy())
    ctrl.reset()
    motors = None

    for i in range(n):
        # True state from dynamics
        pos   = quad.get_position()
        vel   = quad.get_velocity()
        att   = quad.get_attitude()
        omega = quad.get_angular_velocity()
        R_true = euler_to_rotation_matrix(*att)

        # ---- Sensor simulation ----------------------------------------
        # Specific force in body frame (what an IMU measures)
        true_accel_body = R_true.T @ gravity
        if i > 0 and motors is not None:
            T_over_m        = float(np.sum(motors**2) * quad.k_f / quad.mass)
            a_thrust_world  = R_true @ np.array([0.0, 0.0, T_over_m])
            a_drag_world    = -quad.k_d * vel / quad.mass
            a_grav_world    = np.array([0.0, 0.0, -quad.g])
            true_accel_body = R_true.T @ (
                a_thrust_world + a_drag_world + a_grav_world + gravity
            )

        accel_meas = accel_noise.apply(true_accel_body, dt=dt)
        gyro_meas  = gyro_noise.apply(omega, dt=dt)
        mag_meas   = mag_noise.apply(R_true.T @ mag_ref)

        # ---- EKF update -----------------------------------------------
        ekf.predict(gyro_meas, accel_meas)
        ekf.correct_accel(accel_meas)
        ekf.correct_mag(mag_meas)

        z_baro = float(baro_noise.apply(np.array([pos[2]]))[0])
        ekf.correct_altitude(z_baro, r_z=0.0004)   # σ=0.02 m → var≈4e-4

        # GPS at 10 Hz: step() includes dropout check
        if i % 10 == 0:
            gps_pos, gps_vel, gps_valid = gps.step(pos, vel)
            if gps_valid:
                ekf.correct_position(gps_pos, R_pos=np.eye(3) * 0.25)   # σ=0.5 m
                ekf.correct_velocity(gps_vel, R_vel=np.eye(3) * 0.01)   # σ=0.1 m/s

        est = ekf.get_state()

        # ---- Trajectory tracking via cascaded PID ---------------------
        # The controller reads true state from quad internally; the EKF runs
        # as a parallel state observer (as in a real embedded flight stack
        # where sensor fusion outputs feed the control law).
        target      = traj.position[i]
        prev_target = traj.position[max(0, i - 1)]
        motors      = ctrl.compute(target, 0.0, dt, prev_target)

        # ---- Log ------------------------------------------------------
        true_pos[i]  = pos
        est_pos[i]   = est["position"]
        ref_pos[i]   = target
        motor_log[i] = motors
        rotations[i] = R_true

        # ---- Dynamics step --------------------------------------------
        quad.update(dt, motors)

        # ---- Progress print every 5 s ---------------------------------
        if i % 500 == 0:
            track_err = np.linalg.norm(pos - target)
            ekf_err   = np.linalg.norm(est["position"] - pos)
            print(
                f"t={t[i]:5.1f}s  "
                f"pos=[{pos[0]:+.2f} {pos[1]:+.2f} {pos[2]:+.2f}]  "
                f"track_err={track_err:.3f} m  ekf_err={ekf_err:.4f} m"
            )

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #
    tracking_err = np.linalg.norm(true_pos - ref_pos, axis=1)
    ekf_err_vec  = np.linalg.norm(true_pos - est_pos, axis=1)

    print(f"\nMean tracking error : {tracking_err.mean():.4f} m")
    print(f"Max  tracking error : {tracking_err.max():.4f} m")
    print(f"Mean EKF error      : {ekf_err_vec.mean():.4f} m")
    print(f"Max  EKF error      : {ekf_err_vec.max():.4f} m")

    # ------------------------------------------------------------------ #
    # Plots
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Panel 1 — XYZ position tracking
    labels = ["X", "Y", "Z"]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for k, (lbl, col) in enumerate(zip(labels, colors)):
        axes[0, 0].plot(t, ref_pos[:, k],  "--", color=col, alpha=0.5, lw=1.2,
                        label=f"Ref {lbl}")
        axes[0, 0].plot(t, true_pos[:, k], "-",  color=col, lw=1.0,
                        label=f"True {lbl}")
    axes[0, 0].set_ylabel("Position (m)")
    axes[0, 0].set_title("Trajectory Tracking — XYZ")
    axes[0, 0].legend(ncol=2, fontsize=7)
    axes[0, 0].grid(True)

    # Panel 2 — Tracking & EKF errors
    axes[0, 1].plot(t, tracking_err, "b-",  lw=1.2, label="Tracking ‖true − ref‖")
    axes[0, 1].plot(t, ekf_err_vec,  "r--", lw=1.0, label="EKF     ‖est  − true‖")
    axes[0, 1].set_ylabel("Error (m)")
    axes[0, 1].set_title("Position Errors vs Time")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Panel 3 — XY plane (top-down path)
    axes[1, 0].plot(ref_pos[:, 0],  ref_pos[:, 1],  "k--", lw=1.2, alpha=0.5,
                    label="Reference")
    axes[1, 0].plot(true_pos[:, 0], true_pos[:, 1], "b-",  lw=1.0,
                    label="True")
    axes[1, 0].plot(
        est_pos[:, 0], est_pos[:, 1], "r:", lw=0.8, label="EKF estimate"
    )
    axes[1, 0].set_xlabel("X (m)")
    axes[1, 0].set_ylabel("Y (m)")
    axes[1, 0].set_title("XY Path (top-down)")
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    axes[1, 0].set_aspect("equal")

    # Panel 4 — Altitude profile
    axes[1, 1].plot(t, ref_pos[:, 2],  "k--", alpha=0.5, label="Reference Z")
    axes[1, 1].plot(t, true_pos[:, 2], "b-",  lw=1.0,   label="True Z")
    axes[1, 1].plot(t, est_pos[:, 2],  "r:",  lw=0.8,   label="EKF Z")
    axes[1, 1].set_ylabel("Z (m)")
    axes[1, 1].set_title("Altitude Profile")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    for ax in axes.flat:
        ax.set_xlabel("Time (s)")
    axes[1, 0].set_xlabel("X (m)")   # override XY plot axis label

    fig.suptitle("Example 06: Full Pipeline — 3-D Trajectory Following")
    fig.tight_layout()
    plt.show()

    # ------------------------------------------------------------------ #
    # 3-D Viewer
    # ------------------------------------------------------------------ #
    viewer = DroneViewer(port=8081)
    viewer.playback(
        t,
        true_pos,
        rotations,
        filtered_positions=est_pos,
        reference_positions=ref_pos,
    )


if __name__ == "__main__":
    main()
