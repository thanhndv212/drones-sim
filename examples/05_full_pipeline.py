#!/usr/bin/env python3
"""Example 05: Full pipeline — EKF sensor fusion + quadcopter control.

This is the key integration example that wires together:
1. Quadcopter dynamics simulation  (RK4 integration + motor first-order lag)
2. IMU sensor simulation with Gauss-Markov bias random walk
3. GPS sensor simulation via GPSSimulator (replaces raw noise model)
4. EKF state estimation fusing IMU, barometer, and GPS
5. Cascaded PID control using EKF-estimated state
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from drones_sim.dynamics import QuadcopterDynamics
from drones_sim.control.pid import PIDController
from drones_sim.sensors.models import SensorNoiseModel
from drones_sim.sensors import GPSSimulator, GPSConfig
from drones_sim.estimation import ExtendedKalmanFilter
from drones_sim.math_utils import (
    euler_to_rotation_matrix,
    quat_from_euler,
    quat_to_rotation_matrix,
)
from drones_sim.visualization.viewer import DroneViewer


def main():
    # --- Runtime guards ---
    # Set CHECK_NIS=1 to enable Normalized Innovation Squared monitoring.
    # Sustained NIS >> 11.3 (χ²(3,99%)) means the filter is diverging.
    _CHECK_NIS = os.getenv("CHECK_NIS", "0") == "1"

    # --- Setup ---
    # motor_time_constant adds first-order rotor lag (τ=40 ms) to the dynamics.
    quad = QuadcopterDynamics(motor_time_constant=0.04)
    dt = 0.01
    t_max = 10.0
    n = int(t_max / dt)
    t = np.linspace(0, t_max, n)

    gravity = np.array([0.0, 0.0, 9.81])
    mag_ref = np.array([25.0, 5.0, -40.0])

    # Sensor noise models (accel/gyro use Gauss-Markov bias random walk)
    accel_noise = SensorNoiseModel(
        noise_std=0.05, bias_range=0.1,
        bias_time_constant=20.0, bias_random_walk_std=0.005,
    )
    gyro_noise = SensorNoiseModel(
        noise_std=0.01, bias_range=0.005,
        bias_time_constant=60.0, bias_random_walk_std=0.001,
    )
    mag_noise = SensorNoiseModel(noise_std=0.5, bias_range=1.0)
    baro_noise = SensorNoiseModel(noise_std=0.02, bias_range=0.05)  # barometer σ≈2 cm
    # GPSSimulator provides a proper low-rate GNSS receiver model at 10 Hz.
    gps = GPSSimulator(GPSConfig(position_noise_std=0.5, velocity_noise_std=0.1, update_rate=10.0))

    # EKF
    init_state = np.zeros(10)
    init_state[6] = 1.0  # identity quat
    ekf = ExtendedKalmanFilter(dt=dt, initial_state=init_state, gravity=gravity, mag_ref=mag_ref)

    # --- EKF noise tuning ---
    # Loosen velocity process noise so filter tracks dynamics quickly
    ekf.Q[3:6, 3:6] = np.eye(3) * 0.02  # was ~1e-5
    ekf.Q[6:10, 6:10] = np.eye(4) * 0.002  # was ~1e-6
    # Initial velocity uncertainty — honest starting point
    ekf.P[3:6, 3:6] = np.eye(3) * 0.5
    # Trust accelerometer closer to its actual noise level (std=0.05 → var≈0.003)
    ekf.R_accel = np.eye(3) * 0.003

    # Simple hover controller (just z-axis for demonstration)
    # Output spans ±8 N around the hover feedforward; no positive-only clamp needed.
    z_ctrl = PIDController(
        kp=6.0,
        ki=2.0,
        kd=4.0,
        output_limits=(-8.0, 8.0),
        windup_limits=(-3.0, 3.0),
    )
    target_z = 2.0

    # Logging
    true_pos = np.zeros((n, 3))
    est_pos = np.zeros((n, 3))
    true_att = np.zeros((n, 3))
    est_quat = np.zeros((n, 4))
    motor_log = np.zeros((n, 4))
    rotations = np.zeros((n, 3, 3))
    targets_log = np.zeros((n, 3))

    quad.reset()
    motors = None  # needed for first-step specific-force computation

    for i in range(n):
        # --- True state ---
        pos = quad.get_position()
        vel = quad.get_velocity()
        att = quad.get_attitude()
        omega = quad.get_angular_velocity()
        R_true = euler_to_rotation_matrix(*att)

        # --- Simulate sensors ---
        # Accelerometer measures specific force = R^T * (a_linear - g_world)
        # which in hover (a_linear ≈ 0) is just R^T * g (the reaction to gravity).
        # We approximate linear accel from the previous dynamics step.
        lin_accel_world = np.array([0.0, 0.0, quad.g]) - np.array(
            [0.0, 0.0, quad.g]
        )
        # Proper: rotate net force / mass back to body frame as specific force
        # F_net/m = accel (world); specific force = R^T * accel - R^T * (-g) = R^T*(accel + g_vec)
        # In body frame: a_body = R^T * accel_world_total (thrust + drag + gravity) / m
        # Simplest correct formula: specific force = R^T @ (a_total_world + [0,0,g])
        # We read velocity to get accel via finite diff, but easiest is to use dynamics output.
        # Use quad state: accel = dv/dt → computed in update(); approximate as (vel-vel_prev)/dt
        true_accel_body = (
            R_true.T @ gravity
        )  # static gravity component (attitude reference)
        # Add linear dynamics contribution so EKF velocity tracking works
        # thrust accel in world frame = R * [0,0,T/m] ; drag = -k_d*v/m
        if i > 0 and motors is not None:
            T_over_m = float(np.sum(motors**2) * quad.k_f / quad.mass)
            a_thrust_world = R_true @ np.array([0.0, 0.0, T_over_m])
            a_drag_world = -quad.k_d * vel / quad.mass
            a_grav_world = np.array([0.0, 0.0, -quad.g])
            a_total_world = a_thrust_world + a_drag_world + a_grav_world
            # specific force = R^T * a_total + g_body  (what IMU actually measures)
            true_accel_body = R_true.T @ (a_total_world + gravity)
        # Pass dt so Gauss-Markov bias random walk is propagated each step.
        accel_meas = accel_noise.apply(true_accel_body, dt=dt)

        gyro_meas = gyro_noise.apply(omega, dt=dt)
        mag_body = R_true.T @ mag_ref
        mag_meas = mag_noise.apply(mag_body)

        # --- EKF update ---
        # Pass accel to predict() so velocity is propagated via IMU pre-integration.
        # (gyro-only predict kept velocity constant → massive position drift)
        ekf.predict(gyro_meas, accel_meas)
        ekf.correct_accel(accel_meas)  # attitude-only correction
        ekf.correct_mag(mag_meas)
        # Barometer: noisy altitude measurement anchors vertical dead-reckoning.
        # Without this, EKF position drifts hundreds of metres from velocity integration.
        z_baro = float(baro_noise.apply(np.array([pos[2]]))[0])
        ekf.correct_altitude(z_baro, r_z=0.0004)  # var = (0.02)^2
        # GPS at 10 Hz: use GPSSimulator.step() — respects update_rate via index guard.
        if i % 10 == 0:
            gps_pos, _, gps_valid = gps.step(pos, vel)
            if gps_valid:
                ekf.correct_position(gps_pos, R_pos=np.eye(3) * 0.25)  # σ=0.5m → var=0.25
        est = ekf.get_state()

        # --- NIS check (opt-in: CHECK_NIS=1 env var) ---
        if _CHECK_NIS:
            _q   = ekf.x[6:10]
            _R   = quat_to_rotation_matrix(_q)
            _res = accel_meas - _R.T @ gravity
            _H   = ekf._accel_jacobian(_q)
            _S   = _H @ ekf.P @ _H.T + ekf.R_accel
            _nis = float(_res @ np.linalg.inv(_S) @ _res)
            if _nis > 34.0:   # 3× χ²(3,99%) to avoid transient false alarms
                print(f"[NIS WARNING] t={t[i]:.2f}s  NIS={_nis:.1f} — filter may be diverging")

        # --- Control using EKF estimate ---
        est_z = est["position"][2]
        thrust_cmd = z_ctrl.update(target_z, est_z, dt)

        # Convert thrust to equal motor speeds
        # Feedforward provides full m*g; PID corrects the residual error around hover.
        # (Old code used quad.g * quad.mass * quad.k_f ≈ 9.81e-6 N — nearly zero!)
        hover_thrust = quad.g * quad.mass  # [N]  ≈ 9.81 N for typical quad
        assert 5.0 < hover_thrust < 50.0, (
            f"hover_thrust={hover_thrust:.4f} N outside 5–50 N — "
            "check quad.mass and quad.g units"
        )
        total_thrust = thrust_cmd + hover_thrust  # [N]  PID output ± feedforward
        motor_speed = np.sqrt(max(total_thrust / (4 * quad.k_f), 0))  # [rad/s]
        assert motor_speed < 10_000.0, (
            f"motor_speed={motor_speed:.1f} rad/s is physically implausible — "
            "check k_f units or total_thrust computation"
        )
        motors = np.full(4, motor_speed)

        # --- Debug every 1 s (100 steps) ---
        if i % 100 == 0:
            est_err_z = pos[2] - est_z
            ctrl_err = target_z - pos[2]
            print(
                f"t={t[i]:5.2f}s | "
                f"true_z={pos[2]:7.4f}  ekf_z={est_z:7.4f}  target={target_z:.2f} | "
                f"ctrl_err={ctrl_err:+7.4f}  est_err={est_err_z:+7.4f} | "
                f"thrust_cmd={thrust_cmd:+6.3f}  motor_ω={motor_speed:8.1f} rad/s"
            )

        # --- Step dynamics ---
        quad.update(dt, motors)

        # --- Log ---
        true_pos[i] = pos
        est_pos[i] = est["position"]
        true_att[i] = att
        est_quat[i] = est["quaternion"]
        motor_log[i] = motors
        rotations[i] = R_true
        targets_log[i] = [0.0, 0.0, target_z]

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

    # --- 3D Viewer ---
    viewer = DroneViewer(port=8080)
    viewer.playback(
        t,
        true_pos,
        rotations,
        filtered_positions=est_pos,
        reference_positions=targets_log,
    )


if __name__ == "__main__":
    main()
