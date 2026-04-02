"""Trajectory generators for drone simulation.

All generators return arrays of shape (N, 3) for position, velocity,
acceleration, and (N, 4) for quaternion orientation [w,x,y,z].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

from .math_utils import quat_from_euler


@dataclass
class TrajectoryData:
    """Container for a generated trajectory."""
    t: NDArray
    position: NDArray
    velocity: NDArray
    acceleration: NDArray
    orientation_quat: NDArray  # (N, 4) wxyz
    angular_velocity: NDArray  # (N, 3)


def generate_hover_accel_cruise(
    duration: float = 10.0,
    sample_rate: int = 100,
    max_vel: float = 5.0,
    hover_time: float = 1.0,
    accel_time: float = 2.0,
    cruise_time: float = 4.0,
    decel_time: float = 2.0,
) -> TrajectoryData:
    """Generate a drone-like trajectory: hover -> accelerate -> cruise -> decelerate -> hover.

    This is the canonical test trajectory used across the sensor-fusion examples.
    """
    t = np.linspace(0, duration, int(duration * sample_rate))
    dt = 1.0 / sample_rate
    n = len(t)

    position = np.zeros((n, 3))
    velocity = np.zeros((n, 3))
    acceleration = np.zeros((n, 3))
    orientation_quat = np.zeros((n, 4))
    orientation_quat[:, 0] = 1.0  # identity

    hover_samples = int(hover_time * sample_rate)
    accel_samples = int(accel_time * sample_rate)
    cruise_samples = int(cruise_time * sample_rate)
    decel_samples = int(decel_time * sample_rate)

    max_accel = max_vel / accel_time

    # --- Acceleration phase ---
    for i in range(hover_samples, hover_samples + accel_samples):
        t_a = (i - hover_samples) * dt
        acceleration[i, 0] = max_accel
        velocity[i, 0] = max_accel * t_a
        position[i, 0] = 0.5 * max_accel * t_a**2

        pitch = -0.2 * min(1.0, t_a / accel_time)
        orientation_quat[i] = quat_from_euler(0, pitch, 0)

    # --- Cruise phase ---
    cruise_start = hover_samples + accel_samples
    pos_at_cruise = position[cruise_start - 1, 0] if cruise_start > 0 else 0.0
    for i in range(cruise_start, cruise_start + cruise_samples):
        t_c = (i - cruise_start) * dt
        velocity[i, 0] = max_vel
        position[i, 0] = pos_at_cruise + max_vel * t_c

        position[i, 2] = 0.2 * np.sin(t_c * 2)
        velocity[i, 2] = 0.4 * np.cos(t_c * 2)
        acceleration[i, 2] = -0.8 * np.sin(t_c * 2)

        roll = 0.1 * np.sin(t_c * 3)
        orientation_quat[i] = quat_from_euler(roll, -0.05, 0)

    # --- Deceleration phase ---
    decel_start = cruise_start + cruise_samples
    pos_at_decel = position[decel_start - 1, 0] if decel_start > 0 else 0.0
    for i in range(decel_start, decel_start + decel_samples):
        t_d = (i - decel_start) * dt
        acceleration[i, 0] = -max_accel
        velocity[i, 0] = max(0, max_vel - max_accel * t_d)
        position[i, 0] = pos_at_decel + max_vel * t_d - 0.5 * max_accel * t_d**2

        pitch = 0.2 * min(1.0, t_d / decel_time)
        orientation_quat[i] = quat_from_euler(0, pitch, 0)

    # --- Final hover ---
    final_start = decel_start + decel_samples
    if final_start < n:
        final_pos = position[final_start - 1].copy()
        for i in range(final_start, n):
            position[i] = final_pos

    # --- Angular velocity from quaternion differences ---
    angular_velocity = np.zeros((n, 3))
    for i in range(1, n):
        q_curr = np.roll(orientation_quat[i], -1)  # to xyzw
        q_prev = np.roll(orientation_quat[i - 1], -1)
        r_diff = R.from_quat(q_curr) * R.from_quat(q_prev).inv()
        angular_velocity[i] = r_diff.as_rotvec() / dt

    return TrajectoryData(
        t=t,
        position=position,
        velocity=velocity,
        acceleration=acceleration,
        orientation_quat=orientation_quat,
        angular_velocity=angular_velocity,
    )


def generate_circular(
    duration: float = 10.0,
    sample_rate: int = 100,
    radius: float = 5.0,
    angular_vel: float = 0.5,
) -> TrajectoryData:
    """Generate circular motion with varying height and orientation."""
    t = np.linspace(0, duration, int(duration * sample_rate))
    dt = 1.0 / sample_rate
    n = len(t)

    position = np.zeros((n, 3))
    velocity = np.zeros((n, 3))
    acceleration = np.zeros((n, 3))
    orientation_quat = np.zeros((n, 4))
    orientation_quat[:, 0] = 1.0

    for i in range(n):
        ti = t[i]
        position[i, 0] = radius * np.cos(angular_vel * ti)
        position[i, 1] = radius * np.sin(angular_vel * ti)
        position[i, 2] = 0.5 * np.sin(0.5 * angular_vel * ti)

        velocity[i, 0] = -radius * angular_vel * np.sin(angular_vel * ti)
        velocity[i, 1] = radius * angular_vel * np.cos(angular_vel * ti)
        velocity[i, 2] = 0.25 * angular_vel * np.cos(0.5 * angular_vel * ti)

        roll = 0.2 * np.sin(ti)
        pitch = 0.1 * np.cos(2 * ti)
        yaw = np.arctan2(velocity[i, 1], velocity[i, 0])
        orientation_quat[i] = quat_from_euler(roll, pitch, yaw)

    # Numerical acceleration
    for i in range(1, n):
        acceleration[i] = (velocity[i] - velocity[i - 1]) / dt

    # Angular velocity from quaternion differences
    angular_velocity = np.zeros((n, 3))
    for i in range(1, n):
        q_curr = np.roll(orientation_quat[i], -1)
        q_prev = np.roll(orientation_quat[i - 1], -1)
        r_diff = R.from_quat(q_curr) * R.from_quat(q_prev).inv()
        angular_velocity[i] = r_diff.as_rotvec() / dt

    return TrajectoryData(
        t=t,
        position=position,
        velocity=velocity,
        acceleration=acceleration,
        orientation_quat=orientation_quat,
        angular_velocity=angular_velocity,
    )


def generate_waypoint_trajectory(
    waypoints: list[tuple[float, float, float]],
    waypoint_times: list[float],
    dt: float = 0.01,
) -> TrajectoryData:
    """Generate a piecewise-linear trajectory through waypoints.

    Used by the quadcopter controller simulation.
    """
    t_max = waypoint_times[-1]
    t = np.arange(0, t_max, dt)
    n = len(t)

    position = np.zeros((n, 3))
    velocity = np.zeros((n, 3))
    acceleration = np.zeros((n, 3))

    wp = np.array(waypoints)
    for i in range(n):
        ti = t[i]
        for j in range(len(waypoint_times) - 1):
            if waypoint_times[j] <= ti < waypoint_times[j + 1]:
                alpha = (ti - waypoint_times[j]) / (waypoint_times[j + 1] - waypoint_times[j])
                idx_next = min(j + 1, len(waypoints) - 1)
                position[i] = wp[j] * (1 - alpha) + wp[idx_next] * alpha
                break

    # Numerical velocity and acceleration
    for i in range(1, n):
        velocity[i] = (position[i] - position[i - 1]) / dt
    for i in range(1, n):
        acceleration[i] = (velocity[i] - velocity[i - 1]) / dt

    orientation_quat = np.zeros((n, 4))
    orientation_quat[:, 0] = 1.0
    angular_velocity = np.zeros((n, 3))

    return TrajectoryData(
        t=t,
        position=position,
        velocity=velocity,
        acceleration=acceleration,
        orientation_quat=orientation_quat,
        angular_velocity=angular_velocity,
    )
