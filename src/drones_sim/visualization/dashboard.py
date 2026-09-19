"""Result-native diagnostic figures for closed-loop simulations."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ..math_utils import quat_to_euler
from ..simulation import SimulationResult


def plot_simulation(result: SimulationResult, *, title: str | None = None):
    """Build a compact engineering dashboard from a ``SimulationResult``."""
    time = result.time
    figure = plt.figure(figsize=(15, 9), constrained_layout=True)
    grid = figure.add_gridspec(2, 3)

    axis_path = figure.add_subplot(grid[:, 0], projection="3d")
    axis_path.plot(*result.reference_position.T, "--", color="#2ca02c", label="reference")
    axis_path.plot(*result.position.T, color="#1f77b4", label="truth")
    axis_path.plot(*result.estimated_position.T, color="#ff7f0e", alpha=0.7, label="estimate")
    axis_path.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]", title="Flight path")
    axis_path.legend(loc="best")

    axis_position = figure.add_subplot(grid[0, 1])
    labels = ("x", "y", "z")
    for index, label in enumerate(labels):
        axis_position.plot(time, result.position[:, index], label=f"{label} truth")
        axis_position.plot(time, result.reference_position[:, index], "--", alpha=0.65)
    axis_position.set(title="Position tracking", ylabel="position [m]")
    axis_position.grid(alpha=0.25)
    axis_position.legend(ncol=3, fontsize=8)

    axis_error = figure.add_subplot(grid[0, 2])
    tracking = np.linalg.norm(result.tracking_error, axis=1)
    estimation = np.linalg.norm(result.estimation_error, axis=1)
    axis_error.plot(time, tracking, label="tracking")
    axis_error.plot(time, estimation, label="estimation")
    axis_error.set(title="Position errors", ylabel="norm [m]")
    axis_error.grid(alpha=0.25)
    axis_error.legend()

    axis_attitude = figure.add_subplot(grid[1, 1])
    euler = np.array([quat_to_euler(value) for value in result.quaternion])
    for index, label in enumerate(("roll", "pitch", "yaw")):
        axis_attitude.plot(time, np.rad2deg(euler[:, index]), label=label)
    axis_attitude.set(title="Attitude", xlabel="time [s]", ylabel="angle [deg]")
    axis_attitude.grid(alpha=0.25)
    axis_attitude.legend(ncol=3, fontsize=8)

    axis_motor = figure.add_subplot(grid[1, 2])
    axis_motor.plot(time, result.motor_speeds)
    if np.any(result.allocation_saturated):
        axis_motor.fill_between(
            time,
            0.0,
            np.max(result.motor_speeds),
            where=result.allocation_saturated,
            color="#d62728",
            alpha=0.12,
            label="allocation limited",
        )
    axis_motor.set(
        title="Actuator commands", xlabel="time [s]", ylabel="rotor speed [rad/s]"
    )
    axis_motor.grid(alpha=0.25)

    metrics = result.summary()
    heading = title or "Drone simulation"
    figure.suptitle(
        f"{heading}  |  tracking RMSE {metrics['tracking_rmse_m']:.3f} m  |  "
        f"estimation RMSE {metrics['estimation_rmse_m']:.3f} m"
    )
    return figure
