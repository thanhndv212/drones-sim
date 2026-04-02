"""2D matplotlib plots for EKF and quadcopter simulation results."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt


def plot_ekf_results(
    t: NDArray,
    true_pos: NDArray,
    true_vel: NDArray,
    true_quat: NDArray,
    filt_pos: NDArray,
    filt_vel: NDArray,
    filt_quat: NDArray,
    accel: NDArray | None = None,
    gyro: NDArray | None = None,
    mag: NDArray | None = None,
    true_accel_bias: NDArray | None = None,
    filt_accel_bias: NDArray | None = None,
    true_gyro_bias: NDArray | None = None,
    filt_gyro_bias: NDArray | None = None,
) -> plt.Figure:
    """Create a multi-panel figure comparing true vs filtered states."""
    has_bias = true_accel_bias is not None and filt_accel_bias is not None

    nrows = 3 if has_bias else 3
    ncols = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))

    labels = ["X", "Y", "Z"]
    colors = ["blue", "green", "red"]
    styles_true = ["-", "-", "-"]
    styles_filt = ["--", "--", "--"]

    # Position
    ax = axes[0, 0]
    for j in range(3):
        ax.plot(t, true_pos[:, j], colors[j], ls="-", label=f"True {labels[j]}")
        ax.plot(t, filt_pos[:, j], colors[j], ls="--", label=f"Filtered {labels[j]}")
    ax.set_ylabel("Position (m)")
    ax.set_title("Position")
    ax.legend(fontsize=7)
    ax.grid(True)

    # Velocity
    ax = axes[0, 1]
    for j in range(3):
        ax.plot(t, true_vel[:, j], colors[j], ls="-", label=f"True V{labels[j].lower()}")
        ax.plot(t, filt_vel[:, j], colors[j], ls="--", label=f"Filtered V{labels[j].lower()}")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title("Velocity")
    ax.legend(fontsize=7)
    ax.grid(True)

    # Orientation (quaternion)
    ax = axes[1, 0]
    quat_labels = ["qw", "qx", "qy", "qz"]
    quat_colors = ["blue", "green", "red", "cyan"]
    for j in range(4):
        ax.plot(t, true_quat[:, j], quat_colors[j], ls="-", label=f"True {quat_labels[j]}")
        ax.plot(t, filt_quat[:, j], quat_colors[j], ls="--", label=f"Filtered {quat_labels[j]}")
    ax.set_ylabel("Quaternion Components")
    ax.set_title("Orientation (Quaternion)")
    ax.legend(fontsize=6)
    ax.grid(True)

    # Sensor readings (accel)
    ax = axes[1, 1]
    if accel is not None:
        for j in range(3):
            ax.plot(t, accel[:, j], colors[j], label=f"Accel {labels[j]}")
        ax.set_title("Accelerometer Readings")
        ax.set_ylabel("Acceleration (m/s²)")
    ax.legend(fontsize=7)
    ax.grid(True)

    # Gyro or bias
    ax = axes[2, 0]
    if gyro is not None:
        for j in range(3):
            ax.plot(t, gyro[:, j], colors[j], label=f"Gyro {labels[j]}")
        ax.set_title("Gyroscope Readings")
        ax.set_ylabel("Angular Velocity (rad/s)")
    ax.legend(fontsize=7)
    ax.grid(True)

    # 3D trajectory
    ax3d = fig.add_subplot(nrows, ncols, nrows * ncols, projection="3d")
    axes[-1, -1].set_visible(False)
    ax3d.plot(true_pos[:, 0], true_pos[:, 1], true_pos[:, 2], "b-", label="True")
    ax3d.plot(filt_pos[:, 0], filt_pos[:, 1], filt_pos[:, 2], "r--", label="Filtered")
    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    ax3d.set_title("3D Trajectory")
    ax3d.legend(fontsize=7)

    for row in axes:
        for a in row:
            if a.get_visible():
                a.set_xlabel("Time (s)")

    fig.tight_layout()
    return fig


def plot_quadcopter_results(
    t: NDArray,
    position: NDArray,
    attitude: NDArray,
    target_position: NDArray,
    motor_speeds: NDArray,
    waypoints: list[tuple[float, float, float]] | None = None,
) -> plt.Figure:
    """Create a 6-panel figure of quadcopter simulation results."""
    fig = plt.figure(figsize=(16, 12))

    # 3D trajectory
    ax1 = fig.add_subplot(2, 3, 1, projection="3d")
    ax1.plot(position[:, 0], position[:, 1], position[:, 2], "b-", label="Actual")
    ax1.plot(target_position[:, 0], target_position[:, 1], target_position[:, 2], "r--", label="Target")
    if waypoints is not None:
        wp = np.array(waypoints)
        ax1.scatter(wp[:, 0], wp[:, 1], wp[:, 2], c="red", s=100, label="Waypoints")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")
    ax1.set_title("Quadcopter Trajectory")
    ax1.legend(fontsize=7)

    # Position vs time
    ax2 = fig.add_subplot(2, 3, 2)
    for j, (c, l) in enumerate(zip(["r", "g", "b"], ["X", "Y", "Z"])):
        ax2.plot(t, position[:, j], f"{c}-", label=l)
        ax2.plot(t, target_position[:, j], f"{c}--", alpha=0.5)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Position (m)")
    ax2.set_title("Position vs Time")
    ax2.legend()
    ax2.grid(True)

    # Attitude
    ax3 = fig.add_subplot(2, 3, 3)
    for j, (c, l) in enumerate(zip(["r", "g", "b"], ["Roll", "Pitch", "Yaw"])):
        ax3.plot(t, np.rad2deg(attitude[:, j]), f"{c}-", label=l)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Angle (deg)")
    ax3.set_title("Attitude vs Time")
    ax3.legend()
    ax3.grid(True)

    # Motor speeds
    ax4 = fig.add_subplot(2, 3, 4)
    for j in range(4):
        ax4.plot(t, motor_speeds[:, j], label=f"Motor {j + 1}")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Motor Speed")
    ax4.set_title("Motor Speeds vs Time")
    ax4.legend()
    ax4.grid(True)

    # Position error
    ax5 = fig.add_subplot(2, 3, 5)
    err = target_position - position
    for j, (c, l) in enumerate(zip(["r", "g", "b"], ["X", "Y", "Z"])):
        ax5.plot(t, err[:, j], f"{c}-", label=f"{l} Error")
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("Error (m)")
    ax5.set_title("Position Error vs Time")
    ax5.legend()
    ax5.grid(True)

    # Error magnitude
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(t, np.linalg.norm(err, axis=1), "k-")
    ax6.set_xlabel("Time (s)")
    ax6.set_ylabel("Error Magnitude (m)")
    ax6.set_title("Total Position Error Magnitude")
    ax6.grid(True)

    fig.tight_layout()
    return fig
