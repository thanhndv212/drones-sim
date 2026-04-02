#!/usr/bin/env python3
"""Example 03: Quadcopter waypoint navigation with cascaded PID + minimum-snap trajectory.

A minimum-snap polynomial trajectory is generated through the waypoints so that
the reference fed to the controller is C³-continuous (smooth position, velocity,
acceleration, and jerk) rather than a step-function target jump.
"""

import numpy as np
import matplotlib.pyplot as plt

from drones_sim.dynamics import QuadcopterDynamics
from drones_sim.control import QuadcopterController
from drones_sim.trajectory import generate_minimum_snap
from drones_sim.visualization.plots import plot_quadcopter_results


def main():
    quad = QuadcopterDynamics()
    ctrl = QuadcopterController(quad)

    # Waypoints and their absolute timestamps (shared start/end at origin).
    # An extra copy of the last waypoint is added so the trajectory duration
    # covers the full 20 s simulation window.
    waypoints = [
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
        (0.0, 1.0, 1.5),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),   # hold at final position until t=20 s
    ]
    wp_times = [0.0, 2.0, 5.0, 8.0, 11.0, 14.0, 17.0, 20.0]

    t_max = 20.0
    dt = 0.01
    n = int(t_max / dt)
    t = np.linspace(0, t_max, n)

    # Pre-generate minimum-snap reference trajectory.
    traj = generate_minimum_snap(waypoints, times=wp_times, dt=dt)

    pos_log = np.zeros((n, 3))
    att_log = np.zeros((n, 3))
    target_log = np.zeros((n, 3))
    motor_log = np.zeros((n, 4))

    quad.reset(position=np.array([0.0, 0.0, 0.0]))
    ctrl.reset()
    prev_target = np.zeros(3)

    for i in range(n):
        # Smooth min-snap reference (clamped to trajectory length)
        traj_i = min(i, len(traj.position) - 1)
        target = traj.position[traj_i]

        motors = ctrl.compute(target, 0.0, dt, prev_target)
        quad.update(dt, motors)

        pos_log[i] = quad.get_position()
        att_log[i] = quad.get_attitude()
        target_log[i] = target
        motor_log[i] = motors
        prev_target = target.copy()

    plot_quadcopter_results(
        t,
        pos_log,
        att_log,
        target_log,
        motor_log,
        waypoints[:-1],  # original 7 waypoints (last entry is holding duplicate)
    )
    plt.show()


if __name__ == "__main__":
    main()
