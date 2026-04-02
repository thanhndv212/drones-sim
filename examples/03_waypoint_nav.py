#!/usr/bin/env python3
"""Example 03: Quadcopter waypoint navigation with cascaded PID control."""

import numpy as np
import matplotlib.pyplot as plt

from drones_sim.dynamics import QuadcopterDynamics
from drones_sim.control import QuadcopterController
from drones_sim.visualization.plots import plot_quadcopter_results


def main():
    quad = QuadcopterDynamics()
    ctrl = QuadcopterController(quad)

    waypoints = [
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
        (0.0, 1.0, 1.5),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 0.0),
    ]
    wp_times = [0.0, 2.0, 5.0, 8.0, 11.0, 14.0, 17.0, 20.0]

    t_max = 20.0
    dt = 0.01
    n = int(t_max / dt)
    t = np.linspace(0, t_max, n)

    pos_log = np.zeros((n, 3))
    att_log = np.zeros((n, 3))
    target_log = np.zeros((n, 3))
    motor_log = np.zeros((n, 4))

    quad.reset(position=np.array([0.0, 0.0, 0.0]))
    ctrl.reset()
    prev_target = np.array([0.0, 0.0, 0.0])
    wp_arr = np.array(waypoints)

    for i in range(n):
        ti = t[i]
        target = wp_arr[-1].copy()
        for j in range(len(wp_times) - 1):
            if wp_times[j] <= ti < wp_times[j + 1]:
                target = wp_arr[j].copy()
                break

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
        waypoints,
    )
    plt.show()


if __name__ == "__main__":
    main()
