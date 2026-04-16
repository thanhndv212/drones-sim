#!/usr/bin/env python3
"""Example 04: Interactive 3D viewer with viser.

Runs a waypoint navigation simulation and launches a viser web viewer
for interactive 3D playback.
"""

import numpy as np

from drones_sim.dynamics import QuadcopterDynamics
from drones_sim.control import QuadcopterController
from drones_sim.math_utils import euler_to_rotation_matrix
from drones_sim.visualization.viewer import DroneViewer
from drones_sim.models import load_drone_urdf


def main():
    # Load URDF model for 3D display
    urdf_model = load_drone_urdf()
    print(urdf_model)

    # Run simulation
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
    wp_arr = np.array(waypoints)

    t_max = 20.0
    dt = 0.01
    n = int(t_max / dt)
    t = np.linspace(0, t_max, n)

    positions = np.zeros((n, 3))
    rotations = np.zeros((n, 3, 3))
    targets_log = np.zeros((n, 3))

    quad.reset()
    ctrl.reset()
    prev_target = np.zeros(3)

    for i in range(n):
        ti = t[i]
        target = wp_arr[-1].copy()
        for j in range(len(wp_times) - 1):
            if wp_times[j] <= ti < wp_times[j + 1]:
                target = wp_arr[j].copy()
                break

        motors = ctrl.compute(target, 0.0, dt, prev_target)
        quad.update(dt, motors)

        positions[i] = quad.get_position()
        phi, theta, psi = quad.get_attitude()
        rotations[i] = euler_to_rotation_matrix(phi, theta, psi)
        targets_log[i] = target
        prev_target = target.copy()

    # Launch viewer with URDF mesh drone
    viewer = DroneViewer(port=8080)
    viewer.playback(
        t,
        positions,
        rotations,
        waypoints=waypoints,
        reference_positions=targets_log,
        urdf_model=urdf_model,
    )


if __name__ == "__main__":
    main()
