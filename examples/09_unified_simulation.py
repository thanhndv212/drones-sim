#!/usr/bin/env python3
"""Run the reworked model/control/estimation stack through one API."""

import matplotlib.pyplot as plt

from drones_sim import ClosedLoopSimulator, SimulationConfig
from drones_sim.trajectory import generate_circular
from drones_sim.visualization import plot_simulation


def main() -> None:
    reference = generate_circular(
        duration=12.0,
        sample_rate=100,
        radius=1.5,
        angular_vel=0.35,
    )
    simulator = ClosedLoopSimulator(
        config=SimulationConfig(dt=0.01, use_estimator=True, seed=7)
    )
    result = simulator.run(reference)
    for name, value in result.summary().items():
        print(f"{name:34s} {value:.4f}")
    plot_simulation(result, title="Unified nonlinear flight stack")
    plt.show()


if __name__ == "__main__":
    main()
