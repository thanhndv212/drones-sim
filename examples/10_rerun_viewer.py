#!/usr/bin/env python3
"""Visualize or save a synchronized Rerun flight recording."""

from __future__ import annotations

import argparse

from drones_sim import ClosedLoopSimulator, SimulationConfig
from drones_sim.trajectory import generate_circular
from drones_sim.visualization import visualize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save",
        metavar="PATH",
        help="save an .rrd recording instead of opening the viewer",
    )
    parser.add_argument(
        "--connect",
        metavar="URL",
        help="stream to a running Rerun gRPC server",
    )
    args = parser.parse_args()
    if args.save and args.connect:
        parser.error("--save and --connect are mutually exclusive")

    reference = generate_circular(
        duration=12.0,
        sample_rate=100,
        radius=1.5,
        angular_vel=0.35,
    )
    result = ClosedLoopSimulator(
        config=SimulationConfig(dt=0.01, use_estimator=True, seed=7)
    ).run(reference)

    if args.save:
        output = visualize(result, mode="save", path=args.save)
        print(f"Saved Rerun recording: {output}")
    elif args.connect:
        visualize(result, mode="connect", url=args.connect)
    else:
        visualize(result)


if __name__ == "__main__":
    main()
