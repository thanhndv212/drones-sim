"""Task definitions for the quadcopter RL environment."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class HoverTask:
    """Hover at a fixed target position — the simplest baseline task."""

    def __init__(self, target: tuple[float, float, float] = (0.0, 0.0, 2.0)) -> None:
        self._target = np.array(target, dtype=float)

    def reset(self, quad, rng: np.random.Generator | None = None) -> None:
        pass

    def target_pos(self, _quad) -> NDArray:
        return self._target

    def target_vel(self, _quad) -> NDArray:
        return np.zeros(3)


class WaypointTask:
    """Sequence of waypoints — advance when within reach_radius."""

    def __init__(
        self,
        waypoints: list[tuple[float, float, float]],
        reach_radius: float = 0.2,
    ) -> None:
        self.waypoints = [np.array(w, dtype=float) for w in waypoints]
        self.reach_radius = reach_radius
        self._idx = 0

    def reset(self, quad, rng: np.random.Generator | None = None) -> None:
        self._idx = 0

    def target_pos(self, quad) -> NDArray:
        if self._idx < len(self.waypoints):
            dist = np.linalg.norm(self.waypoints[self._idx] - quad.get_position())
            if dist < self.reach_radius and self._idx < len(self.waypoints) - 1:
                self._idx += 1
            return self.waypoints[self._idx]
        return self.waypoints[-1]

    def target_vel(self, _quad) -> NDArray:
        return np.zeros(3)


class TrackingTask:
    """Track a pre-generated TrajectoryData (circular / minimum-snap).

    The closest-in-time reference position is returned at each step.
    """

    def __init__(self, traj) -> None:
        self.traj = traj
        self._step = 0

    def reset(self, quad, rng: np.random.Generator | None = None) -> None:
        self._step = 0

    def target_pos(self, _quad, advance: bool = True) -> NDArray:
        if advance:
            self._step += 1
        idx = min(self._step, len(self.traj.position) - 1)
        return self.traj.position[idx]

    def target_vel(self, _quad) -> NDArray:
        idx = min(self._step, len(self.traj.position) - 1)
        return self.traj.velocity[idx] if hasattr(self.traj, 'velocity') else np.zeros(3)
