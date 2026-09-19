"""Bounded control allocation from body wrench to rotor speeds."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import lsq_linear


@dataclass
class AllocationResult:
    motor_speeds: NDArray
    achieved_wrench: NDArray
    saturated: bool
    residual_norm: float


class ControlAllocator:
    """Weighted, bounded least-squares rotor allocator.

    Solving in squared-speed space respects the one-sided nature of propeller
    thrust. Unlike inverse-then-clip allocation, it redistributes an infeasible
    request over the remaining authority instead of silently changing all four
    wrench axes.
    """

    def __init__(
        self,
        allocation_matrix: NDArray,
        min_speed: float = 0.0,
        max_speed: float = 4000.0,
        weights: NDArray | None = None,
    ) -> None:
        self.matrix = np.asarray(allocation_matrix, dtype=float).copy()
        if self.matrix.shape != (4, 4):
            raise ValueError("allocation_matrix must have shape (4, 4)")
        self.min_speed = float(min_speed)
        self.max_speed = float(max_speed)
        if not 0.0 <= self.min_speed < self.max_speed:
            raise ValueError("motor speed limits are invalid")
        self.weights = np.asarray(
            np.ones(4) if weights is None else weights, dtype=float
        )
        if self.weights.shape != (4,) or np.any(self.weights <= 0.0):
            raise ValueError("weights must be a positive shape-(4,) vector")

    def allocate(self, wrench: NDArray) -> AllocationResult:
        desired = np.asarray(wrench, dtype=float)
        if desired.shape != (4,) or not np.all(np.isfinite(desired)):
            raise ValueError("wrench must be a finite shape-(4,) vector")
        weighted_matrix = self.weights[:, None] * self.matrix
        weighted_wrench = self.weights * desired
        solution = lsq_linear(
            weighted_matrix,
            weighted_wrench,
            bounds=(self.min_speed**2, self.max_speed**2),
            method="bvls",
            tol=1e-10,
        )
        squared_speeds = np.maximum(solution.x, 0.0)
        motor_speeds = np.sqrt(squared_speeds)
        achieved = self.matrix @ squared_speeds
        tolerance = 1e-7 * max(1.0, self.max_speed**2)
        saturated = bool(
            np.any(squared_speeds <= self.min_speed**2 + tolerance)
            or np.any(squared_speeds >= self.max_speed**2 - tolerance)
            or np.linalg.norm(achieved - desired) > 1e-5
        )
        return AllocationResult(
            motor_speeds=motor_speeds,
            achieved_wrench=achieved,
            saturated=saturated,
            residual_norm=float(np.linalg.norm(achieved - desired)),
        )
