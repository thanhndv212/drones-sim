"""Observation builders for the quadcopter RL environment.

Each builder produces a ``dim``-dimensional numpy vector.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class RelativeStateObs:
    """Default observation: position error + velocity + quaternion + body rates.

    17-D vector: [pos_err(3), vel(3), quat(4), omega(3), prev_action(4)].
    All quantities in the world frame where applicable.
    """

    dim = 17

    def __init__(self) -> None:
        self._prev_action = np.zeros(4, dtype=np.float32)

    def build(self, quad, task, action: NDArray | None = None) -> NDArray:
        target = task.target_pos(quad)
        pos_err = target - quad.get_position()
        vel = quad.get_velocity()
        quat = quad.get_quaternion()
        omega = quad.get_angular_velocity()

        if action is not None:
            self._prev_action = action.astype(np.float32)

        return np.concatenate([
            pos_err.astype(np.float32),
            vel.astype(np.float32),
            quat.astype(np.float32),
            omega.astype(np.float32),
            self._prev_action,
        ])
