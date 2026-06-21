"""JSON Lines telemetry logger."""

from __future__ import annotations

import json

import numpy as np


class JsonLogger:
    """Write simulation state as JSON Lines (one JSON object per line).

    Usage identical to ``CsvLogger``::

        logger = JsonLogger("flight.jsonl")
        for ...:
            logger.log(t, quad.get_position(), motors=motors)
        logger.close()
    """

    def __init__(self, path: str) -> None:
        self._file = open(path, "w", encoding="utf-8")  # noqa: SIM115

    def log(
        self,
        t: float,
        state: np.ndarray,
        estimate: np.ndarray | None = None,
        motors: np.ndarray | None = None,
    ) -> None:
        qw, qx, qy, qz = state[6:10].tolist()
        record = {
            "t": float(t),
            "position": [float(v) for v in state[:3]],
            "velocity": [float(v) for v in state[3:6]],
            "quaternion": [float(qw), float(qx), float(qy), float(qz)],
            "angular_velocity": [float(v) for v in state[10:13]],
            "motors": [float(v) for v in motors] if motors is not None else [],
            "estimate": (
                [float(v) for v in estimate] if estimate is not None else None
            ),
        }
        self._file.write(json.dumps(record) + "\n")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()
