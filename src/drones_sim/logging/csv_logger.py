"""CSV telemetry logger."""

from __future__ import annotations

import csv

import numpy as np


class CsvLogger:
    """Append simulation state rows to a CSV file.

    Usage::

        logger = CsvLogger("flight.csv")
        for ...:
            logger.log(t, quad.get_position(), motors=motors)
        logger.close()

    The header is written on the first ``log()`` call so the file is
    self-documenting.
    """

    HEADER = [
        "t", "x", "y", "z", "vx", "vy", "vz",
        "qw", "qx", "qy", "qz", "p", "q", "r",
        "m1", "m2", "m3", "m4",
        "est_x", "est_y", "est_z",
    ]

    def __init__(self, path: str) -> None:
        self._path = path
        self._file = open(path, "w", newline="", encoding="utf-8")  # noqa: SIM115
        self._writer = csv.writer(self._file)
        self._header_written = False

    def log(
        self,
        t: float,
        state: np.ndarray,
        estimate: np.ndarray | None = None,
        motors: np.ndarray | None = None,
    ) -> None:
        """Append one row.

        *state* must be the 13-element dynamics state (see
        ``QuadcopterDynamics.state``).  *estimate* (if given) is a 3-element
        estimated position; *motors* (if given) is a 4-element motor-speed array.
        """
        if not self._header_written:
            self._writer.writerow(self.HEADER)
            self._header_written = True

        qw, qx, qy, qz = state[6:10]
        p, q, r = state[10:13]

        row = [
            t,
            state[0], state[1], state[2],            # pos
            state[3], state[4], state[5],            # vel
            qw, qx, qy, qz,                           # quat
            p, q, r,                                  # omega
            *(motors if motors is not None else [0, 0, 0, 0]),
            *(estimate if estimate is not None else [np.nan, np.nan, np.nan]),
        ]
        self._writer.writerow(row)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()
