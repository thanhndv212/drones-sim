"""GPS sensor simulator following the IMUSimulator pattern.

Simulates a low-rate GNSS receiver producing noisy 3-D position and
optionally 3-D velocity measurements from a ground-truth trajectory.

Typical usage
-------------
>>> gps = GPSSimulator()
>>> gps_data = gps.simulate(traj)
>>> for k in range(len(gps_data.t)):
...     if k % gps_update_steps == 0:
...         ekf.correct_position(gps_data.position[k])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..trajectory import TrajectoryData


@dataclass
class GPSConfig:
    """Configuration parameters for the GPS sensor model."""
    position_noise_std: float = 1.0      # m  (1-sigma, each axis)
    velocity_noise_std: float = 0.1      # m/s (1-sigma, each axis)
    update_rate: float = 5.0             # Hz — GPS fix rate
    dropout_probability: float = 0.0     # probability of fix dropout per epoch


@dataclass
class GPSData:
    """Container for GPS measurement sequences.

    Attributes
    ----------
    t:
        Time stamps of GPS epochs [s], shape (K,).
    position:
        Noisy 3-D position [m], shape (K, 3).
    velocity:
        Noisy 3-D velocity [m/s], shape (K, 3).
    valid:
        Boolean mask: True when the fix is valid (no dropout), shape (K,).
    """
    t: NDArray
    position: NDArray   # (K, 3)
    velocity: NDArray   # (K, 3)
    valid: NDArray      # (K,)  bool


class GPSSimulator:
    """Simulate a low-rate GPS receiver from a ground-truth trajectory.

    The noise model adds independent Gaussian noise to each position and
    velocity axis at the configured update rate.  An optional dropout
    probability marks epochs as invalid (simulating urban-canyon outages).

    Parameters
    ----------
    config:
        Sensor configuration.  Defaults to ``GPSConfig()``.
    seed:
        Random seed for reproducibility.  ``None`` for non-deterministic.
    """

    def __init__(
        self,
        config: GPSConfig | None = None,
        seed: int | None = None,
    ) -> None:
        self.config = config or GPSConfig()
        self._rng = np.random.default_rng(seed)

    def simulate(self, traj: TrajectoryData) -> GPSData:
        """Generate GPS measurements from a full ground-truth trajectory.

        Epochs are spaced ``1 / update_rate`` seconds apart, aligned with
        the nearest sample in ``traj.t``.

        Parameters
        ----------
        traj:
            Ground-truth trajectory.

        Returns
        -------
        GPSData
            GPS measurement sequences at the GPS update rate.
        """
        cfg = self.config
        dt_gps = 1.0 / cfg.update_rate
        t_max  = float(traj.t[-1])
        t_epochs = np.arange(0.0, t_max + dt_gps * 0.5, dt_gps)

        K = len(t_epochs)
        pos_meas = np.zeros((K, 3))
        vel_meas = np.zeros((K, 3))
        valid    = np.ones(K, dtype=bool)

        for i, te in enumerate(t_epochs):
            # Nearest sample index
            idx = int(np.argmin(np.abs(traj.t - te)))

            # Add Gaussian noise
            pos_noise = self._rng.normal(0.0, cfg.position_noise_std, 3)
            vel_noise = self._rng.normal(0.0, cfg.velocity_noise_std, 3)

            pos_meas[i] = traj.position[idx] + pos_noise
            vel_meas[i] = traj.velocity[idx] + vel_noise

            # Optional dropout
            if cfg.dropout_probability > 0.0:
                valid[i] = self._rng.random() >= cfg.dropout_probability

        return GPSData(
            t=t_epochs,
            position=pos_meas,
            velocity=vel_meas,
            valid=valid,
        )

    def step(
        self,
        true_position: NDArray,
        true_velocity: NDArray,
    ) -> tuple[NDArray, NDArray, bool]:
        """Single-step GPS measurement for closed-loop simulation.

        Parameters
        ----------
        true_position:
            Ground-truth 3-D position [m].
        true_velocity:
            Ground-truth 3-D velocity [m/s].

        Returns
        -------
        pos_meas, vel_meas, is_valid
        """
        cfg = self.config
        pos_noise = self._rng.normal(0.0, cfg.position_noise_std, 3)
        vel_noise = self._rng.normal(0.0, cfg.velocity_noise_std, 3)
        is_valid  = (cfg.dropout_probability == 0.0 or
                     self._rng.random() >= cfg.dropout_probability)
        return (
            true_position + pos_noise,
            true_velocity + vel_noise,
            is_valid,
        )
