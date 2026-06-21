"""Modular reward function for the quadcopter environment.

Usage::

    cfg = RewardConfig()
    r = reward(quad, task, action, step_idx, cfg, prev_action)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RewardConfig:
    """Weights for the sum-of-terms reward function.

    All terms are negative (cost) except survival and reach bonuses.
    """

    w_pos: float = 1.0       # position tracking (L2)
    w_vel: float = 0.05      # velocity tracking
    w_attitude: float = 0.1  # tilt penalty
    w_action: float = 0.01   # control effort (L2)
    w_action_d: float = 0.005  # control rate (Δu)
    w_alive: float = 0.1     # survival bonus per step
    w_reach: float = 10.0    # one-shot bonus for reaching target
    w_crash: float = -10.0   # one-shot penalty for crashing
    reach_radius: float = 0.1  # m — target-reached threshold


def reward(quad, task, action: np.ndarray, step_idx: int,
           cfg: RewardConfig, prev_action: np.ndarray | None = None) -> float:
    """Compute per-step reward.

    Parameters
    ----------
    quad: QuadcopterDynamics instance.
    task: Task instance with ``target_pos(quad)`` and ``target_vel(quad)``.
    action: The action that was just executed.
    step_idx: Current step (0-based).
    cfg: Reward weight configuration.
    prev_action: Action from the previous step (for Δu penalty).
    """
    pos_err = float(np.linalg.norm(task.target_pos(quad) - quad.get_position()))
    vel = quad.get_velocity()
    att = quad.get_attitude()
    tilt = float(np.sqrt(att[0]**2 + att[1]**2))

    r_pos = -cfg.w_pos * pos_err
    r_vel = -cfg.w_vel * float(np.linalg.norm(vel - task.target_vel(quad)))
    r_att = -cfg.w_attitude * tilt
    r_action = -cfg.w_action * float(np.sum(action**2))

    r_action_d = 0.0
    if prev_action is not None:
        r_action_d = -cfg.w_action_d * float(np.linalg.norm(action - prev_action))

    r_alive = cfg.w_alive

    r = r_pos + r_vel + r_att + r_action + r_action_d + r_alive

    # Terminal bonus — only at the last step
    if pos_err < cfg.reach_radius:
        r += cfg.w_reach

    return float(r)
