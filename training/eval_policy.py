"""Evaluate a trained RL policy and return aggregate metrics.

Usage::

    python -m training.eval_policy --path checkpoints/final.zip --episodes 20
"""

from __future__ import annotations

import argparse

import numpy as np

from drones_sim.rl.actions import ThrustBodyRatesAction, VelocityLevelAction, LQRResidualAction
from drones_sim.rl.env import QuadcopterEnv
from drones_sim.rl.observations import RelativeStateObs
from drones_sim.rl.reward import RewardConfig, reward
from drones_sim.rl.tasks import HoverTask


def evaluate(
    model_path: str,
    n_episodes: int = 10,
    deterministic: bool = True,
    seed: int = 0,
    action_type: str = "thrust_rates",
) -> dict:
    """Load a checkpoint and evaluate over *n_episodes*.

    Returns a dict with keys: pos_rmse, success_rate, crash_rate, mean_reward.
    """
    import pickle as _pickle

    from stable_baselines3 import PPO

    # PPO with MlpPolicy is CPU-bound (tiny network + CPU env stepping).
    device = "cpu"

    if action_type == "velocity":
        action_param = VelocityLevelAction()
    elif action_type == "lqr_residual":
        action_param = LQRResidualAction()
    else:
        action_param = ThrustBodyRatesAction()

    # Load model and VecNormalize stats
    model = PPO.load(model_path, device=device)
    import os
    stats_path = os.path.join(os.path.dirname(model_path), "vecnormalize.pkl")
    vn_stats = None
    if os.path.exists(stats_path):
        with open(stats_path, "rb") as f:
            vn_stats = _pickle.load(f)

    errors: list[float] = []
    successes: list[bool] = []
    crashes: list[bool] = []
    total_reward = 0.0

    for ep in range(n_episodes):
        env = QuadcopterEnv(
            task=HoverTask(target=(0.0, 0.0, 2.0)),
            action_param=action_param,
            obs_builder=RelativeStateObs(),
            reward_fn=reward,
            reward_cfg=RewardConfig(),
            dt=0.01, episode_len_s=10.0,
        )
        obs, _ = env.reset()
        ep_reward = 0.0
        crashed = False
        final_err = float("inf")

        for _step in range(env.max_steps):
            # Manual observation normalization
            if vn_stats is not None:
                obs_norm = (obs - vn_stats.obs_rms.mean) / np.sqrt(vn_stats.obs_rms.var + 1e-8)
            else:
                obs_norm = obs
            action, _ = model.predict(obs_norm, deterministic=deterministic)
            obs, r, terminated, truncated, info = env.step(action)
            ep_reward += float(r)

            pos_err = float(np.linalg.norm(
                env.task.target_pos(env.quad) - env.quad.get_position()
            ))
            errors.append(pos_err)

            if terminated:
                crashed = True
                break
            if truncated:
                break

        # Final position is the last known position (no auto-reset ambiguity)
        final_err = pos_err
        total_reward += ep_reward
        successes.append(not crashed and final_err < 0.2)
        crashes.append(crashed)

    return {
        "pos_rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "success_rate": float(np.mean(successes)),
        "crash_rate": float(np.mean(crashes)),
        "mean_reward": total_reward / n_episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RL policy")
    parser.add_argument("--path", required=True, help="Path to model .zip file")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--action-type", default="thrust_rates",
                        choices=["thrust_rates", "velocity", "lqr_residual"],
                        help="Action parameterization used during training")
    args = parser.parse_args()

    results = evaluate(args.path, n_episodes=args.episodes, seed=args.seed,
                       action_type=args.action_type)
    for k, v in results.items():
        print(f"{k:>15s}: {v:.4f}")


if __name__ == "__main__":
    main()
