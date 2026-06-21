"""Evaluate a trained RL policy and return aggregate metrics.

Usage::

    python -m training.eval_policy --path checkpoints/final.zip --episodes 20
"""

from __future__ import annotations

import argparse

import numpy as np

from drones_sim.rl.actions import ThrustBodyRatesAction
from drones_sim.rl.env import QuadcopterEnv
from drones_sim.rl.observations import RelativeStateObs
from drones_sim.rl.reward import RewardConfig, reward
from drones_sim.rl.tasks import HoverTask


def evaluate(
    model_path: str,
    n_episodes: int = 10,
    deterministic: bool = True,
    seed: int = 0,
) -> dict:
    """Load a checkpoint and evaluate over *n_episodes*.

    Returns a dict with keys: pos_rmse, success_rate, crash_rate, mean_reward.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    env = QuadcopterEnv(
        task=HoverTask(target=(0.0, 0.0, 2.0)),
        action_param=ThrustBodyRatesAction(),
        obs_builder=RelativeStateObs(),
        reward_fn=reward,
        reward_cfg=RewardConfig(),
        dt=0.01, episode_len_s=10.0,
    )
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, training=False)

    import os

    stats_path = os.path.join(os.path.dirname(model_path), "vecnormalize.pkl")
    if os.path.exists(stats_path):
        # SB3 VecNormalize saves stats as a .pkl; load with the same class
        _vn = VecNormalize.load(stats_path, vec_env)

    model = PPO.load(model_path, env=vec_env)

    errors = []
    successes = []
    crashes = []
    total_reward = 0.0

    for ep in range(n_episodes):
        obs = vec_env.reset()
        ep_reward = 0.0
        crashed = False
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, r, dones, infos = vec_env.step(action)
            ep_reward += float(r[0])
            if dones[0]:
                crashed = True
                done = True
            # Also check env state
            pos_err = float(np.linalg.norm(
                env.task.target_pos(env.quad) - env.quad.get_position()
            ))
            errors.append(pos_err)
            if env._step_idx >= env.max_steps:
                done = True
            if crashed:
                break

        total_reward += ep_reward
        # Success: no crash and final error < 0.2 m
        final_err = float(np.linalg.norm(
            env.task.target_pos(env.quad) - env.quad.get_position()
        ))
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
    args = parser.parse_args()

    results = evaluate(args.path, n_episodes=args.episodes, seed=args.seed)
    for k, v in results.items():
        print(f"{k:>15s}: {v:.4f}")


if __name__ == "__main__":
    main()
