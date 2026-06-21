"""SB3 PPO training entry point for quadcopter RL.

Usage::

    python -m training.train_ppo --config configs/ppo_hover.yaml --timesteps 50000
"""

from __future__ import annotations

import argparse
import os

import yaml

from drones_sim.rl.actions import ThrustBodyRatesAction
from drones_sim.rl.env import QuadcopterEnv
from drones_sim.rl.observations import RelativeStateObs
from drones_sim.rl.reward import RewardConfig, reward
from drones_sim.rl.tasks import HoverTask

_CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


def _make_env(rank: int, cfg: dict, seed: int) -> QuadcopterEnv:
    task = HoverTask(target=tuple(cfg.get("target", [0.0, 0.0, 2.0])))
    r_cfg = RewardConfig(**cfg.get("reward", {}))
    return QuadcopterEnv(
        task=task,
        action_param=ThrustBodyRatesAction(),
        obs_builder=RelativeStateObs(),
        reward_fn=reward,
        reward_cfg=r_cfg,
        dt=cfg.get("dt", 0.01),
        episode_len_s=cfg.get("episode_len_s", 10.0),
        seed=seed + rank,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on quadcopter env")
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true", help="Render with viser")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ppo_cfg = cfg.get("ppo", {})
    env_cfg = cfg.get("env", {})

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    print(f"Creating {args.n_envs} vectorised environments...")
    venv = SubprocVecEnv([
        lambda i=i: _make_env(i, env_cfg, args.seed) for i in range(args.n_envs)
    ])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)

    os.makedirs(_CHECKPOINT_DIR, exist_ok=True)

    model = PPO(
        policy=ppo_cfg.get("policy", "MlpPolicy"),
        env=venv,
        learning_rate=ppo_cfg.get("learning_rate", 3e-4),
        n_steps=ppo_cfg.get("n_steps", 1024),
        batch_size=ppo_cfg.get("batch_size", 64),
        n_epochs=ppo_cfg.get("n_epochs", 10),
        gamma=ppo_cfg.get("gamma", 0.99),
        gae_lambda=ppo_cfg.get("gae_lambda", 0.95),
        clip_range=ppo_cfg.get("clip_range", 0.2),
        ent_coef=ppo_cfg.get("ent_coef", 0.01),
        policy_kwargs=ppo_cfg.get("policy_kwargs", {"net_arch": [128, 128]}),
        tensorboard_log=ppo_cfg.get("tensorboard_log", "./tb/ppo_hover"),
        verbose=1,
    )

    callbacks = [
        CheckpointCallback(
            save_freq=ppo_cfg.get("save_freq", 10000),
            save_path=_CHECKPOINT_DIR,
            name_prefix="ppo_quad",
        ),
    ]

    print(f"Training for {args.timesteps:,} timesteps...")
    model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=True)

    model_path = os.path.join(_CHECKPOINT_DIR, "final")
    model.save(model_path)
    venv.save(os.path.join(_CHECKPOINT_DIR, "vecnormalize.pkl"))
    print(f"Saved model to {model_path}")
    print(f"Saved VecNormalize stats to {_CHECKPOINT_DIR}/vecnormalize.pkl")


if __name__ == "__main__":
    main()
