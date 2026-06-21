#!/usr/bin/env python3
"""Example 07: Train a PPO policy for quadcopter hover stabilization.

This is the "hello world" of the RL pipeline — it trains for 50k steps
using the default PPO config and evaluates the resulting policy.

Requires: pip install stable-baselines3 tensorboard
"""

import os
import sys

try:
    from stable_baselines3 import PPO
except ImportError:
    print("stable-baselines3 is required.  Install: pip install stable-baselines3")
    sys.exit(1)

import numpy as np

from drones_sim.rl.actions import ThrustBodyRatesAction
from drones_sim.rl.env import QuadcopterEnv
from drones_sim.rl.observations import RelativeStateObs
from drones_sim.rl.reward import RewardConfig, reward
from drones_sim.rl.tasks import HoverTask

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "training", "checkpoints")


def main():
    n_steps = 50000
    print(f"Training PPO for {n_steps:,} steps on hover task...")

    env = QuadcopterEnv(
        task=HoverTask(target=(0.0, 0.0, 2.0)),
        action_param=ThrustBodyRatesAction(),
        obs_builder=RelativeStateObs(),
        reward_fn=reward,
        reward_cfg=RewardConfig(),
        dt=0.01,
        episode_len_s=10.0,
        seed=0,
    )

    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    model = PPO(
        "MlpPolicy", vec_env,
        learning_rate=3e-4, n_steps=1024, batch_size=64,
        n_epochs=10, gamma=0.99, gae_lambda=0.95,
        clip_range=0.2, ent_coef=0.01,
        policy_kwargs={"net_arch": [128, 128]},
        tensorboard_log="./tb/ppo_hover",
        verbose=1,
    )

    callbacks = [
        CheckpointCallback(save_freq=10000, save_path=CHECKPOINT_DIR, name_prefix="ppo_quad"),
    ]

    model.learn(total_timesteps=n_steps, callback=callbacks, progress_bar=True)

    model_path = os.path.join(CHECKPOINT_DIR, "ppo_hover_final")
    model.save(model_path)
    vec_env.save(os.path.join(CHECKPOINT_DIR, "vecnormalize.pkl"))
    print(f"Model saved to {model_path}")

    # Quick evaluation
    print("\nEvaluating...")
    vec_env.set_attr("reward_cfg", RewardConfig())
    obs = vec_env.reset()
    errors = []
    for _ in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, dones, infos = vec_env.step(action)
        pos = env.quad.get_position()
        errors.append(np.linalg.norm(pos - np.array([0, 0, 2])))

    rmse = np.sqrt(np.mean(np.square(errors[-50:])))
    print(f"Final position RMSE (last 50 steps): {rmse:.4f} m")
    print("Done! Run 'tensorboard --logdir tb/' to view training curves.")


if __name__ == "__main__":
    main()
