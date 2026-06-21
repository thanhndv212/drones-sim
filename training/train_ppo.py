"""SB3 PPO training entry point for quadcopter RL.

Usage::

    python -m training.train_ppo --config configs/ppo_hover.yaml --timesteps 50000
"""

from __future__ import annotations

import argparse
import os

import yaml

from drones_sim.rl.actions import ThrustBodyRatesAction, VelocityLevelAction, LQRResidualAction
from drones_sim.rl.env import QuadcopterEnv
from drones_sim.rl.observations import RelativeStateObs
from drones_sim.rl.reward import RewardConfig, reward
from drones_sim.rl.tasks import HoverTask

_CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


def _make_env(rank: int, cfg: dict, seed: int, action_type: str = "thrust_rates") -> QuadcopterEnv:
    task = HoverTask(target=tuple(cfg.get("target", [0.0, 0.0, 2.0])))
    r_cfg = RewardConfig(**cfg.get("reward", {}))
    if action_type == "velocity":
        action_param = VelocityLevelAction()
    elif action_type == "lqr_residual":
        action_param = LQRResidualAction()
    else:
        action_param = ThrustBodyRatesAction()
    return QuadcopterEnv(
        task=task,
        action_param=action_param,
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
    parser.add_argument("--timesteps", type=int, default=500000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true", help="Render with viser")
    parser.add_argument("--track", action="store_true", help="Log to wandb")
    parser.add_argument("--wandb-project", default="drones-sim-ppo", help="wandb project name")
    parser.add_argument("--action-type", default="thrust_rates",
                        choices=["thrust_rates", "velocity", "lqr_residual"],
                        help="Action parameterization to use")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ppo_cfg = cfg.get("ppo", {})
    env_cfg = cfg.get("env", {})

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    # ── Tracking ──
    callbacks = []
    if args.track:
        import wandb
        from wandb.integration.sb3 import WandbCallback
        run = wandb.init(
            project=args.wandb_project,
            config={
                "policy": ppo_cfg.get("policy", "MlpPolicy"),
                "timesteps": args.timesteps,
                "n_envs": args.n_envs,
                **{f"ppo/{k}": v for k, v in ppo_cfg.items()},
                **{f"env/{k}": v for k, v in env_cfg.items()},
            },
            sync_tensorboard=True,
        )
        callbacks.append(
            WandbCallback(
                gradient_save_freq=0,  # don't save gradients (heavy)
                model_save_path=f"models/{run.id}",
                verbose=0,
            )
        )

    # PPO with MlpPolicy is CPU-bound: tiny network + CPU env stepping.
    # GPU (CUDA/MPS) transfer overhead outweighs the negligible compute.
    device = "cpu"
    print(f"Using device: {device}")

    print(f"Creating {args.n_envs} vectorised environments (action={args.action_type})...")
    venv = SubprocVecEnv([
        lambda i=i: _make_env(i, env_cfg, args.seed, args.action_type) for i in range(args.n_envs)
    ])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)

    os.makedirs(_CHECKPOINT_DIR, exist_ok=True)

    model = PPO(
        device=device,
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

    callbacks.append(
        CheckpointCallback(
            save_freq=ppo_cfg.get("save_freq", 10000),
            save_path=_CHECKPOINT_DIR,
            name_prefix="ppo_quad",
        ),
    )

    print(f"Training for {args.timesteps:,} timesteps...")
    model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=True)

    model_path = os.path.join(_CHECKPOINT_DIR, "final")
    model.save(model_path)
    venv.save(os.path.join(_CHECKPOINT_DIR, "vecnormalize.pkl"))
    print(f"Saved model to {model_path}")
    print(f"Saved VecNormalize stats to {_CHECKPOINT_DIR}/vecnormalize.pkl")


if __name__ == "__main__":
    main()
