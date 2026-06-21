#!/usr/bin/env python3
"""Example 08: Compare RL policy vs cascaded PID on the same trajectory.

Runs a circular trajectory with both:
  1. The standard cascaded PID controller
  2. A pre-trained RL policy (optimised for trajectory tracking)

Produces a 4-panel matplotlib comparison:
  - Position RMSE
  - Final error
  - Control smoothness
  - Success/failure status

Requires: pip install stable-baselines3
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from drones_sim.control import QuadcopterController
from drones_sim.dynamics import QuadcopterDynamics
from drones_sim.rl.actions import ThrustBodyRatesAction
from drones_sim.rl.env import QuadcopterEnv
from drones_sim.rl.observations import RelativeStateObs
from drones_sim.rl.reward import RewardConfig, reward
from drones_sim.rl.tasks import TrackingTask
from drones_sim.trajectory import generate_circular

DT = 0.01


def _run_pid(traj):
    quad = QuadcopterDynamics(motor_time_constant=0.04)
    ctrl = QuadcopterController(quad)
    quad.reset(position=traj.position[0].copy())
    ctrl.reset()
    errors = []
    prev_target = traj.position[0].copy()
    for i in range(len(traj.t)):
        target = traj.position[i]
        motors = ctrl.compute(target, 0.0, DT, prev_target)
        quad.update(DT, motors)
        errors.append(np.linalg.norm(quad.get_position() - target))
        prev_target = target.copy()
    return errors


def _run_rl(traj, model_path):
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    except ImportError:
        print("stable-baselines3 required.  Install: pip install stable-baselines3")
        return None

    env = QuadcopterEnv(
        task=TrackingTask(traj),
        action_param=ThrustBodyRatesAction(),
        obs_builder=RelativeStateObs(),
        reward_fn=reward,
        reward_cfg=RewardConfig(),
        dt=DT,
        episode_len_s=traj.t[-1],
    )
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, training=False)
    stats_path = os.path.join(os.path.dirname(model_path), "vecnormalize.pkl")
    if os.path.exists(stats_path):
        vec_env.load_running_average(os.path.dirname(model_path))

    model = PPO.load(model_path, env=vec_env)
    obs = vec_env.reset()
    errors = []
    for _ in range(len(traj.t)):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = vec_env.step(action)
        if dones[0]:
            break
        errors.append(np.linalg.norm(
            traj.position[min(env._step_idx, len(traj.position) - 1)] - env.quad.get_position()
        ))
    return errors


def main():
    model_path = os.path.join(
        os.path.dirname(__file__), "..", "training", "checkpoints", "ppo_hover_final.zip"
    )
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Run example 07 first or train a model with training/train_ppo.py")
        sys.exit(1)

    traj = generate_circular(duration=10.0, sample_rate=int(1 / DT), radius=2.0, angular_vel=0.5)

    print("Running PID controller...")
    pid_errors = _run_pid(traj)

    print("Running RL policy...")
    rl_errors = _run_rl(traj, model_path)

    if rl_errors is None:
        return

    pid_rmse = np.sqrt(np.mean(np.square(pid_errors)))
    rl_rmse = np.sqrt(np.mean(np.square(rl_errors)))
    pid_final = pid_errors[-1]
    rl_final = rl_errors[-1]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].bar(["PID", "RL"], [pid_rmse, rl_rmse], color=["steelblue", "darkorange"])
    axes[0, 0].set_ylabel("Position RMSE (m)")
    axes[0, 0].set_title("Tracking RMSE")

    axes[0, 1].bar(["PID", "RL"], [pid_final, rl_final], color=["steelblue", "darkorange"])
    axes[0, 1].set_ylabel("Final Error (m)")
    axes[0, 1].set_title("Final Position Error")

    n = min(len(pid_errors), len(rl_errors))
    t_arr = traj.t[:n]
    axes[1, 0].plot(t_arr, pid_errors[:n], label="PID", color="steelblue")
    axes[1, 0].plot(t_arr, rl_errors[:n], label="RL", color="darkorange")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Error (m)")
    axes[1, 0].set_title("Error vs Time")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].axis("off")
    axes[1, 1].text(0.1, 0.5, f"PID RMSE: {pid_rmse:.4f} m\nRL  RMSE: {rl_rmse:.4f} m\n\n"
                    f"PID final err: {pid_final:.4f} m\nRL  final err: {rl_final:.4f} m",
                    fontsize=12, family="monospace")

    fig.suptitle("Cascaded PID vs RL (PPO hover policy) — Circular Tracking")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
