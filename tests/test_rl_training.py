"""Tests for the RL training pipeline (requires torch + SB3)."""

import importlib.util
import os

import numpy as np
import pytest

_HAS_SB3 = importlib.util.find_spec("stable_baselines3") is not None

if _HAS_SB3:
    from drones_sim.rl.actions import ThrustBodyRatesAction
    from drones_sim.rl.env import QuadcopterEnv
    from drones_sim.rl.observations import RelativeStateObs
    from drones_sim.rl.reward import RewardConfig, reward
    from drones_sim.rl.tasks import HoverTask

    CHECKPOINT_DIR = os.path.join(
        os.path.dirname(__file__), "..", "training", "checkpoints"
    )


@pytest.mark.skipif(not _HAS_SB3, reason="stable-baselines3 not installed")
def test_train_ppo_smoke():
    """Smoke-train PPO for 1000 steps — verifies pipeline doesn't crash."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    env = QuadcopterEnv(
        task=HoverTask(target=(0.0, 0.0, 2.0)),
        action_param=ThrustBodyRatesAction(),
        obs_builder=RelativeStateObs(),
        reward_fn=reward,
        reward_cfg=RewardConfig(),
        dt=0.01,
        episode_len_s=5.0,
        seed=0,
    )
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    model = PPO(
        "MlpPolicy", vec_env,
        learning_rate=3e-4, n_steps=256, batch_size=64,
        n_epochs=4, gamma=0.99,
        policy_kwargs={"net_arch": [64, 64]},
        verbose=0,
    )
    model.learn(total_timesteps=1000)

    # Save and reload
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, "ppo_smoke_test")
    model.save(path)
    assert os.path.exists(path + ".zip")

    # Quick deterministic prediction
    obs = vec_env.reset()
    action, _ = model.predict(obs, deterministic=True)
    assert action.shape == (1, 4)
    assert np.isfinite(action).all()

    env.close()


@pytest.mark.skipif(not _HAS_SB3, reason="stable-baselines3 not installed")
def test_eval_policy_runs():
    """eval_policy.evaluate() should return expected metrics dict."""
    import tempfile

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    env = QuadcopterEnv(
        task=HoverTask(target=(0.0, 0.0, 2.0)),
        action_param=ThrustBodyRatesAction(),
        obs_builder=RelativeStateObs(),
        reward_fn=reward,
        reward_cfg=RewardConfig(),
        dt=0.01, episode_len_s=2.0, seed=0,
    )
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # Train a tiny model
    model = PPO("MlpPolicy", vec_env, policy_kwargs={"net_arch": [32, 32]}, verbose=0)
    model.learn(total_timesteps=500)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "test_model")
        model.save(model_path)

        from training.eval_policy import evaluate

        results = evaluate(model_path + ".zip", n_episodes=2, seed=0)
        assert isinstance(results, dict)
        for key in ["pos_rmse", "success_rate", "crash_rate", "mean_reward"]:
            assert key in results, f"Missing key: {key}"
        assert 0.0 <= results["success_rate"] <= 1.0
        assert 0.0 <= results["crash_rate"] <= 1.0

    env.close()
