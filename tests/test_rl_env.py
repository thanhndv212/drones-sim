"""Contract tests for the QuadcopterEnv gymnasium environment.

These tests verify the env follows the gymnasium.Env contract without
requiring torch or SB3.  If gymnasium is not installed the tests are
silently skipped.
"""

import importlib.util

import numpy as np
import pytest

_HAS_GYM = importlib.util.find_spec("gymnasium") is not None

if _HAS_GYM:
    from drones_sim.rl.actions import ThrustBodyRatesAction
    from drones_sim.rl.env import QuadcopterEnv
    from drones_sim.rl.observations import RelativeStateObs
    from drones_sim.rl.reward import RewardConfig, reward
    from drones_sim.rl.tasks import HoverTask


def _make_env(**kw):
    kwargs = dict(
        task=HoverTask(target=(0.0, 0.0, 2.0)),
        action_param=ThrustBodyRatesAction(),
        obs_builder=RelativeStateObs(),
        reward_fn=reward,
        reward_cfg=RewardConfig(),
        dt=0.01,
        episode_len_s=2.0,
    )
    kwargs.update(kw)
    return QuadcopterEnv(**kwargs)


@pytest.mark.skipif(not _HAS_GYM, reason="gymnasium not installed")
def test_env_contract():
    """Reset + 100 steps — all shapes, dtypes, and finiteness."""
    env = _make_env()
    obs, info = env.reset(seed=0)

    assert isinstance(obs, np.ndarray)
    assert obs.shape == (17,)
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all()
    assert isinstance(info, dict)

    for _ in range(100):
        action = env.action_space.sample()
        obs, r, term, trunc, info = env.step(action)

        assert isinstance(obs, np.ndarray)
        assert obs.shape == env.observation_space.shape
        assert np.isfinite(obs).all()
        assert np.isfinite(r)
        assert isinstance(term, (bool, np.bool_))
        assert isinstance(trunc, (bool, np.bool_))
        if term or trunc:
            break

    env.close()


@pytest.mark.skipif(not _HAS_GYM, reason="gymnasium not installed")
def test_reward_bounds():
    """Reward stays within O(±20) with near-hover actions."""
    env = _make_env()
    env.reset(seed=0)
    hover_thrust = env.quad.mass * env.quad.g
    rewards = []
    for _ in range(200):
        # ThrustBodyRatesAction: small perturbations around hover
        action = np.array([
            hover_thrust * (1.0 + 0.005 * np.random.randn()),
            np.random.randn() * 0.1,
            np.random.randn() * 0.1,
            np.random.randn() * 0.05,
        ])
        _, r, _, _, _ = env.step(action)
        rewards.append(r)
    assert min(rewards) > -20.0, f"Min reward {min(rewards):.2f} too negative"
    assert max(rewards) < 15.0, f"Max reward {max(rewards):.2f} too large"
    env.close()


@pytest.mark.skipif(not _HAS_GYM, reason="gymnasium not installed")
def test_hover_baseline_no_crash():
    """Trivial hover-thrust policy should survive 2 s without crashing."""
    env = _make_env(episode_len_s=2.0)
    obs, _ = env.reset(seed=0)
    hover_thrust = env.quad.mass * env.quad.g
    for _ in range(200):
        # ThrustBodyRatesAction: [thrust_N, ωx, ωy, ωz]
        action = np.array([hover_thrust, 0.0, 0.0, 0.0])
        obs, r, term, trunc, info = env.step(action)
        assert not term, f"Hover-thrust policy crashed at step {env._step_idx}"
        if trunc:
            break
    env.close()


@pytest.mark.skipif(not _HAS_GYM, reason="gymnasium not installed")
def test_reset_deterministic_with_seed():
    """Two resets with the same seed produce identical first observations."""
    env = _make_env()
    obs1, _ = env.reset(seed=42)
    obs2, _ = env.reset(seed=42)
    np.testing.assert_allclose(obs1, obs2, atol=1e-6)


@pytest.mark.skipif(not _HAS_GYM, reason="gymnasium not installed")
def test_hover_baseline_deterministic():
    """Same seed + same action sequence → same trajectory."""
    env = _make_env()
    env.reset(seed=7)
    action = env.action_space.sample()
    pos1 = []
    for _ in range(50):
        obs, _, term, trunc, _ = env.step(action)
        pos1.append(obs[:3].copy())
        if term or trunc:
            break

    env.reset(seed=7)
    pos2 = []
    for _ in range(50):
        obs, _, term, trunc, _ = env.step(action)
        pos2.append(obs[:3].copy())
        if term or trunc:
            break

    for p1, p2 in zip(pos1, pos2):
        np.testing.assert_allclose(p1, p2, atol=1e-6)
    env.close()
