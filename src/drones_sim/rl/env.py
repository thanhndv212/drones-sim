"""Gymnasium environment wrapping QuadcopterDynamics.

``QuadcopterEnv`` is the single integration point::

    env = QuadcopterEnv(task=HoverTask(), action_param=ThrustBodyRatesAction(),
                         obs_builder=RelativeStateObs(), reward_fn=reward_hover)
    obs, _ = env.reset()
    for _ in range(1000):
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)

It follows the standard ``gymnasium.Env`` contract and is compatible with
Stable-Baselines3, CleanRL, Tianshou, and RLlib.
"""

from __future__ import annotations

import numpy as np

from drones_sim.dynamics import QuadcopterDynamics

# gymnasium is an optional dependency (rl extra).  Every import site tests
# for availability so the module can be imported for type-checking and
# contract tests even without torch/gym installed.
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    gym = None  # type: ignore[assignment]
    spaces = None  # type: ignore[assignment]

from .actions import ThrustBodyRatesAction
from .observations import RelativeStateObs
from .reward import RewardConfig, reward
from .tasks import HoverTask

_GRAVITY = np.array([0.0, 0.0, 9.81])
_TILT_LIMIT = np.deg2rad(75.0)  # crash threshold
_POS_LIMIT = 50.0               # out-of-bounds


class QuadcopterEnv(gym.Env if gym else object):
    """Gymnasium wrapper around ``QuadcopterDynamics``.

    Parameters
    ----------
    task : Task
        Defines the target position/velocity (HoverTask, WaypointTask, etc.).
    action_param : ActionParameterization
        Converts policy action → motor speeds.
    obs_builder : ObservationBuilder
        Builds the observation vector from the quad state.
    reward_fn : callable
        ``reward(quad, task, action, step_idx, cfg, prev_action) -> float``.
    reward_cfg : RewardConfig
        Weights for the reward function.
    dt : float
        Simulation time-step [s].
    episode_len_s : float
        Maximum episode duration [s].
    render_mode : str or None
        'viser' for live 3D viewer (requires running viser server).
    seed : int or None
        Random seed for reproducibility.
    """

    metadata = {"render_modes": ["human", "viser"]}

    def __init__(
        self,
        task=None,
        action_param=None,
        obs_builder=None,
        reward_fn=None,
        reward_cfg=None,
        dt: float = 0.01,
        episode_len_s: float = 10.0,
        render_mode: str | None = None,
        seed: int | None = None,
    ):
        if gym is None:
            raise ImportError("gymnasium is required for QuadcopterEnv.  "
                              "Install it with: pip install gymnasium")
        super().__init__()
        self.dt = dt
        self.max_steps = int(episode_len_s / dt)
        self.task = task if task is not None else HoverTask()
        self.action_param = action_param if action_param is not None else ThrustBodyRatesAction()
        self.obs_builder = obs_builder if obs_builder is not None else RelativeStateObs()
        self.reward_fn = reward_fn if reward_fn is not None else reward
        self.reward_cfg = reward_cfg if reward_cfg is not None else RewardConfig()
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.obs_builder.dim,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=self.action_param.low, high=self.action_param.high, dtype=np.float32,
        )

        self.quad = QuadcopterDynamics(motor_time_constant=0.04)
        self._step_idx = 0
        self._rng = np.random.default_rng(seed)
        self._prev_action: np.ndarray | None = None

    # ------------------------------------------------------------------
    # gym.Env contract
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        # Seed the global RNG for SensorNoiseModel compatibility
        np.random.seed(seed)
        self.quad.reset()
        if hasattr(self.task, 'reset'):
            self.task.reset(self.quad, rng=self._rng)
        if hasattr(self.obs_builder, '_prev_action'):
            self.obs_builder._prev_action = np.zeros(4, dtype=np.float32)
        self._step_idx = 0
        self._prev_action = None
        # Pre-fill motor lag with hover-speed so the drone doesn't dip
        # below z=0 on the very first step.
        hover_w = np.sqrt(self.quad.mass * self.quad.g / (4 * self.quad.k_f))
        self.quad.motor_states = np.full(4, hover_w)
        obs = self.obs_builder.build(self.quad, self.task, action=None)
        return obs.astype(np.float32), {"t": 0.0}

    def step(self, action):
        motor_speeds = self.action_param.to_motors(self.quad, action)
        self.quad.update(self.dt, motor_speeds)
        self._step_idx += 1

        obs = self.obs_builder.build(self.quad, self.task, action=action)
        r = self.reward_fn(
            self.quad, self.task, action, self._step_idx,
            self.reward_cfg, self._prev_action,
        )
        self._prev_action = action.copy()

        terminated = self._is_crashed()
        truncated = self._step_idx >= self.max_steps
        info = {"t": self._step_idx * self.dt, "motor_speeds": motor_speeds}
        return obs.astype(np.float32), float(r), terminated, truncated, info

    def _is_crashed(self) -> bool:
        pos = self.quad.get_position()
        att = self.quad.get_attitude()
        if pos[2] < 0.0:
            return True
        if abs(att[0]) > _TILT_LIMIT or abs(att[1]) > _TILT_LIMIT:
            return True
        if np.linalg.norm(pos) > _POS_LIMIT:
            return True
        return False

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self):
        if self.render_mode == "viser":
            _render_viser(self)

    def close(self):
        pass


def _render_viser(env):
    """Minimal viser rendering — push current pose to a viser frame."""
    try:
        import viser
    except ImportError:  # pragma: no cover
        return
    server = getattr(env, "_viser_server", None)
    handle = getattr(env, "_viser_handle", None)
    if server is None:
        server = viser.ViserServer(port=8082)
        handle = server.scene.add_frame("/rl_quad")
        env._viser_server = server
        env._viser_handle = handle
    pos = env.quad.get_position()
    from viser import transforms as vtf

    from drones_sim.math_utils import quat_to_rotation_matrix
    wxyz = vtf.SO3.from_matrix(quat_to_rotation_matrix(env.quad.get_quaternion())).wxyz
    handle.wxyz = tuple(wxyz)
    handle.position = tuple(pos)
