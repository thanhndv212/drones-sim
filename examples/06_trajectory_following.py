#!/usr/bin/env python3
"""Example 06: Full pipeline — EKF-fused 3-D trajectory following.

Interactive viser GUI:
  \u2022 \U0001f3b2 Random Trajectory \u2014 generate a new random min-snap trajectory (shown in green)
  \u2022 \u25b6 Track             \u2014 run the closed-loop simulation for the current trajectory
  \u2022 Time slider         \u2014 scrub through the last simulation frame-by-frame
  \u2022 \u25b6 Play / \u27f3 Reset   \u2014 animate playback at adjustable speed
  \u2022 Show EKF estimate  \u2014 toggle the orange EKF position cloud

Architecture::

    generate_circular / _make_random_traj  \u2192  QuadcopterController  \u2192  QuadcopterDynamics
             (reference)                         (cascaded PID)        (RK4 + motor lag)
                                                                              \u2193
                                                              IMU / baro / GPS simulation
                                                              (Gauss-Markov bias)
                                                                              \u2193
                                                              ExtendedKalmanFilter
                                                                  (16-state observer)
                                                                              \u2193
                                                             Interactive viser 3-D viewer
"""

from __future__ import annotations

import threading
import time as _time

import numpy as np

from drones_sim.dynamics import QuadcopterDynamics
from drones_sim.control import QuadcopterController
from drones_sim.sensors.models import SensorNoiseModel
from drones_sim.sensors import GPSSimulator, GPSConfig
from drones_sim.estimation import ExtendedKalmanFilter
from drones_sim.trajectory import generate_circular, generate_minimum_snap
from drones_sim.math_utils import euler_to_rotation_matrix
from drones_sim.visualization.viewer import DroneViewer

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
_GRAVITY = np.array([0.0, 0.0, 9.81])
_MAG_REF = np.array([25.0, 5.0, -40.0])
_DT      = 0.01


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def _make_sensors() -> dict:
    """Return a fresh set of sensor noise models (independent per simulation run)."""
    return dict(
        accel=SensorNoiseModel(
            noise_std=0.05, bias_range=0.1,
            bias_time_constant=20.0, bias_random_walk_std=0.005,
        ),
        gyro=SensorNoiseModel(
            noise_std=0.01, bias_range=0.005,
            bias_time_constant=60.0, bias_random_walk_std=0.001,
        ),
        mag=SensorNoiseModel(noise_std=0.5, bias_range=1.0),
        baro=SensorNoiseModel(noise_std=0.02, bias_range=0.05),
        gps=GPSSimulator(
            GPSConfig(position_noise_std=0.5, velocity_noise_std=0.1, update_rate=10.0)
        ),
    )


def _make_ekf(start_pos: np.ndarray) -> ExtendedKalmanFilter:
    """Create a 16-state EKF seeded at *start_pos*."""
    init = np.zeros(10)
    init[0:3] = start_pos
    init[6]   = 1.0
    ekf = ExtendedKalmanFilter(dt=_DT, initial_state=init,
                               gravity=_GRAVITY, mag_ref=_MAG_REF)
    ekf.Q[3:6, 3:6]   = np.eye(3) * 0.02
    ekf.Q[6:10, 6:10] = np.eye(4) * 0.002
    ekf.P[3:6, 3:6]   = np.eye(3) * 0.5
    ekf.R_accel        = np.eye(3) * 0.003
    return ekf


def _run_simulation(traj, quad: QuadcopterDynamics, ctrl: QuadcopterController):
    """Run one closed-loop simulation.

    Returns
    -------
    true_pos  : ndarray (N, 3)
    est_pos   : ndarray (N, 3)
    rotations : ndarray (N, 3, 3)
    t         : ndarray (N,)
    """
    n       = len(traj.t)
    t       = traj.t
    sensors = _make_sensors()
    ekf     = _make_ekf(traj.position[0])

    true_pos  = np.zeros((n, 3))
    est_pos   = np.zeros((n, 3))
    rotations = np.zeros((n, 3, 3))

    quad.reset(position=traj.position[0].copy())
    ctrl.reset()
    motors = None

    for i in range(n):
        pos   = quad.get_position()
        vel   = quad.get_velocity()
        att   = quad.get_attitude()
        omega = quad.get_angular_velocity()
        R     = euler_to_rotation_matrix(*att)

        true_accel_body = R.T @ _GRAVITY
        if i > 0 and motors is not None:
            T_m   = float(np.sum(motors**2) * quad.k_f / quad.mass)
            a_thr = R @ np.array([0.0, 0.0, T_m])
            a_drg = -quad.k_d * vel / quad.mass
            true_accel_body = R.T @ (
                a_thr + a_drg + np.array([0.0, 0.0, -quad.g]) + _GRAVITY
            )

        accel_meas = sensors["accel"].apply(true_accel_body, dt=_DT)
        gyro_meas  = sensors["gyro"].apply(omega, dt=_DT)
        mag_meas   = sensors["mag"].apply(R.T @ _MAG_REF)

        ekf.predict(gyro_meas, accel_meas)
        ekf.correct_accel(accel_meas)
        ekf.correct_mag(mag_meas)

        z_baro = float(sensors["baro"].apply(np.array([pos[2]]))[0])
        ekf.correct_altitude(z_baro, r_z=0.0004)

        if i % 10 == 0:
            gps_pos, gps_vel, gps_valid = sensors["gps"].step(pos, vel)
            if gps_valid:
                ekf.correct_position(gps_pos, R_pos=np.eye(3) * 0.25)
                ekf.correct_velocity(gps_vel, R_vel=np.eye(3) * 0.01)

        est    = ekf.get_state()
        target = traj.position[i]
        motors = ctrl.compute(target, 0.0, _DT, traj.position[max(0, i - 1)])

        true_pos[i]  = pos
        est_pos[i]   = est["position"]
        rotations[i] = R

        quad.update(_DT, motors)

    return true_pos, est_pos, rotations, t


def _make_random_traj(rng: np.random.Generator | None = None):
    """Build a smooth random trajectory via minimum-snap.

    Generates 3-5 random waypoints inside a +-4 m XY box at altitudes 1-3 m,
    returning to home [0, 0, 1.5] at the end.  Segment duration ~5 s each.
    """
    if rng is None:
        rng = np.random.default_rng()
    n_inner  = int(rng.integers(3, 6))
    xy       = rng.uniform(-4.0, 4.0, (n_inner, 2))
    z        = rng.uniform(1.0, 3.0, (n_inner, 1))
    inner    = np.hstack([xy, z])
    home     = np.array([[0.0, 0.0, 1.5]])
    waypoints = np.vstack([home, inner, home])
    times     = list(np.linspace(0.0, 5.0 * (len(waypoints) - 1), len(waypoints)))
    return generate_minimum_snap(
        [tuple(float(c) for c in wp) for wp in waypoints],
        times=times,
        dt=_DT,
    )


# ---------------------------------------------------------------------------
# Scene helper
# ---------------------------------------------------------------------------

def _push_cloud(server, name: str, arr: np.ndarray,
                color: tuple, size: float = 0.02) -> None:
    p = arr.astype(np.float32)
    c = np.tile(np.array(color, dtype=np.uint8), (len(p), 1))
    server.scene.add_point_cloud(f"/{name}", points=p, colors=c, point_size=size)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        import viser
    except ImportError:
        raise SystemExit("viser is required: pip install viser")

    quad = QuadcopterDynamics(motor_time_constant=0.04)
    ctrl = QuadcopterController(quad)

    traj = generate_circular(
        duration=30.0, sample_rate=int(1 / _DT), radius=3.0, angular_vel=0.4,
    )
    print(f"Initial trajectory: {len(traj.t)} steps over {traj.t[-1]:.1f} s")

    print("Running initial simulation ...")
    true_pos, est_pos, rotations, t = _run_simulation(traj, quad, ctrl)
    err0 = float(np.linalg.norm(true_pos - traj.position, axis=1).mean())
    print(f"  mean tracking error: {err0:.3f} m")

    viewer = DroneViewer(port=8081)
    srv    = viewer.server

    _push_cloud(srv, "ref_traj",      traj.position, (50, 220, 80),  0.03)
    _push_cloud(srv, "true_traj",     true_pos,      (0, 120, 255),  0.02)
    _push_cloud(srv, "filtered_traj", est_pos,       (255, 80, 0),   0.02)
    frame_name       = viewer.add_quadcopter_frame()
    world_frame_hdl  = viewer.add_world_frame(axes_length=0.6)
    body_frame_hdl   = viewer.add_body_frame_axes(frame_name, axes_length=0.3)

    lock  = threading.Lock()
    state: dict = {
        "traj":      traj,
        "true_pos":  true_pos,
        "est_pos":   est_pos,
        "rotations": rotations,
        "t":         t,
        "playing":   False,
    }

    def _set_frame(idx: int) -> None:
        with lock:
            tp   = state["true_pos"]
            rots = state["rotations"]
        idx = max(0, min(idx, len(tp) - 1))
        viewer.update_quadcopter_pose(frame_name, tp[idx], rots[idx])

    # ---- GUI: trajectory controls ----------------------------------------
    status_md = srv.gui.add_markdown("**Status:** Initial circular trajectory loaded.")
    gen_btn   = srv.gui.add_button("\U0001f3b2  Random Trajectory")
    track_btn = srv.gui.add_button("\u25b6  Track")

    @gen_btn.on_click
    def _on_generate(event: viser.GuiEvent) -> None:
        status_md.content = "**Status:** Generating random trajectory..."
        try:
            new_traj = _make_random_traj()
        except Exception as exc:
            status_md.content = f"**Status:** Generation failed: {exc}"
            return
        with lock:
            state["traj"] = new_traj
        _push_cloud(srv, "ref_traj", new_traj.position, (50, 220, 80), 0.03)
        status_md.content = (
            f"**Status:** Random trajectory ready "
            f"({len(new_traj.t)} steps, {float(new_traj.t[-1]):.1f} s). "
            "Press \u25b6 Track to simulate."
        )

    @track_btn.on_click
    def _on_track(event: viser.GuiEvent) -> None:
        with lock:
            traj_            = state["traj"]
            state["playing"] = False
        play_btn.label    = "\u25b6  Play"
        status_md.content = "**Status:** Simulating..."
        try:
            tp, ep, rots, new_t = _run_simulation(traj_, quad, ctrl)
        except Exception as exc:
            status_md.content = f"**Status:** Simulation failed: {exc}"
            return
        with lock:
            state["true_pos"]  = tp
            state["est_pos"]   = ep
            state["rotations"] = rots
            state["t"]         = new_t
        _push_cloud(srv, "true_traj",     tp, (0, 120, 255), 0.02)
        _push_cloud(srv, "filtered_traj", ep, (255, 80, 0),  0.02)
        err_mean = float(np.linalg.norm(tp - traj_.position[:len(tp)], axis=1).mean())
        slider.max   = len(new_t) - 1
        slider.value = 0
        _set_frame(0)
        status_md.content = f"**Status:** Done \u2014 mean tracking error {err_mean:.3f} m"

    # ---- GUI: playback controls ------------------------------------------
    slider    = srv.gui.add_slider("Time step", min=0, max=len(t) - 1, step=1,
                                   initial_value=0)
    play_btn  = srv.gui.add_button("\u25b6  Play")
    reset_btn = srv.gui.add_button("\u27f3  Reset")
    ekf_chk         = srv.gui.add_checkbox("Show EKF estimate",  initial_value=True)
    world_frame_chk = srv.gui.add_checkbox("Show world frame",   initial_value=True)
    body_frame_chk  = srv.gui.add_checkbox("Show body frame",    initial_value=True)
    speed_inp = srv.gui.add_number("Speed \xd7", initial_value=1.0,
                                   min=0.1, max=100.0, step=0.1)

    @slider.on_update
    def _on_slider(event: viser.GuiEvent) -> None:
        _set_frame(int(slider.value))

    @play_btn.on_click
    def _on_play(event: viser.GuiEvent) -> None:
        with lock:
            state["playing"] = not state["playing"]
        play_btn.label = "\u23f8  Pause" if state["playing"] else "\u25b6  Play"

    @reset_btn.on_click
    def _on_reset(event: viser.GuiEvent) -> None:
        with lock:
            state["playing"] = False
        play_btn.label = "\u25b6  Play"
        slider.value   = 0
        _set_frame(0)

    @ekf_chk.on_update
    def _on_ekf_toggle(event: viser.GuiEvent) -> None:
        with lock:
            ep = state["est_pos"].astype(np.float32)
        color = (255, 80, 0) if ekf_chk.value else (0, 0, 0)
        _push_cloud(srv, "filtered_traj", ep, color, 0.02)

    @world_frame_chk.on_update
    def _on_world_frame_toggle(event: viser.GuiEvent) -> None:
        world_frame_hdl.visible = world_frame_chk.value

    @body_frame_chk.on_update
    def _on_body_frame_toggle(event: viser.GuiEvent) -> None:
        body_frame_hdl.visible = body_frame_chk.value

    # ---- Initial pose & event loop ---------------------------------------
    _set_frame(0)

    if hasattr(srv, "get_port"):
        port = srv.get_port()
    else:
        port = 8081
    print(f"Viewer at http://localhost:{port}/")
    print("  \U0001f3b2 Random Trajectory -> generate  |  \u25b6 Track -> simulate")

    dt_base = float(t[1] - t[0]) if len(t) > 1 else _DT
    try:
        while True:
            with lock:
                is_playing = state["playing"]
                t_arr      = state["t"]
            if is_playing:
                next_idx = int(slider.value) + 1
                if next_idx >= len(t_arr):
                    with lock:
                        state["playing"] = False
                    play_btn.label = "\u25b6  Play"
                    _time.sleep(0.05)
                else:
                    slider.value = next_idx
                    _set_frame(next_idx)
                    _time.sleep(dt_base / float(speed_inp.value))
            else:
                _time.sleep(0.05)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
