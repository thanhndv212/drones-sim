"""Interactive 3D drone visualization using viser.

Provides a web-based 3D viewer for:
- Trajectory playback (true vs filtered)
- Quadcopter body frame with rotors
- Sensor vector overlays (accel, gyro arrows)
- Time-slider scrubbing
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

try:
    import viser
    import viser.transforms as tf
except ImportError:
    viser = None  # type: ignore


def _require_viser():
    if viser is None:
        raise ImportError(
            "viser is required for 3D visualization: pip install viser"
        )


class DroneViewer:
    """Interactive 3D viewer for drone simulation playback."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        _require_viser()
        self.server = viser.ViserServer(host=host, port=port)
        self._arm_length = 0.2
        self._frame_handles: dict[str, object] = {}

    # ------------------------------------------------------------------
    # Trajectory visualization
    # ------------------------------------------------------------------

    def add_trajectory(
        self,
        positions: NDArray,
        name: str = "trajectory",
        color: tuple[int, int, int] = (0, 120, 255),
        line_width: float = 2.0,
    ) -> object:
        """Add a 3D line for a trajectory. Returns the scene node handle."""
        points = positions.astype(np.float32)
        colors = np.tile(np.array(color, dtype=np.uint8), (len(points), 1))
        return self.server.scene.add_point_cloud(
            f"/{name}",
            points=points,
            colors=colors,
            point_size=0.02,
        )

    def add_waypoints(
        self,
        waypoints: list[tuple[float, float, float]],
        name: str = "waypoints",
    ) -> None:
        """Add sphere markers at each waypoint."""
        for i, wp in enumerate(waypoints):
            self.server.scene.add_icosphere(
                f"/{name}/wp_{i}",
                radius=0.05,
                color=(255, 50, 50),
                position=wp,
            )

    # ------------------------------------------------------------------
    # Quadcopter body
    # ------------------------------------------------------------------

    def add_quadcopter_frame(
        self,
        name: str = "quad",
        arm_length: float = 0.2,
    ) -> str:
        """Create a quadcopter body-frame group. Returns the frame name."""
        self._arm_length = arm_length
        frame_name = f"/robot/{name}"

        # Arms as thin cylinders
        arm_positions = [
            (arm_length, 0, 0),
            (0, arm_length, 0),
            (-arm_length, 0, 0),
            (0, -arm_length, 0),
        ]
        arm_colors = [
            (255, 50, 50),   # front = red
            (50, 200, 50),   # right = green
            (100, 100, 100),
            (100, 100, 100),
        ]

        for i, (pos, col) in enumerate(zip(arm_positions, arm_colors)):
            self.server.scene.add_icosphere(
                f"{frame_name}/rotor_{i}",
                radius=0.03,
                color=col,
                position=pos,
            )

        # Center body
        self.server.scene.add_icosphere(
            f"{frame_name}/body",
            radius=0.04,
            color=(50, 50, 200),
            position=(0, 0, 0),
        )

        return frame_name

    def add_quadcopter_urdf(
        self,
        name: str = "quad_urdf",
        urdf_model=None,
    ) -> str:
        """Create a quadcopter body-frame group from a parsed URDF model.

        Each URDF link is rendered as a mesh (box/cylinder/sphere) using
        ``server.scene.add_mesh_simple``.  All meshes are parented under a
        ``/robot/{name}`` frame so a single call to
        :meth:`update_quadcopter_pose` moves the whole assembly.

        Parameters
        ----------
        name:
            scene-graph name for the root frame.
        urdf_model:
            :class:`~drones_sim.models.DroneURDFModel` instance.  When *None*
            the bundled quadcopter URDF is loaded automatically.

        Returns
        -------
        str
            The root frame path (pass to :meth:`update_quadcopter_pose`).
        """
        from drones_sim.models import load_drone_urdf
        from drones_sim.models.urdf_loader import geometry_to_mesh

        if urdf_model is None:
            urdf_model = load_drone_urdf()

        frame_name = f"/robot/{name}"

        for link in urdf_model.links:
            if link.visual is None:
                continue

            # World-frame offset of this link origin (translation only, fixed joints)
            link_pos = urdf_model.link_world_position(link.name)

            # Build mesh in link-local frame then offset by link_pos
            verts, faces = geometry_to_mesh(link.visual)
            verts = verts + link_pos.astype(np.float32)

            color_uint8 = tuple(int(c * 255) for c in link.color_rgba[:3])

            self.server.scene.add_mesh_simple(
                f"{frame_name}/{link.name}",
                vertices=verts,
                faces=faces,
                color=color_uint8,
                wxyz=(1.0, 0.0, 0.0, 0.0),
                position=(0.0, 0.0, 0.0),
                flat_shading=False,
            )

        return frame_name

    def add_world_frame(self, axes_length: float = 0.6) -> object:
        """Add a static world-frame axes indicator at the origin.

        Returns the scene-node handle (has a ``.visible`` property).
        """
        return self.server.scene.add_frame(
            "/world_frame",
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(0.0, 0.0, 0.0),
            show_axes=True,
            axes_length=axes_length,
            axes_radius=axes_length * 0.02,
        )

    def add_body_frame_axes(self, parent_frame_name: str, axes_length: float = 0.3) -> object:
        """Add body-frame axes as a child of *parent_frame_name*.

        Because viser honours the ``/parent/child`` scene-graph hierarchy,
        these axes automatically follow the drone's pose without any
        per-frame update calls.

        Returns the scene-node handle (has a ``.visible`` property).
        """
        return self.server.scene.add_frame(
            f"{parent_frame_name}/body_axes",
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(0.0, 0.0, 0.0),
            show_axes=True,
            axes_length=axes_length,
            axes_radius=axes_length * 0.02,
        )

    def update_quadcopter_pose(
        self,
        frame_name: str,
        position: NDArray,
        rotation_matrix: NDArray,
    ) -> None:
        """Update the pose of a quadcopter frame."""
        wxyz = tf.SO3.from_matrix(rotation_matrix).wxyz
        handle = self._frame_handles.get(frame_name)
        if handle is None:
            try:
                handle = self.server.scene.add_frame(
                    frame_name,
                    wxyz=wxyz,
                    position=tuple(position),
                    show_axes=False,
                )
            except TypeError:
                handle = self.server.scene.add_frame(
                    frame_name,
                    wxyz_quaternion=wxyz,
                    position=tuple(position),
                    show_axes=False,
                )
            self._frame_handles[frame_name] = handle
            return

        handle.wxyz = tuple(wxyz)
        handle.position = tuple(position)

    # ------------------------------------------------------------------
    # Playback with time slider
    # ------------------------------------------------------------------

    def playback(
        self,
        t: NDArray,
        positions: NDArray,
        rotation_matrices: NDArray | None = None,
        filtered_positions: NDArray | None = None,
        waypoints: list[tuple[float, float, float]] | None = None,
        reference_positions: NDArray | None = None,
        urdf_model=None,
    ) -> None:
        """Launch interactive playback with a time slider in the GUI.

        Args:
            reference_positions: If provided, a (N, 3) array of desired positions.
                A live tracking-error plot (‖actual − reference‖) is shown in
                the GUI panel and updates with every frame.
            urdf_model: Optional :class:`~drones_sim.models.DroneURDFModel`.
                When provided the drone body is rendered as a URDF mesh instead
                of the default icosphere representation.

        This blocks and serves the viewer until the user stops it.
        """
        # Add static elements
        self.add_trajectory(positions, "true_traj", color=(0, 120, 255))
        filtered_handle = None
        if filtered_positions is not None:
            filtered_handle = self.add_trajectory(
                filtered_positions,
                "filtered_traj",
                color=(255, 80, 0),
            )
        if waypoints is not None:
            self.add_waypoints(waypoints)
        if reference_positions is not None:
            self.add_trajectory(
                reference_positions,
                "ref_traj",
                color=(50, 220, 80),
            )

        if urdf_model is not None:
            frame_name = self.add_quadcopter_urdf("quad", urdf_model)
        else:
            frame_name = self.add_quadcopter_frame()

        # Simulation time step (seconds per frame) — base value derived from data
        _dt_base = float(t[1] - t[0]) if len(t) > 1 else 0.05

        # --- Tracking-error plot (optional) ---
        _error_img = None
        _errors: NDArray | None = None
        _render_error_plot = None

        if reference_positions is not None:
            _errors = np.linalg.norm(positions - reference_positions, axis=1)
            import io
            import matplotlib.figure
            import matplotlib.backends.backend_agg as _agg

            def _render_error_plot(idx: int) -> NDArray:
                fig = matplotlib.figure.Figure(figsize=(4, 1.9))
                _agg.FigureCanvasAgg(fig)
                ax = fig.add_subplot(111)
                # Full trace in grey
                ax.plot(t, _errors, color="#888888", linewidth=0.7, alpha=0.5)
                # Elapsed trace in orange
                if idx > 0:
                    ax.plot(
                        t[: idx + 1],
                        _errors[: idx + 1],
                        color="#ff8c00",
                        linewidth=1.6,
                    )
                # Current-time cursor
                ax.axvline(
                    t[idx], color="#dd2222", linewidth=1.0, linestyle="--"
                )
                ax.set_xlim(t[0], t[-1])
                ax.set_ylim(
                    bottom=0, top=max(float(_errors.max()) * 1.15, 0.02)
                )
                ax.set_xlabel("t  (s)", fontsize=7)
                ax.set_ylabel("‖e‖  m", fontsize=7)
                ax.set_title(
                    f"Tracking Error  —  now: {_errors[idx]:.3f} m",
                    fontsize=8,
                    fontweight="bold",
                )
                ax.tick_params(labelsize=6)
                ax.grid(True, alpha=0.25, linewidth=0.5)
                fig.tight_layout(pad=0.4)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=90)
                buf.seek(0)
                import PIL.Image

                return np.array(PIL.Image.open(buf).convert("RGB"))

            _error_img = self.server.gui.add_image(_render_error_plot(0))

        # Wall-clock time of the last error-plot redraw (throttled to ≤10 fps)
        import time as _time
        _plot_last_draw: list[float] = [0.0]
        _PLOT_MIN_INTERVAL = 0.1  # seconds

        def _set_frame(idx: int) -> None:
            pos = positions[idx]
            R = (
                rotation_matrices[idx]
                if rotation_matrices is not None
                else np.eye(3)
            )
            self.update_quadcopter_pose(frame_name, pos, R)
            if _error_img is not None:
                now = _time.monotonic()
                if now - _plot_last_draw[0] >= _PLOT_MIN_INTERVAL:
                    _error_img.image = _render_error_plot(idx)
                    _plot_last_draw[0] = now

        # Time slider
        slider = self.server.gui.add_slider(
            "Time step",
            min=0,
            max=len(t) - 1,
            step=1,
            initial_value=0,
        )

        @slider.on_update
        def _on_slider(event: viser.GuiEvent) -> None:
            _set_frame(int(slider.value))

        # Playback controls
        playing = [False]
        play_btn = self.server.gui.add_button("▶  Play")
        reset_btn = self.server.gui.add_button("⟳  Reset")
        # EKF estimate toggle (only shown when filtered_positions provided)
        if filtered_handle is not None:
            ekf_chk = self.server.gui.add_checkbox(
                "Show EKF estimate", initial_value=True
            )

            @ekf_chk.on_update
            def _on_ekf_toggle(event: viser.GuiEvent) -> None:
                filtered_handle.visible = ekf_chk.value

        @play_btn.on_click
        def _on_play(event: viser.GuiEvent) -> None:
            playing[0] = not playing[0]
            play_btn.label = "⏸  Pause" if playing[0] else "▶  Play"

        speed_input = self.server.gui.add_number(
            "Speed ×",
            initial_value=1.0,
            min=0.1,
            max=100.0,
            step=0.1,
        )

        @reset_btn.on_click
        def _on_reset(event: viser.GuiEvent) -> None:
            playing[0] = False
            play_btn.label = "▶  Play"
            slider.value = 0
            _set_frame(0)

        # Initial pose
        _set_frame(0)

        if hasattr(self.server, "get_port"):
            port = self.server.get_port()
        else:
            port = self.server.request_port

        print(
            f"Viewer running at http://localhost:{port}/"
        )
        try:
            while True:
                if playing[0]:
                    next_idx = int(slider.value) + 1
                    if next_idx >= len(t):
                        # Reached end — stop playback
                        playing[0] = False
                        play_btn.label = "▶  Play"
                        _time.sleep(0.05)
                    else:
                        slider.value = next_idx
                        _set_frame(next_idx)
                        _time.sleep(_dt_base / float(speed_input.value))
                else:
                    _time.sleep(0.05)
        except KeyboardInterrupt:
            pass
