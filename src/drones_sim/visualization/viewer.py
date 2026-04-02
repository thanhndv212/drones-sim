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
    ) -> None:
        """Add a 3D line for a trajectory."""
        points = positions.astype(np.float32)
        colors = np.tile(np.array(color, dtype=np.uint8), (len(points), 1))
        self.server.scene.add_point_cloud(
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
    ) -> None:
        """Launch interactive playback with a time slider in the GUI.

        This blocks and serves the viewer until the user stops it.
        """
        # Add static elements
        self.add_trajectory(positions, "true_traj", color=(0, 120, 255))
        if filtered_positions is not None:
            self.add_trajectory(
                filtered_positions,
                "filtered_traj",
                color=(255, 80, 0),
            )
        if waypoints is not None:
            self.add_waypoints(waypoints)

        frame_name = self.add_quadcopter_frame()

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
            idx = int(slider.value)
            pos = positions[idx]
            if rotation_matrices is not None:
                R = rotation_matrices[idx]
            else:
                R = np.eye(3)
            self.update_quadcopter_pose(frame_name, pos, R)

        # Initial pose
        if rotation_matrices is not None:
            R0 = rotation_matrices[0]
        else:
            R0 = np.eye(3)
        self.update_quadcopter_pose(frame_name, positions[0], R0)

        if hasattr(self.server, "get_port"):
            port = self.server.get_port()
        else:
            port = self.server.request_port

        print(
            f"Viewer running at http://localhost:{port}/"
        )
        try:
            while True:
                import time
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
