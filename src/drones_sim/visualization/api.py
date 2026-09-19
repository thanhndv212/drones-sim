"""Backend-neutral visualization entry point."""

from __future__ import annotations

from typing import Literal

from ..simulation import SimulationResult
from .dashboard import plot_simulation
from .rerun_viewer import visualize_rerun

VisualizationBackend = Literal["rerun", "matplotlib", "viser"]


def visualize(
    result: SimulationResult,
    *,
    backend: VisualizationBackend = "rerun",
    **kwargs,
):
    """Visualize a simulation result with Rerun by default.

    ``rerun`` accepts the arguments of :func:`visualize_rerun`, including
    ``mode``, ``path``, and ``url``. ``matplotlib`` accepts ``title``.
    ``viser`` accepts ``host``, ``port``, ``urdf_model``, or a prebuilt
    ``viewer`` and starts its blocking playback UI.
    """
    if backend == "rerun":
        return visualize_rerun(result, **kwargs)
    if backend == "matplotlib":
        return plot_simulation(result, **kwargs)
    if backend == "viser":
        # Keep the legacy backend lazy so Rerun/Matplotlib users do not need
        # the optional Viser dependency installed.
        from .viewer import DroneViewer

        viewer = kwargs.pop("viewer", None)
        host = kwargs.pop("host", "0.0.0.0")
        port = kwargs.pop("port", 8080)
        urdf_model = kwargs.pop("urdf_model", None)
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"unexpected Viser options: {names}")
        if viewer is None:
            viewer = DroneViewer(host=host, port=port)
        viewer.playback_result(result, urdf_model=urdf_model)
        return viewer
    raise ValueError(
        f"unknown visualization backend {backend!r}; "
        "expected 'rerun', 'matplotlib', or 'viser'"
    )
