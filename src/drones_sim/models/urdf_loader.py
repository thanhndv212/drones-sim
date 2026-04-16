"""URDF-based quadcopter model loader.

Parses the bundled quadcopter URDF file and provides data structures
for visualization and physics integration.  Uses only the Python
standard library (xml.etree.ElementTree) — no external URDF packages
are required.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class URDFGeometry:
    """A single URDF geometry primitive."""

    geom_type: str  # "box", "cylinder", "sphere"
    size: tuple[
        float, ...
    ] = ()  # box: (x,y,z); cylinder: (radius,length); sphere: (radius,)

    # Origin relative to the owning link frame.
    origin_xyz: NDArray = field(default_factory=lambda: np.zeros(3))
    origin_rpy: NDArray = field(default_factory=lambda: np.zeros(3))


@dataclass
class URDFLink:
    """Parsed URDF link."""

    name: str
    mass: float = 0.0
    inertia: NDArray = field(default_factory=lambda: np.zeros((3, 3)))
    inertia_origin_xyz: NDArray = field(default_factory=lambda: np.zeros(3))

    visual: URDFGeometry | None = None
    collision: URDFGeometry | None = None
    color_rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)

    @property
    def color_uint8(self) -> tuple[int, int, int]:
        """Return ``(R, G, B)`` as uint8 values in [0, 255]."""
        r, g, b, _ = self.color_rgba
        return (int(r * 255), int(g * 255), int(b * 255))


@dataclass
class URDFJoint:
    """Parsed URDF joint."""

    name: str
    joint_type: str  # "fixed", "revolute", …
    parent: str
    child: str
    origin_xyz: NDArray = field(default_factory=lambda: np.zeros(3))
    origin_rpy: NDArray = field(default_factory=lambda: np.zeros(3))


@dataclass
class DroneURDFModel:
    """Complete parsed URDF model."""

    name: str
    links: list[URDFLink] = field(default_factory=list)
    joints: list[URDFJoint] = field(default_factory=list)
    materials: dict[str, tuple[float, float, float, float]] = field(
        default_factory=dict
    )

    # Convenience aggregates (filled after parsing).
    total_mass: float = 0.0
    total_inertia: NDArray = field(default_factory=lambda: np.zeros((3, 3)))

    def __repr__(self) -> str:
        link_names = [l.name for l in self.links]
        return (
            f"DroneURDFModel(name={self.name!r}, "
            f"links={link_names}, "
            f"total_mass={self.total_mass:.3f} kg)"
        )

    # ----- kinematic helpers -----

    def link_world_position(self, link_name: str) -> NDArray:
        """Compute the position of *link_name* relative to base_link by
        walking up the kinematic chain (fixed joints only)."""
        pos = np.zeros(3)
        current = link_name
        visited: set[str] = set()
        while current != "base_link" and current not in visited:
            visited.add(current)
            for j in self.joints:
                if j.child == current:
                    pos = pos + j.origin_xyz
                    current = j.parent
                    break
            else:
                break
        return pos

    def link_world_rpy(self, link_name: str) -> NDArray:
        """Accumulate RPY along the kinematic chain (additive for fixed joints)."""
        rpy = np.zeros(3)
        current = link_name
        visited: set[str] = set()
        while current != "base_link" and current not in visited:
            visited.add(current)
            for j in self.joints:
                if j.child == current:
                    rpy = rpy + j.origin_rpy
                    current = j.parent
                    break
            else:
                break
        return rpy


# ---------------------------------------------------------------------------
# URDF file path helper
# ---------------------------------------------------------------------------


def get_urdf_path() -> Path:
    """Return the absolute path to the bundled ``quadcopter.urdf``."""
    return Path(__file__).parent / "quadcopter.urdf"


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _parse_origin(elem: ET.Element | None) -> tuple[NDArray, NDArray]:
    """Extract xyz and rpy from an <origin> element."""
    if elem is None:
        return np.zeros(3), np.zeros(3)
    xyz = np.array([float(v) for v in elem.get("xyz", "0 0 0").split()])
    rpy = np.array([float(v) for v in elem.get("rpy", "0 0 0").split()])
    return xyz, rpy


def _parse_geometry(parent: ET.Element | None) -> URDFGeometry | None:
    """Parse a <visual> or <collision> block into a URDFGeometry."""
    if parent is None:
        return None
    geom_el = parent.find("geometry")
    if geom_el is None:
        return None

    origin_xyz, origin_rpy = _parse_origin(parent.find("origin"))

    for prim in ("box", "cylinder", "sphere"):
        el = geom_el.find(prim)
        if el is not None:
            if prim == "box":
                size = tuple(float(v) for v in el.get("size", "0 0 0").split())
            elif prim == "cylinder":
                size = (
                    float(el.get("radius", "0")),
                    float(el.get("length", "0")),
                )
            else:  # sphere
                size = (float(el.get("radius", "0")),)
            return URDFGeometry(
                geom_type=prim,
                size=size,
                origin_xyz=origin_xyz,
                origin_rpy=origin_rpy,
            )
    return None


def load_drone_urdf(urdf_path: str | Path | None = None) -> DroneURDFModel:
    """Parse a URDF file and return a :class:`DroneURDFModel`.

    Parameters
    ----------
    urdf_path:
        Path to the ``.urdf`` file.  Defaults to the bundled quadcopter model.
    """
    if urdf_path is None:
        urdf_path = get_urdf_path()
    tree = ET.parse(str(urdf_path))
    root = tree.getroot()

    model = DroneURDFModel(name=root.get("name", "unnamed"))

    # --- materials (top-level) ---
    for mat_el in root.findall("material"):
        name = mat_el.get("name", "")
        color_el = mat_el.find("color")
        if color_el is not None:
            rgba = tuple(
                float(v) for v in color_el.get("rgba", "0.5 0.5 0.5 1").split()
            )
            model.materials[name] = rgba  # type: ignore[assignment]

    # --- links ---
    for link_el in root.findall("link"):
        link = URDFLink(name=link_el.get("name", ""))

        # Inertial
        inertial = link_el.find("inertial")
        if inertial is not None:
            mass_el = inertial.find("mass")
            if mass_el is not None:
                link.mass = float(mass_el.get("value", "0"))
            inertia_el = inertial.find("inertia")
            if inertia_el is not None:
                ixx = float(inertia_el.get("ixx", "0"))
                ixy = float(inertia_el.get("ixy", "0"))
                ixz = float(inertia_el.get("ixz", "0"))
                iyy = float(inertia_el.get("iyy", "0"))
                iyz = float(inertia_el.get("iyz", "0"))
                izz = float(inertia_el.get("izz", "0"))
                link.inertia = np.array(
                    [
                        [ixx, ixy, ixz],
                        [ixy, iyy, iyz],
                        [ixz, iyz, izz],
                    ]
                )
            origin_el = inertial.find("origin")
            if origin_el is not None:
                link.inertia_origin_xyz, _ = _parse_origin(origin_el)

        # Visual
        link.visual = _parse_geometry(link_el.find("visual"))

        # Collision
        link.collision = _parse_geometry(link_el.find("collision"))

        # Color — look up material by name
        vis_el = link_el.find("visual")
        if vis_el is not None:
            mat_el = vis_el.find("material")
            if mat_el is not None:
                mat_name = mat_el.get("name", "")
                # Inline color overrides material name reference
                color_el = mat_el.find("color")
                if color_el is not None:
                    link.color_rgba = tuple(  # type: ignore[assignment]
                        float(v)
                        for v in color_el.get("rgba", "0.5 0.5 0.5 1").split()
                    )
                elif mat_name in model.materials:
                    link.color_rgba = model.materials[mat_name]

        model.links.append(link)

    # --- joints ---
    for joint_el in root.findall("joint"):
        origin_xyz, origin_rpy = _parse_origin(joint_el.find("origin"))
        parent_el = joint_el.find("parent")
        child_el = joint_el.find("child")
        joint = URDFJoint(
            name=joint_el.get("name", ""),
            joint_type=joint_el.get("type", "fixed"),
            parent=parent_el.get("link", "") if parent_el is not None else "",
            child=child_el.get("link", "") if child_el is not None else "",
            origin_xyz=origin_xyz,
            origin_rpy=origin_rpy,
        )
        model.joints.append(joint)

    # --- aggregates ---
    model.total_mass = sum(l.mass for l in model.links)
    model.total_inertia = sum(l.inertia for l in model.links)  # type: ignore[assignment]

    return model


# ---------------------------------------------------------------------------
# Mesh generation from URDF geometry primitives
# ---------------------------------------------------------------------------


def _box_mesh(sx: float, sy: float, sz: float) -> tuple[NDArray, NDArray]:
    """Generate vertices and faces for an axis-aligned box centred at origin."""
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    vertices = np.array(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],  # bottom
            [4, 6, 5],
            [4, 7, 6],  # top
            [0, 4, 5],
            [0, 5, 1],  # front
            [2, 6, 7],
            [2, 7, 3],  # back
            [0, 3, 7],
            [0, 7, 4],  # left
            [1, 5, 6],
            [1, 6, 2],  # right
        ],
        dtype=np.uint32,
    )
    return vertices, faces


def _cylinder_mesh(
    radius: float,
    length: float,
    n_segments: int = 16,
) -> tuple[NDArray, NDArray]:
    """Generate vertices and faces for a Z-axis cylinder centred at origin."""
    angles = np.linspace(0, 2 * np.pi, n_segments, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    hz = length / 2

    # Bottom + top ring vertices, then two cap centres
    bottom = np.stack(
        [cos_a * radius, sin_a * radius, np.full(n_segments, -hz)], axis=-1
    )
    top = np.stack(
        [cos_a * radius, sin_a * radius, np.full(n_segments, hz)], axis=-1
    )
    centres = np.array([[0, 0, -hz], [0, 0, hz]], dtype=np.float64)

    vertices = np.vstack([bottom, top, centres]).astype(np.float32)
    n = n_segments
    bc = 2 * n  # bottom centre index
    tc = 2 * n + 1  # top centre index

    faces_list: list[list[int]] = []
    for i in range(n):
        j = (i + 1) % n
        # Side quad as two triangles
        faces_list.append([i, j, j + n])
        faces_list.append([i, j + n, i + n])
        # Bottom cap
        faces_list.append([bc, j, i])
        # Top cap
        faces_list.append([tc, i + n, j + n])

    faces = np.array(faces_list, dtype=np.uint32)
    return vertices, faces


def _sphere_mesh(
    radius: float,
    n_lat: int = 8,
    n_lon: int = 16,
) -> tuple[NDArray, NDArray]:
    """Generate vertices and faces for a UV sphere centred at origin."""
    verts: list[list[float]] = []
    # Poles
    verts.append([0, 0, radius])  # north
    verts.append([0, 0, -radius])  # south

    for i in range(1, n_lat):
        theta = np.pi * i / n_lat
        for j in range(n_lon):
            phi = 2 * np.pi * j / n_lon
            verts.append(
                [
                    radius * np.sin(theta) * np.cos(phi),
                    radius * np.sin(theta) * np.sin(phi),
                    radius * np.cos(theta),
                ]
            )

    vertices = np.array(verts, dtype=np.float32)

    faces_list: list[list[int]] = []
    # North-pole fan
    for j in range(n_lon):
        jn = (j + 1) % n_lon
        faces_list.append([0, 2 + j, 2 + jn])
    # South-pole fan
    south_ring_start = 2 + (n_lat - 2) * n_lon
    for j in range(n_lon):
        jn = (j + 1) % n_lon
        faces_list.append([1, south_ring_start + jn, south_ring_start + j])
    # Body quads
    for i in range(n_lat - 2):
        ring = 2 + i * n_lon
        next_ring = ring + n_lon
        for j in range(n_lon):
            jn = (j + 1) % n_lon
            faces_list.append([ring + j, next_ring + j, next_ring + jn])
            faces_list.append([ring + j, next_ring + jn, ring + jn])

    faces = np.array(faces_list, dtype=np.uint32)
    return vertices, faces


def _rpy_to_matrix(rpy: NDArray) -> NDArray:
    """Convert roll-pitch-yaw (XYZ extrinsic) to a 3×3 rotation matrix."""
    cr, sr = np.cos(rpy[0]), np.sin(rpy[0])
    cp, sp = np.cos(rpy[1]), np.sin(rpy[1])
    cy, sy = np.cos(rpy[2]), np.sin(rpy[2])
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def geometry_to_mesh(geom: URDFGeometry) -> tuple[NDArray, NDArray]:
    """Convert a :class:`URDFGeometry` to ``(vertices, faces)`` arrays.

    The returned vertices are already transformed by the geometry's local
    origin (xyz + rpy).
    """
    if geom.geom_type == "box":
        verts, faces = _box_mesh(*geom.size)
    elif geom.geom_type == "cylinder":
        verts, faces = _cylinder_mesh(geom.size[0], geom.size[1])
    elif geom.geom_type == "sphere":
        verts, faces = _sphere_mesh(geom.size[0])
    else:
        raise ValueError(f"Unknown geometry type: {geom.geom_type!r}")

    # Apply the geometry-local origin transform
    R = _rpy_to_matrix(geom.origin_rpy)
    verts = (R @ verts.T).T + geom.origin_xyz.astype(np.float32)
    return verts.astype(np.float32), faces
