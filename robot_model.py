"""
robot_model.py — shared MuJoCo model loading and scene helpers.

Keeps the joint-name definitions, MJCF loading (with lazy conversion from the
URDF) and target-marker plumbing in one place so ``simulate.py`` and the tests
cannot drift apart on any of it.
"""
from __future__ import annotations

import os

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "assets", "galaxea_r1pro_standalone")
URDF_PATH = os.path.join(PKG, "galaxea_r1pro.urdf")
MJCF_PATH = os.path.join(PKG, "galaxea_r1pro.xml")
# The camera variant (tools/convert_urdf.py --cameras) is the same robot plus the
# RealSense wrist meshes, so it can be dropped in for screenshots and demos:
#     R1PRO_MJCF=.../galaxea_r1pro_cameras.xml python3 simulate.py
MJCF_PATH = os.environ.get("R1PRO_MJCF") or MJCF_PATH

# --- controlled DOFs -------------------------------------------------------
# Mobile base: three real joints in the MJCF (see tools/convert_urdf.py), so the WBC
# plans them like any other DOF.
BASE_JOINTS = ["base_x", "base_y", "base_yaw"]
TORSO_JOINTS = [f"torso_joint{i}" for i in range(1, 5)]
LEFT_ARM_JOINTS = [f"left_joint{i}" for i in range(1, 8)]
RIGHT_ARM_JOINTS = [f"right_joint{i}" for i in range(1, 8)]
# End-effector sites, created by the converter at each gripper centre.
LEFT_EE_SITE = "left_ee"
RIGHT_EE_SITE = "right_ee"

# Extra frames the whole-body controller can track (WholeBodyController(frames=...)):
# the torso top link, and the mobile base, which only moves in x, y and yaw
# (rz = rotation about world z).
TORSO_BODY = "torso_link4"
BASE_BODY = "base_link"
R1_FRAMES = {
    "torso": {"body": TORSO_BODY},
    "base": {"body": BASE_BODY, "components": ("x", "y", "rz")},
}

# Home posture lives with the converter, which linearizes the servo gains about
# it; re-exported here so runtime code has one obvious import site.
from tools.convert_urdf import HOME_POSE  # noqa: E402  (re-export)

MARKER_COLORS = [
    (0.9, 0.15, 0.15, 0.9),
    (0.15, 0.4, 0.95, 0.9),
]


def ensure_mjcf(force=False):
    """Return the MJCF path, converting from the URDF if needed.

    The XML is a build artifact of ``galaxea_r1pro.urdf``; regenerate it
    whenever it is missing or older than its source.
    """
    stale = (not os.path.exists(MJCF_PATH) or
             os.path.getmtime(MJCF_PATH) < os.path.getmtime(URDF_PATH))
    if force or stale:
        from tools.convert_urdf import convert
        convert(URDF_PATH, MJCF_PATH)
    return MJCF_PATH


def _marker_xml(n):
    """Free-floating marker bodies, appended to the scene before compiling.

    MuJoCo models are static once compiled, so target markers cannot be added
    at runtime the way PyBullet's ``createMultiBody`` allowed. Instead we
    compile ``n`` of them up front, parked underground, and move them into
    place with ``add_marker``.
    """
    out = []
    for i in range(n):
        r, g, b, a = MARKER_COLORS[i % len(MARKER_COLORS)]
        out.append(
            f'<body name="marker{i}" mocap="true" pos="0 0 -1">'
            f'<geom type="sphere" size="0.035" rgba="{r} {g} {b} {a}" '
            f'contype="0" conaffinity="0"/></body>')
    return "\n".join(out)


def load_model(n_markers=0, force_convert=False):
    """Compile the scene.

    Returns ``(model, data, marker_names)``. Marker bodies are mocap bodies, so
    they are posed directly through ``data.mocap_pos`` and take part in no
    dynamics.
    """
    path = ensure_mjcf(force=force_convert)
    with open(path, "r", encoding="utf-8") as f:
        xml = f.read()
    if n_markers:
        xml = xml.replace("</worldbody>", _marker_xml(n_markers) +
                          "\n</worldbody>", 1)
    # Compile from string with the package dir as the asset root so relative
    # mesh paths still resolve.
    model = mujoco.MjModel.from_xml_string(xml, _asset_dict(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, [f"marker{i}" for i in range(n_markers)]


def _asset_dict(mjcf_path):
    """Load every mesh referenced by the MJCF into an in-memory asset dict.

    ``from_xml_string`` has no directory context, so meshes must be handed to
    it explicitly. Keys are the paths exactly as written in the XML.
    """
    import xml.etree.ElementTree as ET

    root = ET.parse(mjcf_path).getroot()
    base = os.path.dirname(mjcf_path)
    meshdir = ""
    comp = root.find("compiler")
    if comp is not None and comp.get("meshdir"):
        meshdir = comp.get("meshdir")
    assets = {}
    for mesh in root.iter("mesh"):
        rel = mesh.get("file")
        if not rel:
            continue
        full = os.path.join(base, meshdir, rel)
        with open(full, "rb") as f:
            assets[rel] = f.read()
    return assets


def add_marker(model, data, name, pos):
    """Move a compiled marker body to a world position."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise KeyError(f"marker body {name!r} not in model")
    mocap_id = model.body_mocapid[bid]
    if mocap_id < 0:
        raise ValueError(f"body {name!r} is not a mocap body")
    data.mocap_pos[mocap_id] = np.asarray(pos, float)
    mujoco.mj_forward(model, data)
