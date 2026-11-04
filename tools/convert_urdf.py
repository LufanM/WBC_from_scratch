"""
convert_urdf.py — turn the Galaxea R1 Pro URDF into a MuJoCo (MJCF) model.

Why a conversion step: MuJoCo can compile a URDF directly, but the URDF alone
is not a usable *scene*. We need four things the URDF cannot express:

1. A planar mobile base. The URDF's ``base_link`` is the root; MuJoCo would
   weld it to the world. We insert three joints (``base_x``, ``base_y``,
   ``base_yaw``) so the base becomes a real 3-DOF planar floating frame the
   whole-body controller can plan for, in both kinematic and dynamic mode.
2. End-effector sites. URDF fixed joints are merged away by the MuJoCo
   compiler, so ``left_gripper_center`` / ``right_gripper_center`` do not
   survive as bodies. We re-create them as sites on ``*_link7``, composing the
   fixed-joint chain by hand so the EE frame is exactly the URDF's.
3. A ground plane, lights and a camera (a scene, not just a robot).
4. Position/velocity actuators for the controlled joints.

Run:
    python tools/convert_urdf.py            # writes assets/.../galaxea_r1pro.xml
    python tools/convert_urdf.py --cameras  # ... galaxea_r1pro_cameras.xml

The default model leaves out the RealSense wrist-camera visual meshes: the
d435 body is 231k triangles per arm and purely cosmetic. ``--cameras`` writes
a second model that includes them, for screenshots and demos. Both share the
same kinematics, actuators and collision geometry, so a controller written
against one works unchanged on the other.
"""
from __future__ import annotations

import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np

# This script lives in tools/; the robot assets are under the repo root.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "assets", "galaxea_r1pro_standalone")
URDF = os.path.join(PKG, "galaxea_r1pro.urdf")
MJCF = os.path.join(PKG, "galaxea_r1pro.xml")
# Same robot plus the RealSense visual meshes; see --cameras below.
MJCF_CAM = os.path.join(PKG, "galaxea_r1pro_cameras.xml")

# The RealSense camera body, transcoded from d435.glb by convert_glb.py.
D435_OBJ = "meshes/realsense/d435.obj"

# Fixed-joint chain from *_link7 down to the gripper centre (the EE frame).
EE_CHAIN = ["{side}_gripper_base_joint", "{side}_gripper_center_joint"]


# ------------------------------------------------------------------ URDF math
def rpy_to_mat(rpy):
    r, pi, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(pi), np.sin(pi)
    cy, sy = np.cos(y), np.sin(y)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def mat_to_quat(R):
    """Rotation matrix -> MuJoCo quaternion [w, x, y, z]."""
    w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    if w > 1e-8:
        x = (R[2, 1] - R[1, 2]) / (4 * w)
        y = (R[0, 2] - R[2, 0]) / (4 * w)
        z = (R[1, 0] - R[0, 1]) / (4 * w)
    else:  # 180 deg rotation: fall back to the largest diagonal term
        i = int(np.argmax(np.diag(R)))
        v = np.zeros(3)
        v[i] = np.sqrt(max(0.0, (1 + 2 * R[i, i] - np.trace(R)) / 4.0))
        j, k = (i + 1) % 3, (i + 2) % 3
        v[j] = (R[j, i] + R[i, j]) / (4 * v[i])
        v[k] = (R[k, i] + R[i, k]) / (4 * v[i])
        x, y, z = v
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def joint_origin(root, name):
    """(pos, R) of a URDF joint's <origin> relative to its parent link."""
    for j in root.iter("joint"):
        if j.get("name") != name:
            continue
        o = j.find("origin")
        pos = np.zeros(3)
        R = np.eye(3)
        if o is not None:
            if o.get("xyz"):
                pos = np.array([float(v) for v in o.get("xyz").split()])
            if o.get("rpy"):
                R = rpy_to_mat([float(v) for v in o.get("rpy").split()])
        return pos, R
    raise KeyError(f"joint {name!r} not found in URDF")


def ee_frame(root, side):
    """Compose the fixed-joint chain link7 -> gripper centre."""
    pos, R = np.zeros(3), np.eye(3)
    for tmpl in EE_CHAIN:
        p_i, R_i = joint_origin(root, tmpl.format(side=side))
        pos = pos + R @ p_i
        R = R @ R_i
    return pos, R


# --------------------------------------------------------------- URDF cleanup
def preprocess_urdf(text, cameras=False):
    """Strip what MuJoCo cannot consume and inject its compiler hints.

    - The RealSense visual meshes point at ``d435.glb``, and MuJoCo has no
      glTF support (its extension table covers .stl/.obj/.ply/.msh only, and
      the mesh decoder plugins are obj/stl/usd). With ``cameras=True`` we
      retarget the reference at the ``d435.obj`` baked by ``convert_glb.py``;
      by default those <visual> blocks are dropped instead, because the baked
      mesh is 231k triangles per camera and purely decorative. Either way the
      links keep their camera_stand collision geometry.
    - A <mujoco> block sets meshdir and enables inertia balancing: several
      URDF links have inertia tensors that violate the triangle inequality,
      which MuJoCo rejects outright.
    - fusestatic="false" prevents merging fixed joints, preserving links like
      zed_link, gripper_base, realsense_link as separate bodies.
    """
    if cameras:
        if not os.path.exists(os.path.join(PKG, D435_OBJ)):
            raise FileNotFoundError(
                f"{D435_OBJ} missing — run `python3 tools/convert_glb.py` first")
        text = text.replace("meshes/realsense/d435.glb", D435_OBJ)
    else:
        text = re.sub(
            r"\s*<visual>(?:(?!</visual>).)*d435\.glb(?:(?!</visual>).)*</visual>",
            "", text, flags=re.S)
    hint = ('<mujoco><compiler meshdir="." balanceinertia="true" '
            'discardvisual="false" strippath="false" fusestatic="false"/></mujoco>')
    return text.replace('<robot name="R1Pro">',
                        f'<robot name="R1Pro">\n  {hint}\n{BASE_CHAIN}', 1)


# A URDF cannot express a planar joint, so the mobile base is built from three
# single-DOF joints chained through two massless dummy links. `base_link` (the
# URDF's original root) becomes the child of that chain instead of the world,
# which is what turns it into a controllable 3-DOF base in MuJoCo.
DUMMY_INERTIAL = ('<inertial><mass value="1e-6"/>'
                  '<inertia ixx="1e-9" ixy="0" ixz="0" iyy="1e-9" iyz="0" '
                  'izz="1e-9"/></inertial>')
BASE_CHAIN = f"""  <link name="base_footprint">{DUMMY_INERTIAL}</link>
  <link name="base_x_link">{DUMMY_INERTIAL}</link>
  <link name="base_y_link">{DUMMY_INERTIAL}</link>
  <joint name="base_x" type="prismatic">
    <parent link="base_footprint"/><child link="base_x_link"/>
    <axis xyz="1 0 0"/>
    <limit effort="1e4" lower="-20" upper="20" velocity="5"/>
  </joint>
  <joint name="base_y" type="prismatic">
    <parent link="base_x_link"/><child link="base_y_link"/>
    <axis xyz="0 1 0"/>
    <limit effort="1e4" lower="-20" upper="20" velocity="5"/>
  </joint>
  <joint name="base_yaw" type="continuous">
    <parent link="base_y_link"/><child link="base_link"/>
    <axis xyz="0 0 1"/>
    <limit effort="1e4" velocity="5"/>
  </joint>
"""

# Joints the whole-body controller actuates: torso chain + both 7-DOF arms.
# Wheels, steering and grippers stay passive.
TORSO_JOINTS = [f"torso_joint{i}" for i in range(1, 5)]
LEFT_ARM_JOINTS = [f"left_joint{i}" for i in range(1, 8)]
RIGHT_ARM_JOINTS = [f"right_joint{i}" for i in range(1, 8)]
BASE_JOINTS = ["base_x", "base_y", "base_yaw"]
ACTUATED = BASE_JOINTS + TORSO_JOINTS + LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS

# Position (not velocity) actuators for the dynamic path: a pure velocity servo
# has no integral action, so gravity torque on the torso and arms produces a
# standing steady-state droop. The dynamic path integrates the WBC velocity into
# a position setpoint and lets these servos hold it.
#
# Gains are not hand-tuned constants but derived from the model's own inertia
# (see `tune_gains`): kp = w^2 * M_ii, kv = 2*zeta*w*M_ii. Fixed gains cannot
# work here because the effective inertia spans three orders of magnitude, from
# a 100 kg base slide to a wrist roll — one constant is either mush at the base
# or a violent oscillation at the wrist.
SERVO_BANDWIDTH = 20.0   # rad/s, closed-loop natural frequency
SERVO_DAMPING = 1.2      # zeta, slightly over-damped: no overshoot

# Home posture: torso upright, arms bent out in front rather than hanging
# straight. Mid-range would put the elbows in a near-singular straight-arm pose,
# which makes the first WBC ticks poorly conditioned. Gains are also linearized
# about this posture, so it lives here rather than in the runtime module.
HOME_POSE = {
    "torso_joint1": 0.0,
    "torso_joint2": 0.5,
    "torso_joint3": -0.5,
    "torso_joint4": 0.0,
    "left_joint1": -1.2, "left_joint2": 1.0, "left_joint3": 0.0,
    "left_joint4": -1.0, "left_joint5": 0.0, "left_joint6": 0.0,
    "left_joint7": 0.0,
    "right_joint1": -1.2, "right_joint2": 1.0, "right_joint3": 0.0,
    "right_joint4": -1.0, "right_joint5": 0.0, "right_joint6": 0.0,
    "right_joint7": 0.0,
}


# --------------------------------------------------------------- MJCF assembly
def _indent(elem, level=0):
    pad = "\n" + "  " * level
    if len(elem):
        if not (elem.text or "").strip():
            elem.text = pad + "  "
        for child in elem:
            _indent(child, level + 1)
        if not (child.tail or "").strip():
            child.tail = pad
    if level and not (elem.tail or "").strip():
        elem.tail = pad


def add_scene(root):
    """Ground plane, sky, lights and a tracking camera."""
    asset = root.find("asset") or ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", name="grid", type="2d", builtin="checker",
                  rgb1="0.28 0.30 0.33", rgb2="0.22 0.24 0.27",
                  width="512", height="512")
    ET.SubElement(asset, "material", name="grid", texture="grid",
                  texrepeat="12 12", reflectance="0.1")
    wb = root.find("worldbody")
    ET.SubElement(wb, "geom", name="floor", type="plane", size="0 0 0.05",
                  material="grid", condim="3")
    ET.SubElement(wb, "light", pos="0 0 4", dir="0 0 -1", directional="true")
    ET.SubElement(wb, "light", pos="2 -2 3", dir="-0.5 0.5 -1",
                  directional="false")
    ET.SubElement(wb, "camera", name="track", mode="fixed",
                  pos="2.6 -2.2 1.9", xyaxes="0.65 0.76 0 -0.3 0.26 0.92")


# The URDF puts base_link's origin at the chassis underside, with the wheels
# reaching 41.4 mm below it. Left as-is the chassis rests *in* the floor and the
# wheels sink through it, so the whole robot is lifted by that much.
RIDE_HEIGHT = 0.0414

# Bodies whose collision geometry is switched off. The base is a commanded
# planar joint, not a wheel-driven vehicle, so drivetrain contacts contribute
# nothing but ground friction fighting the base servo — and the chassis/wheel
# meshes interpenetrate each other in the URDF anyway. Arm links stay
# collidable.
NO_COLLIDE = (["base_link"] +
              [f"wheel_motor_link{i}" for i in (1, 2, 3)] +
              [f"steer_motor_link{i}" for i in (1, 2, 3)])

# Self-collision pairs that are always in contact in the URDF's own geometry
# (the forearm meshes overlap by ~2 mm at every posture), so contact between
# them is meaningless and only adds solver noise.
EXCLUDE_PAIRS = [(f"{s}_link5", f"{s}_link7") for s in ("left", "right")]


def set_ride_height(root):
    """Lift the base chain so the wheels rest on the floor plane."""
    # base_footprint is the welded URDF root, so the compiler merges it into
    # the world; base_x_link is the outermost body that actually survives.
    body = find_body(root, "base_x_link")
    pos = [float(v) for v in (body.get("pos") or "0 0 0").split()]
    pos[2] += RIDE_HEIGHT
    body.set("pos", " ".join(f"{v:.6f}" for v in pos))


def disable_drivetrain_collisions(root):
    """Turn off contacts on the chassis/wheels and the overlapping forearms."""
    for name in NO_COLLIDE:
        for geom in find_body(root, name).findall("geom"):
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    for a, b in EXCLUDE_PAIRS:
        ET.SubElement(contact, "exclude", body1=a, body2=b)


def fix_mirrored_meshes(mjcf_root, urdf_root):
    """Re-point every mesh geom at the asset the URDF actually asked for.

    The right arm and the second gripper finger are built by mirroring the
    left-hand meshes (``scale="1 -1 1"`` / ``"1 -1 -1"`` in the URDF). MuJoCo's
    URDF importer only creates one mirrored asset per .obj file: the *first*
    link that references e.g. link6.obj with a negative scale gets a proper
    mirrored mesh, and every later reference to the same file silently falls
    back to the unmirrored one. That produced two visible artefacts in the
    viewer: right_link6 rendering as two overlapping shells (mirrored visual
    geom + unmirrored collision geom in the same body), and right_finger2 /
    the collision fingers sitting on the wrong side of the gripper.

    So we redo the asset table ourselves. For every URDF link we read the
    mesh file and scale off its <visual>, intern one MJCF <mesh> asset per
    distinct (file, scale) pair, and rewrite all mesh geoms in the matching
    MJCF body to use it. Unreferenced assets left over from the import are
    dropped.
    """
    asset = mjcf_root.find("asset")
    if asset is None:
        return
    by_name = {m.get("name"): m for m in asset.findall("mesh")}
    interned = {}      # (file, scale) -> asset name
    for mesh in asset.findall("mesh"):
        interned.setdefault((mesh.get("file"), mesh.get("scale")),
                            mesh.get("name"))

    def intern(file, scale):
        """Asset name for this (file, scale), creating the asset if needed."""
        key = (file, scale)
        if key not in interned:
            stem = os.path.splitext(os.path.basename(file))[0]
            sig = "".join("n" if v.startswith("-") else "p"
                          for v in (scale or "1 1 1").split())
            name = f"{stem}_{sig}"
            n = 2
            while name in by_name:
                name, n = f"{stem}_{sig}{n}", n + 1
            attrs = {"name": name, "file": file}
            if scale:
                attrs["scale"] = scale
            by_name[name] = ET.SubElement(asset, "mesh", **attrs)
            interned[key] = name
        return interned[key]

    def want_for(node):
        return intern(node.get("filename"), node.get("scale"))

    fixed = 0
    for link in urdf_root.findall("link"):
        visuals = link.findall("visual/geometry/mesh")
        if not visuals:
            continue
        try:
            body = find_body(mjcf_root, link.get("name"))
        except KeyError:
            continue  # fixed-joint child merged away by the compiler

        # A link may carry several visuals (the wrist camera has a stand plus
        # the d435 body). Match each MJCF geom to the visual naming the same
        # .obj so a multi-visual link is not collapsed onto one asset; geoms
        # whose file matches nothing (collision-only shapes) are left alone.
        by_file = {os.path.basename(v.get("filename")): v for v in visuals}
        sole = want_for(visuals[0]) if len(visuals) == 1 else None
        for geom in body.findall("geom"):
            if geom.get("type") != "mesh":
                continue
            cur = by_name.get(geom.get("mesh"))
            src = by_file.get(os.path.basename(cur.get("file"))) if cur else None
            want = want_for(src) if src is not None else sole
            if want and geom.get("mesh") != want:
                geom.set("mesh", want)
                fixed += 1

    # Drop assets nothing references any more.
    used = {g.get("mesh") for g in mjcf_root.iter("geom")}
    dropped = 0
    for name, mesh in by_name.items():
        if name not in used:
            asset.remove(mesh)
            dropped += 1
    print(f"[convert] mesh assets: repointed {fixed} geoms, "
          f"dropped {dropped} unused")


def find_body(root, name):
    for b in root.iter("body"):
        if b.get("name") == name:
            return b
    raise KeyError(f"body {name!r} not found in compiled MJCF")


def add_ee_sites(mjcf_root, urdf_root):
    """Attach a site at each gripper centre.

    MuJoCo's compiler merges URDF fixed joints away, so ``*_gripper_center``
    is gone as a body by the time we get MJCF. We recompute its transform
    relative to ``*_link7`` from the original URDF and place a site there —
    that site is the frame the WBC tracks.
    """
    for side in ("left", "right"):
        pos, R = ee_frame(urdf_root, side)
        body = find_body(mjcf_root, f"{side}_link7")
        ET.SubElement(
            body, "site", name=f"{side}_ee", type="sphere", size="0.012",
            rgba="0.1 0.9 0.4 1" if side == "left" else "0.9 0.3 0.1 1",
            pos=" ".join(f"{v:.8f}" for v in pos),
            quat=" ".join(f"{v:.8f}" for v in mat_to_quat(R)),
        )


def add_actuators(root):
    """Position servos for the base + torso + arm joints (dynamic mode).

    Gains are placeholders here; ``tune_gains`` overwrites them once the model
    has been compiled and its mass matrix can be read.
    """
    act = root.find("actuator")
    if act is None:
        act = ET.SubElement(root, "actuator")
    joint_names = {j.get("name") for j in root.iter("joint")}
    for name in ACTUATED:
        if name not in joint_names:
            raise KeyError(f"actuated joint {name!r} missing from MJCF")
        ET.SubElement(act, "position", name=f"{name}_p", joint=name,
                      kp="1", kv="1")


def tune_gains(path):
    """Rewrite servo gains from the compiled model's own mass matrix.

    Loads the MJCF, reads the diagonal of the mass matrix at the home posture,
    and sets kp/kv per joint so every servo has the same closed-loop bandwidth
    and damping ratio regardless of the inertia it drives. This is why the
    torso (tens of kg·m²) and the wrist (grams) can share one control law.
    """
    import mujoco

    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    for name, val in HOME_POSE.items():
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            d.qpos[m.jnt_qposadr[jid]] = val
    mujoco.mj_forward(m, d)
    M = np.zeros((m.nv, m.nv))
    try:                                   # MuJoCo >= 3.3: mj_fullM(m, d, dst)
        mujoco.mj_fullM(m, d, M)
    except TypeError:                      # older: mj_fullM(m, dst, d.qM)
        mujoco.mj_fullM(m, M, d.qM)

    w, zeta = SERVO_BANDWIDTH, SERVO_DAMPING
    gains = {}
    for name in ACTUATED:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        inertia = max(float(M[m.jnt_dofadr[jid], m.jnt_dofadr[jid]]), 1e-4)
        gains[name] = (w * w * inertia, 2.0 * zeta * w * inertia)

    tree = ET.parse(path)
    root = tree.getroot()
    for a in root.iter("position"):
        kp, kv = gains[a.get("joint")]
        a.set("kp", f"{kp:.6g}")
        a.set("kv", f"{kv:.6g}")
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return gains


def add_gripper_equality(mjcf_root, urdf_root):
    """Add equality constraints for gripper mimic joints.

    URDF uses <mimic> tags to couple gripper_joint2 to gripper_joint.
    MuJoCo doesn't read <mimic>, so we manually add <equality> constraints.

    For each gripper:
      - joint1: left_gripper_joint (master, axis="0 1 0")
      - joint2: left_gripper_joint2 (slave, axis="0 -1 0", mimic joint1)

    The axis directions are opposite, so when joint1 moves +x, joint2 moves -x,
    creating symmetric closing motion.
    """
    equality = mjcf_root.find("equality")
    if equality is None:
        # Insert <equality> after <worldbody> and before <actuator>
        wb = mjcf_root.find("worldbody")
        idx = list(mjcf_root).index(wb) + 1
        equality = ET.Element("equality")
        mjcf_root.insert(idx, equality)

    # Add constraints for both grippers
    for side in ("left", "right"):
        master = f"{side}_gripper_joint"
        slave = f"{side}_gripper_joint2"

        # MuJoCo equality constraint: joint slave follows joint master
        # polycoef="0 1 0 0 0" means: slave = 1 * master (multiplier=1, offset=0)
        ET.SubElement(
            equality, "joint",
            joint1=master,
            joint2=slave,
            polycoef="0 1 0 0 0"
        )

    print(f"[convert] added gripper mimic constraints (equality)")


def tune_options(root):
    """Simulation options: 2 ms steps and an implicit integrator.

    ``implicitfast`` keeps the velocity actuators stable at this timestep;
    the default Euler integrator needs a much smaller step for the stiff
    torso gains.
    """
    opt = root.find("option")
    if opt is None:
        opt = ET.SubElement(root, "option")
    opt.set("timestep", "0.002")
    opt.set("integrator", "implicitfast")
    opt.set("gravity", "0 0 -9.81")

    # Preserve compiler settings from preprocess_urdf
    comp = root.find("compiler")
    if comp is not None:
        comp.set("angle", "radian")
        # Ensure our preprocessing settings are preserved
        if not comp.get("fusestatic"):
            comp.set("fusestatic", "false")
        if not comp.get("balanceinertia"):
            comp.set("balanceinertia", "true")
        if not comp.get("discardvisual"):
            comp.set("discardvisual", "false")



def convert(urdf_path=URDF, out_path=MJCF, cameras=False):
    import mujoco  # imported here so the URDF math above stays dependency-light

    with open(urdf_path, "r", encoding="utf-8") as f:
        urdf_text = f.read()
    urdf_root = ET.fromstring(urdf_text)
    if not cameras:
        # Keep the MJCF-side mesh bookkeeping in step with the patched URDF.
        for link in urdf_root.findall("link"):
            for vis in link.findall("visual"):
                mesh = vis.find("geometry/mesh")
                if mesh is not None and "d435" in (mesh.get("filename") or ""):
                    link.remove(vis)

    # Compile the patched URDF, then re-serialise it as MJCF so we can edit it.
    patched = os.path.join(os.path.dirname(urdf_path), "_mujoco_patched.urdf")
    with open(patched, "w", encoding="utf-8") as f:
        f.write(preprocess_urdf(urdf_text, cameras=cameras))
    model = mujoco.MjModel.from_xml_path(patched)
    raw = os.path.join(os.path.dirname(urdf_path), "_raw.xml")
    mujoco.mj_saveLastXML(raw, model)

    tree = ET.parse(raw)
    root = tree.getroot()
    root.set("model", "galaxea_r1pro")
    tune_options(root)
    add_scene(root)
    set_ride_height(root)
    disable_drivetrain_collisions(root)
    fix_mirrored_meshes(root, urdf_root)
    add_ee_sites(root, urdf_root)
    add_gripper_equality(root, urdf_root)
    add_actuators(root)

    _indent(root)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    for tmp in (patched, raw):
        os.remove(tmp)
    tune_gains(out_path)

    # Verify the result actually compiles and the sites survived.
    m = mujoco.MjModel.from_xml_path(out_path)
    sites = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i)
             for i in range(m.nsite)]
    print(f"[convert] wrote {out_path}")
    print(f"[convert] nq={m.nq} nv={m.nv} nu={m.nu} bodies={m.nbody}")
    print(f"[convert] sites: {sites}")
    assert {"left_ee", "right_ee"} <= set(sites), "EE sites missing"
    return out_path


if __name__ == "__main__":
    argv = [a for a in sys.argv[1:] if a != "--cameras"]
    cameras = "--cameras" in sys.argv[1:]
    src = argv[0] if argv else URDF
    dst = argv[1] if len(argv) > 1 else (MJCF_CAM if cameras else MJCF)
    convert(src, dst, cameras=cameras)
