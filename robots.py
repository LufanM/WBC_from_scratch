"""
robots.py —— 机器人配置：让同一套全身控制程序（如 demo_hqp_interactive.py）适配不同的 URDF。

一个机器人由 RobotConfig 描述，只包含“和具体机器人有关的信息”：

    怎么得到 MuJoCo 模型    load()，返回 (model, data)
    关节分组                torso / left_arm / right_arm（可选 base）
    末端                    左右末端 site 名（WBC 跟踪的坐标系）
    躯干坐标系              body 名，以及跟踪哪些分量
    初始姿态                {关节名: 角度}
    躯干安全盒（可选）      世界坐标 (x_min, x_max, z_min, z_max)；None 表示以初始躯干位置为中心自动给出

任意 URDF 都可以用 load_urdf() 加载：它在加载时用 MjSpec 补上 WBC 仿真需要、而 URDF 里没有的东西——
位置执行器（增益按初始姿态下的关节惯量整定）、末端 site、重力补偿、地面和灯光，可选平面底盘
（base_x / base_y / base_yaw）。所以接入新机器人只需写一个 RobotConfig：

    MY_ROBOT = RobotConfig(
        name="my_robot",
        load=lambda: load_urdf("path/to/robot.urdf",
                               actuated=TORSO + LEFT + RIGHT,
                               ee_sites={"left_ee": "left_tool_link", "right_ee": "right_tool_link"},
                               home=HOME),
        torso_joints=TORSO, left_arm_joints=LEFT, right_arm_joints=RIGHT,
        left_ee_site="left_ee", right_ee_site="right_ee",
        torso_body="torso_top_link", home=HOME)

也可以写成 JSON 文件，用 load_config_file() 读入（见该函数说明）。

内置：
    r1        本仓库 assets/ 中原始 URDF（Galaxea R1 Pro），经 tools/convert_urdf.py 转换的 MJCF
    r1_urdf   同一个 URDF，直接用通用加载器 load_urdf() 加载（验证通用路径）
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))

# 位置伺服的闭环带宽和阻尼比，与 tools/convert_urdf.py 相同
SERVO_BANDWIDTH = 20.0   # rad/s
SERVO_DAMPING = 1.2
MESH_EXTENSIONS = (".stl", ".obj", ".msh", ".ply")    # MuJoCo 能直接加载的网格格式


def full_mass_matrix(model, data):
    """稠密质量矩阵。MuJoCo 3.3 起 mj_fullM 改为 (m, d, dst)，旧版为 (m, dst, d.qM)。"""
    M = np.zeros((model.nv, model.nv))
    try:
        mujoco.mj_fullM(model, data, M)
    except TypeError:
        mujoco.mj_fullM(model, M, data.qM)
    return M


@dataclass
class RobotConfig:
    name: str
    load: Callable[[], tuple]
    torso_joints: list
    left_arm_joints: list
    right_arm_joints: list
    left_ee_site: str
    right_ee_site: str
    torso_body: str
    home: dict = field(default_factory=dict)
    base_joints: list = field(default_factory=list)     # 可选的底盘自由度（--base 时参与规划）
    torso_components: tuple = ("x", "z")                 # 躯干任务跟踪的分量
    torso_box: tuple | None = None                       # (x_min, x_max, z_min, z_max)，世界坐标
    torso_box_margin: tuple = (0.15, 0.25, 0.40, 0.10)   # 自动安全盒：初始躯干 x 向后/向前，z 向下/向上
    stress_hand_dx: float = 0.9                          # B 键压力点：双手在初始位置前方多远（m）
    gravcomp: bool = True                                # 仿真中开重力补偿（等效于伺服带重力前馈）

    def arm_joints(self):
        return list(self.left_arm_joints) + list(self.right_arm_joints)


# =============================================================================
# 通用 URDF 加载器
# =============================================================================
def _resolve_package_paths(text, urdf_dir):
    """把 package://<包名>/相对路径 换成本地绝对路径：从 URDF 所在目录向上找名为 <包名> 的目录，
    找不到就把相对路径接在 URDF 所在目录下。"""
    def repl(m):
        pkg, rel = m.group(1), m.group(2)
        d = urdf_dir
        while True:
            if os.path.basename(d) == pkg and os.path.exists(os.path.join(d, rel)):
                return os.path.join(d, rel)
            parent = os.path.dirname(d)
            if parent == d:
                return os.path.join(urdf_dir, rel)
            d = parent
    return re.sub(r"package://([^/]+)/([^\"']+)", repl, text)


def _drop_unsupported_meshes(text):
    """去掉 MuJoCo 不能加载的网格（如 .glb / .dae）所在的 <visual> / <collision> 块。"""
    def keep(block):
        files = re.findall(r'filename="([^"]+)"', block)
        return all(f.lower().endswith(MESH_EXTENSIONS) for f in files)
    for tag in ("visual", "collision"):
        text = re.sub(rf"\s*<{tag}>.*?</{tag}>",
                      lambda m: m.group(0) if keep(m.group(0)) else "", text, flags=re.S)
    return text


def load_urdf(urdf_path, actuated, ee_sites, home=None, planar_base=False,
              floor=True, gravcomp=True):
    """加载任意 URDF，补上 WBC 仿真需要的部分，返回 (model, data)。

    actuated     要加位置执行器的关节（WBC 规划的关节）
    ee_sites     {site 名: link 名} 或 {site 名: (link 名, [x, y, z])}：在该 link 上加末端 site。
                 fixed 关节连接的 link（如 *_gripper_center）不会被合并，可以直接用
    home         {关节名: 角度}：初始姿态，同时用于整定执行器增益
    planar_base  在 URDF 根 link 上加 base_x / base_y（滑动）、base_yaw（转动）三个关节
    """
    urdf_path = os.path.abspath(urdf_path)
    urdf_dir = os.path.dirname(urdf_path)
    with open(urdf_path, encoding="utf-8") as f:
        text = f.read()
    text = _drop_unsupported_meshes(_resolve_package_paths(text, urdf_dir))
    # meshdir 设为 URDF 所在目录；保留路径、保留 visual、不合并 fixed 关节、修正不合法的惯量
    hint = (f'<mujoco><compiler meshdir="{urdf_dir}" balanceinertia="true" '
            f'discardvisual="false" strippath="false" fusestatic="false"/></mujoco>')
    text = re.sub(r"(<robot\b[^>]*>)", r"\1\n" + hint, text, count=1)

    spec = mujoco.MjSpec.from_string(text)
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    world = spec.worldbody
    root = world.first_body()

    # 碰撞：运动学 WBC 演示不需要接触，机器人几何体全部不参与碰撞
    # （URDF 的碰撞体常常互相穿插；底盘由位置伺服驱动的平面关节承载）
    for g in spec.geoms:
        g.contype, g.conaffinity = 0, 0
    if floor:
        world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5, 5, 0.1],
                       rgba=[0.25, 0.3, 0.35, 1], contype=0, conaffinity=1)
        world.add_light(pos=[0, 0, 4], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    if planar_base:
        root.add_joint(name="base_x", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0],
                       range=[-20, 20])
        root.add_joint(name="base_y", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 1, 0],
                       range=[-20, 20])
        root.add_joint(name="base_yaw", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1])
        actuated = ["base_x", "base_y", "base_yaw"] + [j for j in actuated
                                                       if j not in ("base_x", "base_y", "base_yaw")]
    if gravcomp:
        for b in spec.bodies:
            if b.name != "world":
                b.gravcomp = 1.0
    for site, ref in ee_sites.items():
        link, pos = (ref, [0, 0, 0]) if isinstance(ref, str) else (ref[0], list(ref[1]))
        body = spec.body(link)
        if body is None:
            raise KeyError(f"URDF 中没有 link {link!r}（末端 site {site!r}）")
        body.add_site(name=site, pos=pos, size=[0.012, 0.012, 0.012], rgba=[0.1, 0.9, 0.4, 1])
    acts = []
    for j in actuated:
        a = spec.add_actuator(name=f"{j}_p", target=j, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.set_to_position(kp=1.0, kv=0.0)
        acts.append(a)

    # 执行器增益：按初始姿态下的关节惯量整定，kp = I w^2、kv = 2 zeta w I
    model = spec.compile()
    data = mujoco.MjData(model)
    _set_home(model, data, home or {})
    if floor:                                 # 把机器人抬到最低点刚好在地面上
        root.pos = list(np.asarray(root.pos) + [0.0, 0.0, -_lowest_point(model, data)])
    M = full_mass_matrix(model, data)
    w, zeta = SERVO_BANDWIDTH, SERVO_DAMPING
    for a, j in zip(acts, actuated):
        dof = model.joint(j).dofadr[0]
        inertia = max(float(M[dof, dof]), 1e-4)
        a.set_to_position(kp=inertia * w * w, kv=2.0 * zeta * w * inertia)
    model = spec.compile()
    data = mujoco.MjData(model)
    _set_home(model, data, home or {})
    return model, data


def _lowest_point(model, data):
    """机器人几何体（不含地面）在世界坐标系中的最低 z（用各 geom 的包围盒估计）。"""
    z = np.inf
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == 0:
            continue
        R = data.geom_xmat[g].reshape(3, 3)
        center = data.geom_xpos[g] + R @ model.geom_aabb[g, :3]
        z = min(z, center[2] - np.abs(R[2]) @ model.geom_aabb[g, 3:])
    return 0.0 if not np.isfinite(z) else float(z)


def _set_home(model, data, home):
    for name, val in home.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            data.qpos[model.jnt_qposadr[jid]] = val
    mujoco.mj_forward(model, data)


# =============================================================================
# 内置机器人
# =============================================================================
def _r1_names():
    import robot_model as rm
    return rm


def load_mjcf(path, gravcomp=True):
    """加载 MJCF；gravcomp=True 时在编译前给所有 body 开重力补偿。

    重力补偿必须在编译前设置：运行时改 model.body_gravcomp 不生效（MuJoCo 在编译时统计
    ngravcomp，为 0 时直接跳过重力补偿）。
    """
    spec = mujoco.MjSpec.from_file(path)
    if gravcomp:
        for b in spec.bodies:
            if b.name != "world":
                b.gravcomp = 1.0
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _load_r1():
    rm = _r1_names()
    # MJCF 由 assets/ 中的原始 URDF 经 tools/convert_urdf.py 转换（过期时自动重转）
    return load_mjcf(rm.ensure_mjcf())


def r1_interactive_home():
    """R1 交互演示的初始姿态，参照 AlphaBrainSystem mujoco_hqp_interactive 的初始姿态设计：

    躯干   前三个俯仰关节取 (a, -2a, -a)，a = 30°：微蹲，躯干保持直立（第 3 关节转轴为 -y，
           三者的俯仰正好抵消），上下、前后都留有活动余量；腰部转动为 0
    手臂   上臂自然下垂、肩部略外展 0.2 rad（在限位斥力区之外，否则停下后会被斥力推着飘移），肘弯 90°，前臂朝前：双手在胸前，
           约 (0.47, ±0.31, 1.05) m
    """
    rm = _r1_names()
    a = np.radians(30.0)
    home = dict(zip(rm.TORSO_JOINTS, [a, -2.0 * a, -a, 0.0]))
    arm = [0.0, 0.2, 0.0, -np.pi / 2, 0.0, 0.0, 0.0]
    home.update(zip(rm.LEFT_ARM_JOINTS, arm))
    home.update(zip(rm.RIGHT_ARM_JOINTS, arm))
    return home


def _r1_config(name, load):
    rm = _r1_names()
    return RobotConfig(
        name=name, load=load,
        torso_joints=list(rm.TORSO_JOINTS),
        left_arm_joints=list(rm.LEFT_ARM_JOINTS),
        right_arm_joints=list(rm.RIGHT_ARM_JOINTS),
        base_joints=list(rm.BASE_JOINTS),
        left_ee_site=rm.LEFT_EE_SITE, right_ee_site=rm.RIGHT_EE_SITE,
        torso_body=rm.TORSO_BODY,
        home=r1_interactive_home())


def _load_r1_urdf():
    rm = _r1_names()
    return load_urdf(rm.URDF_PATH,
                     actuated=rm.TORSO_JOINTS + rm.LEFT_ARM_JOINTS + rm.RIGHT_ARM_JOINTS,
                     ee_sites={rm.LEFT_EE_SITE: "left_gripper_center",
                               rm.RIGHT_EE_SITE: "right_gripper_center"},
                     home=rm.HOME_POSE, planar_base=True)


ROBOTS = {
    "r1": lambda: _r1_config("r1", _load_r1),
    "r1_urdf": lambda: _r1_config("r1_urdf", _load_r1_urdf),
}


def load_config_file(path):
    """从 JSON 文件读机器人配置，用通用加载器加载其中的 URDF。键与 RobotConfig 同名，另加：

        "urdf": "robot.urdf"（相对 JSON 文件所在目录），
        "ee_sites": {"left_ee": "left_tool_link", "right_ee": ["right_link7", [0, 0, 0.1]]},
        "planar_base": false

    例：
        {"name": "my_robot", "urdf": "my_robot.urdf",
         "torso_joints": ["waist_pitch", "waist_yaw"],
         "left_arm_joints": ["l1", "l2", "l3", "l4", "l5", "l6", "l7"],
         "right_arm_joints": ["r1", "r2", "r3", "r4", "r5", "r6", "r7"],
         "ee_sites": {"left_ee": "l_tool", "right_ee": "r_tool"},
         "left_ee_site": "left_ee", "right_ee_site": "right_ee",
         "torso_body": "torso_link", "home": {"l4": -1.2, "r4": -1.2}}
    """
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    urdf = os.path.join(os.path.dirname(os.path.abspath(path)), cfg.pop("urdf"))
    ee_sites = cfg.pop("ee_sites")
    planar_base = cfg.pop("planar_base", False)
    actuated = cfg["torso_joints"] + cfg["left_arm_joints"] + cfg["right_arm_joints"]
    if planar_base:
        cfg.setdefault("base_joints", ["base_x", "base_y", "base_yaw"])
    for key in ("torso_components", "torso_box", "torso_box_margin"):
        if cfg.get(key) is not None:
            cfg[key] = tuple(cfg[key])
    home = cfg.get("home", {})
    return RobotConfig(load=lambda: load_urdf(urdf, actuated, ee_sites, home=home,
                                              planar_base=planar_base,
                                              gravcomp=cfg.get("gravcomp", True)), **cfg)


def get_robot(name_or_path):
    """内置名字（见 ROBOTS）或 JSON 配置文件路径 → RobotConfig。"""
    if name_or_path in ROBOTS:
        return ROBOTS[name_or_path]()
    if os.path.isfile(name_or_path):
        return load_config_file(name_or_path)
    raise KeyError(f"未知机器人 {name_or_path!r}：可选 {list(ROBOTS)}，或给 JSON 配置文件路径")
