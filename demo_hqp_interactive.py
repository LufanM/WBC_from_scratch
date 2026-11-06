"""
demo_hqp_interactive.py —— 可拖拽目标点的 HQP / WBC 交互仿真（躯干 + 双臂，可选底盘）

迁移自 AlphaBrainSystem 的 src/spinal/mujoco/mujoco_hqp_interactive.cpp：操作方式、四种控制模式、
状态机、插值、HUD、--demo 自检和 --data-record 记录格式都保持一致；求解器换成本仓库的
wbc_function（WholeBodyController + HQPSolver），不再依赖 QPPlanner / yaml。

机器人由 robots.py 中的 RobotConfig 描述（关节分组、末端、躯干坐标系、初始姿态、安全盒），
默认是本仓库 assets/ 中原始 URDF 的 Galaxea R1 Pro；换机器人只需换配置（--robot）。

=============================== 运行 ===============================
带窗口交互（需要显示器）：
  python3 demo_hqp_interactive.py

无窗口自检：按固定脚本依次跑四种模式（含 模式4 vs 模式1 的放置点 A/B 对照），打印误差后退出：
  python3 demo_hqp_interactive.py --demo

参数：所有参数都有默认值，分组列表和默认值用 --help 查看，当前生效的值用 --print-params 查看；
可以在命令行逐个给（--kp-pos 20 --torso-cost 1.0），也可以写成 JSON 文件用 --config 读入，
命令行优先于文件。参数的含义和调法见 HQP_GUIDE.md。

运行方式相关的选项：
  --robot r1        机器人：r1（默认，assets/ 中的 URDF 经 tools/convert_urdf.py 转换）、
                    r1_urdf（同一 URDF 用通用加载器 robots.load_urdf 加载），或 JSON 配置文件路径
  --base            底盘也参与规划（机器人有底盘关节时）。默认底盘不动，与 C++ 版一致
  --kinematic       运动学模式：关节直接等于指令（不经过物理仿真和执行器），用于排查控制本身
  --data-record     记录到 hqp_sim_trace/<时间>/trace.csv，绘图用 tools/plot_trace.py
  --trace-dir DIR   指定记录目录（同时开启 --data-record）
  --debug-input     在终端打印收到的键盘 / 鼠标事件

=========================== 控制模式设计 ===========================
手和躯干各自有一个状态，四种控制形式就是这两个状态的组合：

  手    : TRACK(跟踪目标)  / FROZEN(冻结)
  躯干  : TRACK(跟踪目标)  / FREE(不指定目标，和手臂一起送手) / FROZEN(冻结)

  模式1  全身协同·仅手端跟踪      手=TRACK   躯干=FREE
  模式2  全身协同·手+躯干跟踪     手=TRACK   躯干=TRACK     (HQP：手第 0 层，躯干第 1 层)
  模式3  仅躯干跟踪·手臂冻结      手=FROZEN  躯干=TRACK
  模式4  仅手臂跟踪·躯干冻结      手=TRACK   躯干=FROZEN    (与模式1做 A/B 对照)

对应到本仓库的 HQP（solver="hqp"）：
  第 -1 层  躯干安全盒（x、z 范围，见 RobotConfig.torso_box；线性化的软约束，最小化违反量后锁定）
  第  0 层  双手 6D 位姿（两只手同层，按权重加权）
  第  1 层  躯干位置 x、z（模式1 开 --torso-return-gain 时为回到初始位置）
  最后一层  附带阻尼 λ 和姿态项 ||v - v_posture||²_Wp
  冻结      WholeBodyController.set_frozen（lb = ub = 0），每层都满足
  关节限位  位置 + 速度限位作为上下界，每层都满足

状态机（模式随拖动自动对齐，不会出现拖动和模式不一致）：
  - 拖手球 / 按 P：手变为 TRACK；躯干若为 TRACK 则降为 FREE，若为 FROZEN 保持冻结。
  - 拖躯干球：躯干变为 TRACK，手保持不变。
  - 数字键 1~4 可随时直接切换模式。启动默认模式2（双手、躯干都保持在初始位置）。
所有切换都是实时、无扰的：新进入 TRACK 的部分以当前实测位姿作为目标（不跳变）。
拖动过程中（按下到松开）数字键 / P / B / R / L 被忽略。

=============================== 操作 ===============================
视角：左键 / 右键拖动 旋转，中键拖动 平移，滚轮 缩放
拖动目标球：
  1. 按住 Ctrl，左键按在球上开始拖动（同时按住 Shift：本次拖动为上下移动）。
  2. 保持左键按住移动鼠标：只移动黄色目标点。按住 X / Y / Z 只沿对应世界轴移动
     （0.6 s 内按过的轴键都当作按住，兼容远程桌面）。躯干球只能前后 / 上下。
  3. 松开左键：确认目标，生成直线 + 五次多项式（最小 jerk）插值，HQP 跟踪过去。
按键：
  1 / 2 / 3 / 4  切换控制模式      L  双手联动开关（默认开）
  P  双手目标前移 place-dx           B  压力测试点（双手前伸到够不到处，进入模式2）
  R  所有关节 moveJ 回初始姿态       T  机器人 40% 透明
  H  显示 / 隐藏帮助                Esc  退出
目标球颜色：绿 = 正在跟踪的指令目标，黄 = 拖到的目标点，红 = 持续跟不上（> 2 cm 超过 1 s），
蓝 = 冻结部分的实际位置，灰 = 自由部分的实际位置，橙 = 躯干指令目标。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import mujoco

import robots
import wbc_function as wf
from wbc_function.config import add_arguments, apply_arguments, apply_dict, param, to_dict
from wbc_function.tasks import quat_error

HERE = os.path.dirname(os.path.abspath(__file__))



LEFT, RIGHT, TORSO = 0, 1, 2
PART_NAMES = ("left hand", "right hand", "torso")

g_stop = False
g_debug_input = False


# ---------------------------------------------------------------- 四元数 [w, x, y, z]
def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def quat_from_rotvec(r):
    th = float(np.linalg.norm(r))
    if th < 1e-12:
        q = np.array([1.0, 0.5 * r[0], 0.5 * r[1], 0.5 * r[2]])
    else:
        q = np.concatenate([[math.cos(0.5 * th)], math.sin(0.5 * th) * np.asarray(r) / th])
    return q / np.linalg.norm(q)


IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])


# ---------------------------------------------------------------- 控制模式
HAND_TRACK, HAND_FROZEN = "track", "frozen"
TORSO_TRACK, TORSO_FREE, TORSO_FROZEN = "track", "free", "frozen"


@dataclass(frozen=True)
class ControlMode:
    hands: str = HAND_TRACK
    torso: str = TORSO_TRACK


def mode_from_number(n):
    return {1: ControlMode(HAND_TRACK, TORSO_FREE),
            2: ControlMode(HAND_TRACK, TORSO_TRACK),
            3: ControlMode(HAND_FROZEN, TORSO_TRACK)}.get(n, ControlMode(HAND_TRACK, TORSO_FROZEN))


def on_hand_target_event(m):
    """拖手 / 按 P：手进入 TRACK；躯干若在 TRACK 则降为 FREE，FROZEN 保持。"""
    return ControlMode(HAND_TRACK, TORSO_FREE if m.torso == TORSO_TRACK else m.torso)


def on_torso_target_event(m):
    """拖躯干：躯干进入 TRACK；手保持不变。"""
    return ControlMode(m.hands, TORSO_TRACK)


def mode_number(m):
    if m.hands == HAND_FROZEN:
        return 3
    return {TORSO_FREE: 1, TORSO_TRACK: 2}.get(m.torso, 4)


def mode_name(m):
    return {1: "1 WBC: hands track, torso free", 2: "2 HQP: hands + torso track",
            3: "3 torso track, arms frozen"}.get(mode_number(m), "4 arms track, torso frozen")


# ---------------------------------------------------------------- 插值
class LineSegment:
    """位置走直线、姿态走 slerp，共用一条五次多项式（最小 jerk）时间缩放 s(t)。

    峰值速度 = 1.875 * 距离 / T，因此 T = max(t_min, 1.875·距离/v_max, 1.875·角度/w_max)。
    velocity() 给出插值速度，作为任务的前馈（FrameTask 的 vel）。
    """

    def __init__(self):
        self.start = np.zeros(3)
        self.goal = np.zeros(3)
        self.ori_start = IDENTITY_QUAT.copy()
        self.ori_goal = IDENTITY_QUAT.copy()
        self.rotvec = np.zeros(3)        # ori_start -> ori_goal 的旋转向量（世界坐标系）
        self.t0 = 0.0
        self.duration = 0.0

    def hold(self, p, q, now):
        self.start = np.array(p, float)
        self.goal = self.start.copy()
        self.ori_start = np.array(q, float)
        self.ori_goal = self.ori_start.copy()
        self.rotvec = np.zeros(3)
        self.t0 = now
        self.duration = 0.0

    def _tau(self, now):
        return min(max((now - self.t0) / self.duration, 0.0), 1.0)

    def phase(self, now):
        if self.duration <= 0.0:
            return 1.0
        t = self._tau(now)
        return t * t * t * (10.0 - 15.0 * t + 6.0 * t * t)

    def phase_rate(self, now):
        """ds/dt。"""
        if self.duration <= 0.0 or now >= self.t0 + self.duration or now <= self.t0:
            return 0.0
        t = self._tau(now)
        return 30.0 * t * t * (1.0 - t) ** 2 / self.duration

    def sample(self, now):
        if self.duration <= 0.0:
            return self.goal.copy()
        return self.start + self.phase(now) * (self.goal - self.start)

    def sample_ori(self, now):
        if self.duration <= 0.0:
            return self.ori_goal.copy()
        return quat_mul(quat_from_rotvec(self.phase(now) * self.rotvec), self.ori_start)

    def velocity(self, now):
        """插值的 [线速度; 角速度]（世界坐标系）。"""
        r = self.phase_rate(now)
        return np.concatenate([r * (self.goal - self.start), r * self.rotvec])

    def done(self, now):
        return self.duration <= 0.0 or now >= self.t0 + self.duration

    def remaining(self, now):
        return 0.0 if self.done(now) else self.t0 + self.duration - now

    def required_duration(self, new_goal, new_ori, now, vmax, wmax, t_min):
        dist = float(np.linalg.norm(new_goal - self.sample(now)))
        ang = float(np.linalg.norm(quat_error(self.sample_ori(now), new_ori)))
        if dist < 1e-6 and ang < 1e-4:
            return 0.0
        return max(t_min, 1.875 * dist / vmax, 1.875 * ang / wmax)

    def retarget(self, new_goal, new_ori, now, dur):
        """从当前插值点重新出发（位置、姿态都连续）。"""
        self.start = self.sample(now)
        self.ori_start = self.sample_ori(now)
        self.goal = np.array(new_goal, float)
        self.ori_goal = np.array(new_ori, float)
        self.rotvec = quat_error(self.ori_start, self.ori_goal)
        self.t0 = now
        self.duration = dur


@dataclass
class PartState:
    seg: LineSegment = field(default_factory=LineSegment)
    ghost: np.ndarray = field(default_factory=lambda: np.zeros(3))   # 拖动中的目标点
    home: np.ndarray = field(default_factory=lambda: np.zeros(3))
    home_ori: np.ndarray = field(default_factory=lambda: IDENTITY_QUAT.copy())
    over_error_since: float = -1.0
    unreachable: bool = False


def demo_wbc_config():
    """本程序的控制器参数：在 WBCConfig 默认值的基础上开误差限幅、关回中（与 C++ 版一致）。"""
    return wf.WBCConfig(max_pos_error=0.02, max_rot_error=0.1, posture_k=0.0)


@dataclass
class Options:
    """本程序的全部参数。wbc 为控制器和求解器的参数，其余是本程序自己的。"""
    wbc: wf.WBCConfig = field(default_factory=demo_wbc_config)
    solver: str = param("hqp", "求解器：hqp / wqp / wls", "运行", type=str)
    control_dt: float = param(0.01, "控制周期 (s)，必须是模型物理步长的整数倍", "运行")
    command_base: str = param("reference", "WBC 的计算构型和指令累加起点：reference / cmd / measured",
                              "运行", type=str)
    cmd_lead_max: float = param(0.05, "指令最多领先实测关节角多少 (rad)", "运行")
    kinematic: bool = param(False, "运动学模式：关节直接等于指令，不经过物理仿真", "运行")
    # 分层（数字越小越优先）
    hand_priority: int = param(0, "双手任务所在层", "分层", type=int)
    torso_priority: int = param(1, "躯干任务所在层", "分层", type=int)
    box_priority: int = param(-1, "安全盒（软约束）所在层", "分层", type=int)
    # 手臂和躯干
    hand_weight: float = param(1.0, "手的任务权重", "手臂和躯干")
    torso_weight: float = param(1.0, "躯干的任务权重", "手臂和躯干")
    arm_cost: float = param(2e-3, "手臂关节的 W_p（动起来的代价）", "手臂和躯干")
    torso_cost: float = param(0.2, "模式1 下躯干关节的 W_p", "手臂和躯干")
    torso_return_gain: float = param(0.0, "模式1 下躯干回初始位置的增益（每周期纠正比例）", "手臂和躯干")
    # 奇异处理
    adaptive_lambda: float = param(0.05, "自适应阻尼上限 λmax，0 表示不开", "奇异处理")
    adaptive_lambda_sigma: float = param(0.1, "开始加阻尼的 σ0 (m/rad)", "奇异处理")
    # 躯干安全盒
    box_soft: bool = param(True, "安全盒用软约束（否则硬约束）", "躯干安全盒")
    box_beta: float = param(0.5, "安全盒的缓冲系数 β", "躯干安全盒")
    box_w_lin: float = param(1e3, "软约束的线性罚权重", "躯干安全盒")
    box_w_quad: float = param(1e1, "软约束的二次罚权重", "躯干安全盒")
    torso_box: tuple = param(None, "安全盒 x_min x_max z_min z_max（世界坐标，覆盖机器人配置）",
                             "躯干安全盒", nargs=4)
    torso_box_margin: tuple = param(None, "自动安全盒：向后 向前 向下 向上 (m，覆盖机器人配置)",
                                    "躯干安全盒", nargs=4)
    torso_margin: float = param(0.005, "拖动目标离安全盒边界的余量 (m)", "躯干安全盒")
    torso_leash: float = param(0.0, "躯干目标最多领先实测多少 (m)，0 表示不限", "躯干安全盒")
    # 目标插值
    hand_vmax: float = param(0.15, "手目标的最大线速度 (m/s)", "目标插值")
    hand_wmax: float = param(0.5, "手目标的最大角速度 (rad/s)", "目标插值")
    torso_vmax: float = param(0.08, "躯干目标的最大线速度 (m/s)", "目标插值")
    t_min: float = param(1.0, "一段插值的最短时长 (s)", "目标插值")
    movej_vmax: float = param(0.5, "R 键 moveJ 的峰值关节速度 (rad/s)", "目标插值")
    place_dx: float = param(0.35, "P 键放置点的前向距离 (m)", "目标插值")
    stress_hand_dx: float = param(None, "B 键压力点的前向距离 (m，覆盖机器人配置)", "目标插值")
    # 显示
    unreachable_error: float = param(0.02, "误差超过它持续 1 s 判为不可达 (m)", "显示")
    print_track: bool = param(False, "周期打印目标 / 实测 / 误差", "显示")
    print_period: float = param(0.5, "打印周期（仿真时间 s）", "显示")


# ---------------------------------------------------------------- 仿真初始化
def load_sim(robot: robots.RobotConfig, control_dt):
    model, data = robot.load()        # 重力补偿（robot.gravcomp）由加载函数在编译前加上
    steps = max(1, int(round(control_dt / model.opt.timestep)))
    if abs(steps * model.opt.timestep - control_dt) > 1e-9:
        raise RuntimeError(f"control dt {control_dt} 不是物理步长 {model.opt.timestep} 的整数倍")
    return model, data, steps


def set_initial_posture(robot: robots.RobotConfig, model, data):
    """机器人配置里的初始姿态；所有位置执行器的指令设为当前关节角（保持不动）。"""
    for name, angle in robot.home.items():
        j = model.joint(name)
        data.qpos[j.qposadr[0]] = angle
        data.qvel[j.dofadr[0]] = 0.0
    mujoco.mj_forward(model, data)
    for a in range(model.nu):
        if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT:
            data.ctrl[a] = data.qpos[model.jnt_qposadr[model.actuator_trnid[a, 0]]]


# ---------------------------------------------------------------- 控制器
class InteractiveController:
    def __init__(self, model, data, substeps, opt: Options, robot: robots.RobotConfig,
                 use_base=False):
        self.m, self.d, self.substeps, self.opt = model, data, substeps, opt
        self.robot = robot
        self.dt = opt.control_dt
        self.use_base = use_base and bool(robot.base_joints)
        r = robot
        self.joints = ((list(r.base_joints) if self.use_base else [])
                       + list(r.torso_joints) + r.arm_joints())
        self.ee_sites = [r.left_ee_site, r.right_ee_site]

        frames = {"torso": {"body": r.torso_body, "components": tuple(r.torso_components),
                            "priority": opt.torso_priority, "weight": opt.torso_weight}}
        # WBC 在 self.ref 上计算：reference 模式下它是参考构型，否则每周期复制实测状态
        self.ref = mujoco.MjData(model)
        self.wbc = wf.WholeBodyController(model, self.ref, self.joints, self.ee_sites,
                                          base_free=self.use_base, base_joints=r.base_joints,
                                          solver=opt.solver, frames=frames, dt=self.dt,
                                          config=opt.wbc)
        w = self.wbc
        self.lambda_base = opt.wbc.lambda_reg
        for site in self.ee_sites:
            w.task(site).priority = opt.hand_priority
            w.task(site).weight = opt.hand_weight
        w.set_joint_cost(r.arm_joints(), opt.arm_cost)
        w.add_constraint("torso_box", self._torso_box)
        print(f"[wbc] robot = {r.name}, {len(self.joints)} joints"
              f"{' (base planned)' if self.use_base else ''}")
        print(f"[wbc] task weight hand/torso = {opt.hand_weight}/{opt.torso_weight}, "
              f"joint cost arm/torso(mode 1) = {opt.arm_cost}/{opt.torso_cost}")
        c = opt.wbc
        print(f"[wbc] solver = {opt.solver}, dq_max = {c.max_joint_vel} rad/s, lambda = {c.lambda_reg}, "
              f"Kp = {c.kp_pos}/{c.kp_rot} 1/s, control dt = {self.dt} s")
        if opt.adaptive_lambda > 0.0:
            print(f"[wbc] adaptive lambda: lambda = {c.lambda_reg} + {opt.adaptive_lambda} * "
                  f"(1 - sigma_min / {opt.adaptive_lambda_sigma})^2 when sigma_min < "
                  f"{opt.adaptive_lambda_sigma}")

        # 关节 -> 执行器（位置执行器，ctrl 为关节目标角）
        jid_to_act = {int(model.actuator_trnid[a, 0]): a for a in range(model.nu)}
        missing = [n for n in self.joints if model.joint(n).id not in jid_to_act]
        if missing:
            raise RuntimeError(f"这些受控关节没有位置执行器：{missing}")
        self.act = np.array([jid_to_act[model.joint(n).id] for n in self.joints])
        self.arm_cols = [np.array([self.joints.index(n) for n in r.left_arm_joints]),
                         np.array([self.joints.index(n) for n in r.right_arm_joints])]
        self.box = None                  # 躯干安全盒 (x_min, x_max, z_min, z_max)，initialize() 中确定

        self.parts = [PartState() for _ in range(3)]
        self.errors = np.zeros(3)
        self.mode = mode_from_number(2)
        self.q_home = None
        self.movej = None                # dict(start, t0, duration)
        self.torso_y = 0.0
        self.link_hands = True
        self.degraded = False
        self.fails = 0
        self.degraded_steps = 0
        self.trace = None
        self.trace_rows = 0
        self.arm_sigma = np.zeros(2)
        self.next_print_time = 0.0
        self.solve_ms = 0.0
        self.last_cmd = None

    # ---- 状态读取
    def time(self):
        return float(self.d.time)

    def read_q(self):
        """实测的受控关节角。"""
        return self.d.qpos[self.wbc.qadr].copy()

    def hand_pose(self, p):
        """实测的手位姿 (位置, 四元数)。"""
        sid = self.m.site(self.ee_sites[p]).id
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, self.d.site_xmat[sid])
        return self.d.site_xpos[sid].copy(), quat

    def measured(self, p):
        """实测的手 / 躯干位置。"""
        if p == TORSO:
            return self.d.xpos[self.m.body(self.robot.torso_body).id].copy()
        return self.hand_pose(p)[0]

    def _sync_ref(self):
        """准备 WBC 计算用的构型 self.ref：先复制实测状态；reference 模式下受控关节换成
        上一周期的指令（限制在实测 ± cmd_lead_max 内）。"""
        self.ref.qpos[:] = self.d.qpos
        self.ref.qvel[:] = 0.0
        if self.opt.command_base == "reference" and self.last_cmd is not None:
            q = self.read_q()
            self.ref.qpos[self.wbc.qadr] = np.clip(self.last_cmd, q - self.opt.cmd_lead_max,
                                                   q + self.opt.cmd_lead_max)
        mujoco.mj_kinematics(self.m, self.ref)
        mujoco.mj_comPos(self.m, self.ref)

    # ---- 初始化
    def initialize(self):
        self._sync_ref()
        now = self.time()
        self.q_home = self.read_q()
        self.d.ctrl[self.act] = self.q_home
        for p in (LEFT, RIGHT):
            pos, quat = self.hand_pose(p)
            self.parts[p].home, self.parts[p].home_ori = pos, quat
        self.parts[TORSO].home = self.measured(TORSO)
        self.torso_y = self.parts[TORSO].home[1]
        th = self.parts[TORSO].home
        box = self.opt.torso_box if self.opt.torso_box is not None else self.robot.torso_box
        if box is not None:
            self.box = tuple(box)
        else:                            # 以初始躯干位置为中心自动给出
            bx, fx, dz, uz = self.opt.torso_box_margin or self.robot.torso_box_margin
            self.box = (th[0] - bx, th[0] + fx, th[2] - dz, th[2] + uz)
        x_min, x_max, z_min, z_max = self.box
        if not (x_min <= th[0] <= x_max and z_min <= th[2] <= z_max):
            raise RuntimeError(f"initial torso [{th[0]}, {th[2]}] outside safety box {self.box}")
        print(f"[wbc] torso safety box x [{x_min:.3f}, {x_max:.3f}] z [{z_min:.3f}, {z_max:.3f}]")
        for p in self.parts:
            p.seg.hold(p.home, p.home_ori, now)
            p.ghost = p.home.copy()
        self.mode = mode_from_number(2)
        self.apply_mode(self.mode, force=True)

    # ---- 模式切换（无扰）
    def request_mode(self, nxt):
        if self.moving_home():
            return
        self.apply_mode(nxt, False)

    def begin_drag(self, part):
        self.apply_mode(on_torso_target_event(self.mode) if part == TORSO
                        else on_hand_target_event(self.mode), False)
        self.parts[part].ghost = self.parts[part].seg.goal.copy()
        if part != TORSO and self.link_hands:
            other = RIGHT if part == LEFT else LEFT
            self.parts[other].ghost = self.parts[other].seg.goal.copy()

    def cancel_drag(self, part):
        self.parts[part].ghost = self.parts[part].seg.goal.copy()
        if part != TORSO and self.link_hands:
            other = RIGHT if part == LEFT else LEFT
            self.parts[other].ghost = self.parts[other].seg.goal.copy()
        print(f"[target] {PART_NAMES[part]} drag cancelled")

    def drag_to(self, part, p):
        target = self.clamp_torso(p, self.opt.torso_margin) if part == TORSO else np.array(p, float)
        delta = target - self.parts[part].ghost
        self.parts[part].ghost = target
        if part != TORSO and self.link_hands:
            other = RIGHT if part == LEFT else LEFT
            self.parts[other].ghost = self.parts[other].ghost + delta

    def commit_drag(self, part):
        if not self.is_tracked(part):
            print(f"[target] {PART_NAMES[part]} is not tracked in mode {mode_number(self.mode)}, "
                  f"drag ignored", file=sys.stderr)
            self.cancel_drag(part)
            return
        now = self.time()
        fmt = lambda g: "[" + ", ".join(f"{x:.4f}" for x in g) + "]"
        if part != TORSO and self.link_hands:
            print(f"[target] both hands (linked, dragged {PART_NAMES[part]}) -> left "
                  f"{fmt(self.parts[LEFT].ghost)} right {fmt(self.parts[RIGHT].ghost)} "
                  f"mode {mode_number(self.mode)}")
        else:
            print(f"[target] {PART_NAMES[part]} -> {fmt(self.parts[part].ghost)} "
                  f"mode {mode_number(self.mode)}")
        if part == TORSO:
            self.commit_torso(self.parts[TORSO].ghost, now)
        elif self.link_hands:
            self.commit_hands(self.parts[LEFT].ghost, self.parts[LEFT].seg.ori_goal,
                              self.parts[RIGHT].ghost, self.parts[RIGHT].seg.ori_goal, now)
        else:
            self.commit_hand(part, self.parts[part].ghost, self.parts[part].seg.ori_goal, now)

    def place_hands_forward(self):
        if self.moving_home():
            return
        self.apply_mode(on_hand_target_event(self.mode), False)
        now = self.time()
        goal = []
        for p in (LEFT, RIGHT):
            g = self.parts[p].home.copy()
            g[0] += self.opt.place_dx
            self.parts[p].ghost = g.copy()
            goal.append(g)
        self.commit_hands(goal[0], self.parts[LEFT].seg.ori_goal,
                          goal[1], self.parts[RIGHT].seg.ori_goal, now)
        print(f"[P] place point: left {np.round(goal[0], 4)} right {np.round(goal[1], 4)} "
              f"mode {mode_number(self.mode)}")

    def go_to_stress_point(self):
        if self.moving_home():
            return
        now = self.time()
        self.apply_mode(mode_from_number(2), False)
        # 双手：初始位置前方 stress_hand_dx（手臂伸不到）；躯干：初始位置
        stress_dx = self.opt.stress_hand_dx
        dx = np.array([self.robot.stress_hand_dx if stress_dx is None else stress_dx, 0.0, 0.0])
        left, right = self.parts[LEFT].home + dx, self.parts[RIGHT].home + dx
        self.parts[TORSO].ghost = self.clamp_torso(self.parts[TORSO].home, self.opt.torso_margin)
        self.commit_torso(self.parts[TORSO].ghost, now)
        self.parts[LEFT].ghost, self.parts[RIGHT].ghost = left.copy(), right.copy()
        self.commit_hands(left, self.parts[LEFT].seg.ori_goal,
                          right, self.parts[RIGHT].seg.ori_goal, now)
        print(f"[B] stress test point: left {np.round(left, 4)} right {np.round(right, 4)} "
              f"torso {np.round(self.parts[TORSO].ghost, 4)}  (mode 2)")

    def return_home(self):
        """R：所有关节（包括冻结部分）在关节空间按五次多项式回初始关节角（不经过 HQP），
        同时把手和躯干的目标更新为初始位姿；到位后 HQP 以这些目标继续，模式不变。"""
        if self.moving_home():
            return
        now = self.time()
        q = self.read_q()
        dist = float(np.max(np.abs(self.q_home - q)))
        duration = max(self.opt.t_min, 1.875 * dist / self.opt.movej_vmax)
        self.movej = {"start": q, "t0": now, "duration": duration}
        for p in range(3):
            self.parts[p].seg.hold(self.parts[p].home,
                                   IDENTITY_QUAT if p == TORSO else self.parts[p].home_ori, now)
            self.parts[p].ghost = self.parts[p].home.copy()
            self.reset_reachability(p)
        self.torso_y = self.parts[TORSO].home[1]
        print(f"[R] moveJ to initial posture: {duration:.2f} s "
              f"(max joint distance {math.degrees(dist):.2f} deg)")

    def moving_home(self):
        return self.movej is not None

    def move_torso_to(self, goal):
        """供 --demo 使用，等同拖躯干。返回实际下发的（夹到安全盒内的）目标。"""
        self.apply_mode(on_torso_target_event(self.mode), False)
        self.parts[TORSO].ghost = self.clamp_torso(goal, self.opt.torso_margin)
        self.commit_torso(self.parts[TORSO].ghost, self.time())
        return self.parts[TORSO].ghost.copy()

    def toggle_link(self):
        self.link_hands = not self.link_hands

    # ---- 每个控制周期
    def step(self):
        now = self.time()
        if self.movej is not None:
            self.step_movej(now)
            return
        self._sync_ref()
        q = self.read_q()
        self._set_tasks(now)
        self.update_arm_sigma()           # 开了 --adaptive-lambda 时同时重算 lambda

        t0 = time.perf_counter()
        qd, _ = self.wbc.step()
        self.solve_ms = 1e3 * (time.perf_counter() - t0)
        res = self.wbc.last_result
        self.degraded = bool(res.degraded)
        self.degraded_steps += int(self.degraded)
        if not res.ok or not np.all(np.isfinite(qd)):
            self.fails += 1
            cmd = q                        # 失败时保持当前关节
        else:
            cmd = np.clip(self._command_base(q) + qd * self.dt, self.wbc.lower, self.wbc.upper)
        self._apply(cmd)
        self.update_errors()
        if self.trace is not None:
            self.write_trace_row(cmd)
        self.maybe_print_tracking()

    def _command_base(self, q):
        """关节指令的累加起点（见 --command-base）。"""
        if self.opt.command_base == "reference":
            return self.wbc.get_joint_positions()      # 参考构型（已在 _sync_ref 中限幅）
        if self.opt.command_base == "measured" or self.last_cmd is None:
            return q
        return np.clip(self.last_cmd, q - self.opt.cmd_lead_max, q + self.opt.cmd_lead_max)

    def _apply(self, cmd):
        self.last_cmd = cmd
        self.d.ctrl[self.act] = cmd
        if self.opt.kinematic:             # 运动学模式：关节直接到达指令
            self.d.qpos[self.wbc.qadr] = cmd
            self.d.qvel[:] = 0.0
            self.d.time += self.substeps * self.m.opt.timestep
        else:
            for _ in range(self.substeps):
                mujoco.mj_step(self.m, self.d)
        mujoco.mj_forward(self.m, self.d)  # 让位姿、雅可比对应步进后的状态

    def step_movej(self, now):
        mj = self.movej
        tau = min(max((now - mj["t0"]) / mj["duration"], 0.0), 1.0)
        s = tau ** 3 * (10.0 - 15.0 * tau + 6.0 * tau * tau)
        cmd = mj["start"] + s * (self.q_home - mj["start"])
        self._apply(cmd)
        self.errors[:] = 0.0                # moveJ 期间不按末端目标判误差 / 不可达
        if self.trace is not None:
            self.write_trace_row(cmd)
        self.maybe_print_tracking()
        if tau >= 1.0:
            self.movej = None
            self.sync_planner(self.time())
            print(f"[R] at initial posture, HQP resumes (mode {mode_number(self.mode)})")

    def _set_tasks(self, now):
        """按模式设置任务开关、目标、前馈和冻结。"""
        w = self.wbc
        hands_track = self.mode.hands == HAND_TRACK
        for p, site in ((LEFT, self.ee_sites[0]), (RIGHT, self.ee_sites[1])):
            t = w.task(site)
            t.enabled = hands_track
            if hands_track:
                seg = self.parts[p].seg
                t.set_target(seg.sample(now), seg.sample_ori(now), vel=seg.velocity(now))
        torso = w.task("torso")
        target = self.torso_target(now)
        torso.enabled = target is not None
        if target is not None:
            torso.set_target(target[0], IDENTITY_QUAT, vel=target[1])
            torso.kp_pos = target[2]

    def torso_target(self, now):
        """本周期躯干任务的 (目标位置, 前馈速度, kp) ；None 表示没有躯干任务。"""
        if self.mode.torso == TORSO_TRACK:
            seg = self.parts[TORSO].seg
            p = seg.sample(now)
            if self.opt.torso_leash > 0.0:      # 目标牵引限幅（C++ 版的 --torso-leash）
                cur = self.measured(TORSO)
                d = p - cur
                if np.linalg.norm(d) > self.opt.torso_leash:
                    p = cur + d * (self.opt.torso_leash / np.linalg.norm(d))
            return p, seg.velocity(now), None
        if self.mode.torso == TORSO_FREE and self.opt.torso_return_gain > 0.0:
            # 模式1 回中：每个周期纠正偏差的 k 倍，即速度增益 k/dt
            return self.parts[TORSO].home, None, self.opt.torso_return_gain / self.dt
        return None

    def _torso_box(self, c):
        """躯干安全盒：x、z 两行线性化约束（默认为最高层的软约束：先最小化违反量）。"""
        p, _ = c.get_frame_pose("torso")
        J = c.frame_jacobian("torso")[[0, 2]]
        x_min, x_max, z_min, z_max = self.box
        o = self.opt
        return wf.linearized_constraint("torso_box", J, p[[0, 2]], [x_min, z_min], [x_max, z_max],
                                        self.dt, beta=o.box_beta, soft=o.box_soft,
                                        w_lin=o.box_w_lin, w_quad=o.box_w_quad,
                                        priority=o.box_priority)

    # ---- --print-track
    @staticmethod
    def num4(v):
        if abs(v) < 0.5e-4:
            v = 0.0
        return f"{v: .4f}"

    def vec4(self, v):
        return "[" + " ".join(self.num4(x) for x in v) + "]"

    def maybe_print_tracking(self):
        if not self.opt.print_track or self.time() < self.next_print_time:
            return
        self.next_print_time = self.time() + self.opt.print_period
        lines = [f"[track] t={self.time():.4f} s  mode {mode_number(self.mode)}"
                 + ("  (R moveJ, targets = initial poses)" if self.movej else "")]

        def line(name, target, state, why_none, extra=""):
            if target is None:
                lines.append(f"  {name:<10} target ({why_none})  state {self.vec4(state)}")
                return
            e = (state - target) * 1000.0
            lines.append(f"  {name:<10} target {self.vec4(target)}  state {self.vec4(state)}  "
                         f"err mm {self.vec4(e)} |e| [{self.num4(np.linalg.norm(e))}]{extra}")

        now = self.time()
        for p in (LEFT, RIGHT):
            if self.mode.hands == HAND_TRACK:
                seg = self.parts[p].seg
                ang = np.degrees(np.linalg.norm(quat_error(self.hand_pose(p)[1], seg.sample_ori(now))))
                line(PART_NAMES[p], seg.sample(now), self.measured(p), "",
                     f"  ori err [{self.num4(ang)}] deg")
            else:
                line(PART_NAMES[p], None, self.measured(p), "no hand task: arms frozen")
        tt = self.torso_target(now) if self.movej is None else None
        line(PART_NAMES[TORSO], None if tt is None else tt[0], self.measured(TORSO),
             "no torso task: frozen" if self.mode.torso == TORSO_FROZEN else "no torso task: free")
        lines.append(f"  arm sigma_min L/R [{self.num4(self.arm_sigma[0])}] "
                     f"[{self.num4(self.arm_sigma[1])}] m/rad  lambda [{self.num4(self.wbc.lambda_reg)}]")
        print("\n".join(lines), flush=True)

    # ---- --data-record（列与 C++ 版相同，tools/plot_trace.py 可直接画）
    def enable_trace(self, path):
        self.trace = open(path, "w")
        cols = ["t", "mode"] + [f"cmd_{n}" for n in self.joints] + \
               [f"meas_{n}" for n in self.joints]
        for part in ("left", "right", "torso"):
            for kind in ("target", "fk"):
                cols += [f"{part}_{kind}_{a}" for a in "xyz"]
        self.trace.write(",".join(cols) + "\n")

    def close_trace(self):
        if self.trace is not None:
            self.trace.close()
            self.trace = None

    def write_trace_row(self, cmd):
        now = self.time()
        meas = self.read_q()
        vals = [f"{now:.9g}", str(mode_number(self.mode))]
        vals += [f"{x:.9g}" for x in cmd] + [f"{x:.9g}" for x in meas]

        def put(v):
            vals.extend(["nan"] * 3 if v is None else [f"{x:.9g}" for x in v])
        hands = self.mode.hands == HAND_TRACK
        put(self.parts[LEFT].seg.sample(now) if hands else None)
        put(self.measured(LEFT))
        put(self.parts[RIGHT].seg.sample(now) if hands else None)
        put(self.measured(RIGHT))
        tt = self.torso_target(now)
        put(None if tt is None else tt[0])
        put(self.measured(TORSO))
        self.trace.write(",".join(vals) + "\n")
        self.trace_rows += 1
        if self.trace_rows % 20 == 0:
            self.trace.flush()             # 绘图脚本 --live 实时读取

    # ---- 显示用
    def command(self, p):
        return self.parts[p].seg.sample(self.time())

    def ghost(self, p):
        return self.parts[p].ghost

    def handle(self, p):
        """拾取 / 拖拽的锚点：跟踪中的部分用目标点，否则用实际位置。"""
        return self.parts[p].ghost if self.is_tracked(p) else self.measured(p)

    def is_tracked(self, p):
        return self.mode.torso == TORSO_TRACK if p == TORSO else self.mode.hands == HAND_TRACK

    def is_frozen(self, p):
        return self.mode.torso == TORSO_FROZEN if p == TORSO else self.mode.hands == HAND_FROZEN

    def remaining_motion(self):
        now = self.time()
        r = self.movej["t0"] + self.movej["duration"] - now if self.movej else 0.0
        for p in range(3):
            if self.is_tracked(p):
                r = max(r, self.parts[p].seg.remaining(now))
        return r

    def home(self, p):
        return self.parts[p].home

    # ---- 内部
    def arm_sigma_min(self, s):
        """手位置对本臂 7 个关节的雅可比（3×7）的最小奇异值 (m/rad)，接近奇异时趋近 0。"""
        J = self.wbc.frame_jacobian(self.ee_sites[s])[:3, self.arm_cols[s]]
        return float(np.linalg.svd(J, compute_uv=False).min())

    def update_arm_sigma(self):
        for s in (0, 1):
            self.arm_sigma[s] = self.arm_sigma_min(s)
        if self.opt.adaptive_lambda <= 0.0:
            return
        sigma = float(self.arm_sigma.min())
        extra = 0.0
        if sigma < self.opt.adaptive_lambda_sigma:
            r = 1.0 - sigma / self.opt.adaptive_lambda_sigma
            extra = self.opt.adaptive_lambda * r * r
        self.wbc.lambda_reg = self.lambda_base + extra

    def clamp_torso(self, p, margin):
        p = np.array(p, float)
        x_min, x_max, z_min, z_max = self.box
        p[0] = min(max(p[0], x_min + margin), x_max - margin)
        p[1] = self.torso_y
        p[2] = min(max(p[2], z_min + margin), z_max - margin)
        return p

    def reset_reachability(self, part):
        self.parts[part].over_error_since = -1.0
        self.parts[part].unreachable = False

    def commit_torso(self, goal, now):
        seg = self.parts[TORSO].seg
        seg.retarget(goal, IDENTITY_QUAT, now, seg.required_duration(
            goal, IDENTITY_QUAT, now, self.opt.torso_vmax, self.opt.hand_wmax, self.opt.t_min))
        self.reset_reachability(TORSO)

    def commit_hand(self, part, goal, ori, now):
        seg = self.parts[part].seg
        seg.retarget(goal, ori, now, seg.required_duration(
            goal, ori, now, self.opt.hand_vmax, self.opt.hand_wmax, self.opt.t_min))
        self.reset_reachability(part)

    def commit_hands(self, lg, lo, rg, ro, now):
        """双手一起下发：取两者所需时长的较大值，保证同时出发、同时到达。"""
        ls, rs = self.parts[LEFT].seg, self.parts[RIGHT].seg
        o = self.opt
        dur = max(ls.required_duration(lg, lo, now, o.hand_vmax, o.hand_wmax, o.t_min),
                  rs.required_duration(rg, ro, now, o.hand_vmax, o.hand_wmax, o.t_min))
        ls.retarget(lg, lo, now, dur)
        rs.retarget(rg, ro, now, dur)
        self.reset_reachability(LEFT)
        self.reset_reachability(RIGHT)

    @staticmethod
    def is_valid_mode(m):
        return m.hands == HAND_TRACK or m.torso == TORSO_TRACK   # 手冻结时必须有躯干任务

    def apply_mode(self, nxt, force):
        if not force and nxt == self.mode:
            return
        if not self.is_valid_mode(nxt):
            print("[mode] rejected invalid combination (hands frozen without torso task)",
                  file=sys.stderr)
            return
        now = self.time()
        hands_were_track = not force and self.mode.hands == HAND_TRACK
        torso_was_track = not force and self.mode.torso == TORSO_TRACK
        # 无扰：新进入 TRACK 的部分，从 WBC 当前所用构型（self.ref）下的位姿出发
        self._sync_ref()
        if nxt.hands == HAND_TRACK and not hands_were_track:
            for p in (LEFT, RIGHT):
                pos, quat = self.wbc.get_frame_pose(self.ee_sites[p])
                self.parts[p].seg.hold(pos, quat, now)
                self.parts[p].ghost = pos.copy()
                self.reset_reachability(p)
        if nxt.torso == TORSO_TRACK and not torso_was_track:
            pos = self.wbc.get_frame_pose("torso")[0]
            self.torso_y = pos[1]
            anchor = self.clamp_torso(pos, 0.0)
            self.parts[TORSO].seg.hold(anchor, IDENTITY_QUAT, now)
            self.reset_reachability(TORSO)
            safe = self.clamp_torso(anchor, self.opt.torso_margin)   # 在边界余量带内：插值退回
            self.parts[TORSO].ghost = safe
            if np.linalg.norm(safe - anchor) > 1e-9:
                self.commit_torso(safe, now)
        self.mode = nxt
        self.sync_planner(now)
        print(f"[mode] -> {mode_name(self.mode)}")

    def sync_planner(self, now):
        """按当前模式设置冻结组（任务目标每个周期在 _set_tasks 中设置）。"""
        frozen = []
        if self.mode.torso == TORSO_FROZEN:
            frozen += list(self.robot.torso_joints)
        if self.mode.hands == HAND_FROZEN:
            frozen += self.robot.arm_joints()
        self.wbc.set_frozen(frozen)
        # 躯干自由时，它动多少只由 W_p 决定；躯干力臂长，代价要比手臂大才不会抢着动。
        # 其他模式躯干要么跟踪目标、要么冻结，用默认代价
        free = self.mode.torso == TORSO_FREE
        self.wbc.set_joint_cost(self.robot.torso_joints, self.opt.torso_cost if free else None)

    def update_errors(self):
        now = self.time()
        for p in range(3):
            s = self.parts[p]
            if not self.is_tracked(p):
                self.errors[p] = 0.0
                s.unreachable = False
                s.over_error_since = -1.0
                continue
            diff = self.measured(p) - s.seg.sample(now)
            if p == TORSO:
                diff[1] = 0.0                 # 躯干 y 不是任务目标
            self.errors[p] = float(np.linalg.norm(diff))
            if self.errors[p] > self.opt.unreachable_error:
                if s.over_error_since < 0:
                    s.over_error_since = now
                s.unreachable = now - s.over_error_since > 1.0
            else:
                s.over_error_since = -1.0
                s.unreachable = False


# ---------------------------------------------------------------- 无窗口自检
def run_demo(ctl: InteractiveController, place_dx):
    def run_for(seconds):
        end = ctl.time() + seconds
        max_err = 0.0
        while not g_stop and ctl.time() < end:
            ctl.step()
            max_err = max(max_err, ctl.errors[LEFT], ctl.errors[RIGHT])
        return max_err

    settle = 2.0
    run_motion = lambda: run_for(ctl.remaining_motion() + settle)
    hand_final_error = lambda: max(ctl.errors[LEFT], ctl.errors[RIGHT])

    # A：模式4（躯干冻结，仅手臂）伸向放置点
    ctl.request_mode(mode_from_number(4))
    ctl.place_hands_forward()
    run_motion()
    err_arm_only = hand_final_error()
    reach_arm_only = ctl.measured(LEFT)[0] - ctl.home(LEFT)[0]
    ctl.return_home()
    run_motion()

    # B：模式1（全身协同，躯干自由）伸向同一放置点
    ctl.request_mode(mode_from_number(1))
    ctl.place_hands_forward()
    run_motion()
    err_wbc = hand_final_error()
    reach_wbc = ctl.measured(LEFT)[0] - ctl.home(LEFT)[0]
    ctl.return_home()
    run_motion()

    # C：模式2（双手固定，躯干下降 8 cm、前移 8 cm、再回来）
    ctl.request_mode(mode_from_number(2))
    torso0 = ctl.move_torso_to(ctl.home(TORSO))
    run_motion()
    hqp_hand_err = 0.0
    c_moves = []
    for delta in ([0, 0, -0.08], [0.08, 0, -0.08], [0, 0, 0]):
        sent = ctl.move_torso_to(torso0 + np.array(delta))
        d = sent - torso0
        c_moves.append(f"({d[0] * 100:.4f}, {d[2] * 100:.4f})")
        if np.linalg.norm(d - np.array([delta[0], 0.0, delta[2]])) > 1e-3:
            print(f"[demo] torso target clipped by safety box: requested dx/dz "
                  f"{delta[0]}/{delta[2]}, sent {d[0]:.4f}/{d[2]:.4f}")
        hqp_hand_err = max(hqp_hand_err, run_motion())

    # D：模式3（手臂冻结，躯干下降 5 cm，手随身体移动）
    ctl.request_mode(mode_from_number(3))
    hand_before = ctl.measured(LEFT)
    d_sent = ctl.move_torso_to(torso0 + np.array([0, 0, -0.05])) - torso0
    run_motion()
    hand_moved = float(np.linalg.norm(ctl.measured(LEFT) - hand_before))
    torso_err_mode3 = ctl.errors[TORSO]

    tol = 0.005
    vs = lambda err: (f"（参考阈值 {tol * 1000:.1f} mm：" + ("超出" if err > tol else "未超出") + "）")
    limit = ctl.opt.unreachable_error * 1000
    a_note = (f"（超过不可达阈值 {limit:.1f} mm，符合预期：仅手臂够不到）"
              if err_arm_only * 1000 > limit else
              f"（未超过不可达阈值 {limit:.1f} mm：仅手臂也够到了）")
    print("\n===== demo result =====")
    print(f"A 模式4 仅手臂 -> 放置点(+{place_dx:.4f} m): 实际前伸 {reach_arm_only:.4f} m, "
          f"末端剩余误差 {err_arm_only * 1000:.4f} mm{a_note}")
    print(f"B 模式1 全身协同 -> 同一放置点:     实际前伸 {reach_wbc:.4f} m, "
          f"末端剩余误差 {err_wbc * 1000:.4f} mm{vs(err_wbc)}")
    print(f"C 模式2 躯干指令位移 (dx, dz) cm: {' '.join(c_moves)}, 双手末端最大误差 "
          f"{hqp_hand_err * 1000:.4f} mm{vs(hqp_hand_err)}")
    print(f"D 模式3 手臂冻结, 躯干指令下降 {-d_sent[2] * 100:.4f} cm: 手随身体移动 "
          f"{hand_moved * 1000:.4f} mm, 躯干跟踪误差 {torso_err_mode3 * 1000:.4f} mm{vs(torso_err_mode3)}")
    print(f"QP 失败次数 {ctl.fails}，降级步数 {ctl.degraded_steps}")
    print("注：参考阈值仅作提示，不影响退出码（退出码只看 QP 失败次数）。")
    return 4 if ctl.fails > 0 else 0


# ---------------------------------------------------------------- 窗口与交互
class Viewer:
    AXIS_KEY_HOLD_SEC = 0.6          # 远程桌面：0.6 s 内按过的轴键也算按住
    ROBOT_TRANSPARENCY = 0.4
    MARKER_PULL = 0.4                # 目标球沿视线拉向相机的距离 (m)，保证不被机身挡住
    MAX_STEPS_PER_FRAME = 40

    def __init__(self, ctl: InteractiveController):
        import glfw
        self.glfw = glfw
        self.ctl = ctl
        self.m = ctl.m
        self.window = None
        self.left_down = self.right_down = self.middle_down = False
        self.last_x = self.last_y = 0.0
        self.dragging = -1
        self.drag_vertical = False
        self.drag_axis = -1
        self.axis_key_down = [False, False, False]
        self.axis_key_last_press = [-1e9, -1e9, -1e9]
        self.hover = -1
        self.show_help = True
        self.robot_transparent = False
        self.slower_than_realtime = False

    # ---- 几何：鼠标射线、拾取、拖动约束
    def _camera(self):
        head, fwd, up = np.zeros(3), np.zeros(3), np.zeros(3)
        mujoco.mjv_cameraInModel(head, fwd, up, self.scene)
        return head, fwd, up

    def _proj(self):
        w, h = self.glfw.get_window_size(self.window)
        aspect = w / h if h > 0 else 1.0
        t = math.tan(0.5 * math.radians(self.m.vis.global_.fovy))
        return w, h, aspect, t

    def ray(self, mx, my):
        head, f, u = self._camera()
        w, h, aspect, t = self._proj()
        r = np.cross(f, u)
        r /= np.linalg.norm(r)
        nx = 2.0 * mx / max(1, w) - 1.0
        ny = 1.0 - 2.0 * my / max(1, h)
        d = f + nx * t * aspect * r + ny * t * u
        return head, d / np.linalg.norm(d)

    def pick(self, mx, my):
        """屏幕上离鼠标最近、且在 25 像素内的目标球。"""
        c, f, u = self._camera()
        w, h, aspect, t = self._proj()
        r = np.cross(f, u)
        r /= np.linalg.norm(r)
        best, best_d = -1, 25.0
        for p in range(3):
            v = self.ctl.handle(p) - c
            z = v @ f
            if z <= 1e-3:
                continue
            sx = (v @ r / (z * t * aspect) + 1.0) * 0.5 * w
            sy = (1.0 - v @ u / (z * t)) * 0.5 * h
            d = math.hypot(sx - mx, sy - my)
            if d < best_d:
                best, best_d = p, d
        return best

    def plane_hit(self, mx, my, anchor, vertical):
        o, d = self.ray(mx, my)
        if vertical:
            _, fwd, _ = self._camera()
            n = np.array([fwd[0], fwd[1], 0.0])
            if np.linalg.norm(n) < 1e-6:
                return None
            n /= np.linalg.norm(n)
        else:
            n = np.array([0.0, 0.0, 1.0])
        denom = n @ d
        if abs(denom) < 1e-6:
            return None
        s = n @ (anchor - o) / denom
        return None if s <= 0.0 else o + s * d

    def held_axis(self):
        keys = (self.glfw.KEY_X, self.glfw.KEY_Y, self.glfw.KEY_Z)
        now = self.glfw.get_time()
        best, best_t = -1, -1e18
        for a in range(3):
            held = (self.axis_key_down[a] or self.glfw.get_key(self.window, keys[a]) == self.glfw.PRESS
                    or now - self.axis_key_last_press[a] < self.AXIS_KEY_HOLD_SEC)
            if held and self.axis_key_last_press[a] >= best_t:
                best, best_t = a, self.axis_key_last_press[a]
        return best

    def axis_hit(self, mx, my, anchor, axis):
        o, d = self.ray(mx, my)
        a = np.eye(3)[axis]
        w = anchor - o
        b = a @ d
        denom = 1.0 - b * b
        if denom < 1e-4:
            return None                      # 该轴几乎正对视线
        return anchor + ((b * (d @ w) - a @ w) / denom) * a

    def drag_hit(self, mx, my, anchor, axis):
        if axis >= 0:
            return self.axis_hit(mx, my, anchor, axis)
        hit = self.plane_hit(mx, my, anchor, self.drag_vertical)
        return None if hit is None else self.constrain(self.dragging, anchor, hit)

    def constrain(self, part, anchor, hit):
        p = anchor.copy()
        if self.drag_vertical:
            p[2] = hit[2]
        else:
            p[0] = hit[0]
            if part != TORSO:
                p[1] = hit[1]                 # 躯干只允许前后
        return p

    # ---- 画目标球
    def add_sphere(self, pos, radius, rgba):
        scn = self.scene
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, radius),
                            np.asarray(pos, float), np.eye(3).ravel(), np.asarray(rgba, np.float32))
        g.category = mujoco.mjtCatBit.mjCAT_DECOR
        scn.ngeom += 1

    def add_marker(self, pos, radius, rgba):
        c, _, _ = self._camera()
        dist = float(np.linalg.norm(pos - c))
        s = min(self.MARKER_PULL, 0.5 * dist) / dist if dist > 1e-6 else 0.0
        self.add_sphere(pos + s * (c - pos), radius * (1.0 - s), rgba)

    def add_markers(self):
        green, yellow = (0.1, 0.85, 0.2, 1.0), (1.0, 0.85, 0.1, 0.45)
        red, blue, grey = (0.95, 0.15, 0.1, 0.9), (0.2, 0.45, 1.0, 0.45), (0.7, 0.7, 0.7, 0.45)
        orange = (1.0, 0.55, 0.1, 1.0)
        ctl = self.ctl
        for p in range(3):
            base = 0.035 if p == TORSO else 0.028
            scale = 1.4 if p in (self.hover, self.dragging) else 1.0
            if ctl.is_tracked(p):
                self.add_marker(ctl.command(p), base * scale, orange if p == TORSO else green)
                show_ghost = (self.dragging == p or np.linalg.norm(ctl.ghost(p) - ctl.command(p)) > 1e-3
                              or ctl.parts[p].unreachable)
                if show_ghost:
                    self.add_marker(ctl.ghost(p), base * 1.15 * scale,
                                    red if ctl.parts[p].unreachable else yellow)
            else:
                self.add_marker(ctl.measured(p), base * scale, blue if ctl.is_frozen(p) else grey)

    def make_robot_transparent(self):
        for i in range(self.scene.ngeom):
            g = self.scene.geoms[i]
            if g.objtype != mujoco.mjtObj.mjOBJ_GEOM or g.objid < 0 or self.m.geom_bodyid[g.objid] == 0:
                continue
            g.rgba[3] *= 1.0 - self.ROBOT_TRANSPARENCY
            g.transparent = 1

    def render(self):
        ctl = self.ctl
        mujoco.mjv_updateScene(self.m, ctl.d, self.opt, None, self.cam,
                               mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        if self.robot_transparent:
            self.make_robot_transparent()
        self.add_markers()
        fw, fh = self.glfw.get_framebuffer_size(self.window)
        viewport = mujoco.MjrRect(0, 0, fw, fh)
        mujoco.mjr_render(viewport, self.scene, self.con)
        if ctl.moving_home():
            tag = "   [R: moveJ home, keys locked]"
        elif self.dragging < 0:
            tag = ""
        else:
            tag = {0: "   [dragging along X: keys locked]", 1: "   [dragging along Y: keys locked]",
                   2: "   [dragging along Z: keys locked]"}.get(self.drag_axis, "   [dragging: keys locked]")
        m = ctl.mode
        res = ctl.wbc.last_result
        levels = ""
        if res is not None and res.levels:
            levels = "  levels " + " ".join(f"{l.priority}:{l.status[:6]}" for l in res.levels)
        status = (
            f"Mode {mode_name(m)}   solver {ctl.opt.solver}\n"
            f"Hands: {m.hands.upper()}   Torso: {m.torso.upper()}   Link hands: "
            f"{'ON' if ctl.link_hands else 'OFF'}{tag}\n"
            f"Err L/R/T: {ctl.errors[0] * 1000:.1f} / {ctl.errors[1] * 1000:.1f} / "
            f"{ctl.errors[2] * 1000:.1f} mm"
            f"{' [UNREACHABLE]' if any(p.unreachable for p in ctl.parts) else ''}\n"
            f"Arm sigma_min L/R: {ctl.arm_sigma[0]:.3f} / {ctl.arm_sigma[1]:.3f} m/rad   "
            f"lambda: {ctl.wbc.lambda_reg:.4g}\n"
            f"Time {ctl.time():.1f}s  QP fails {ctl.fails}  degraded {ctl.degraded_steps}"
            f"{'  (degraded now)' if ctl.degraded else ''}  solve {ctl.solve_ms:.2f} ms{levels}"
            f"{'  (slower than real time)' if self.slower_than_realtime else ''}"
            f"{'  [robot 40% transparent]' if self.robot_transparent else ''}")
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                           viewport, status, None, self.con)
        if self.show_help:
            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, viewport,
                "Ctrl+drag ball\nCtrl+Shift+drag\nHold X/Y/Z + drag\nRelease\n1 / 2 / 3 / 4\nL\nP\nB\nR\nT\n"
                "Mouse / wheel\nH / Esc",
                "move horizontally (torso: fore/aft)\nmove vertically\nmove along world axis only\n"
                "plan + HQP to target\nWBC hands | HQP hands+torso | torso only | arms only\n"
                "link hands (tray)\nhands to place point\nhands + torso to stress test point\n"
                "moveJ all joints home\nrobot transparent 40% on/off\nrotate, pan, zoom\nhelp / quit",
                self.con)
        self.glfw.swap_buffers(self.window)

    # ---- 回调
    def _move_camera(self, action, reldx, reldy):
        """新版 MuJoCo 的 mjv_moveCamera 没有 scene 参数，旧版有。"""
        try:
            mujoco.mjv_moveCamera(self.m, action, reldx, reldy, self.cam)
        except TypeError:
            mujoco.mjv_moveCamera(self.m, action, reldx, reldy, self.scene, self.cam)

    def on_key(self, window, key, scancode, action, mods):
        glfw = self.glfw
        if g_debug_input:
            name = glfw.get_key_name(key, scancode)
            act = {glfw.PRESS: "press", glfw.RELEASE: "release"}.get(action, "repeat")
            print(f"[input] key {key} ({name or '?'}) scancode {scancode} {act} mods {mods} "
                  f"dragging {self.dragging}")
        if key in (glfw.KEY_X, glfw.KEY_Y, glfw.KEY_Z):
            a = {glfw.KEY_X: 0, glfw.KEY_Y: 1, glfw.KEY_Z: 2}[key]
            self.axis_key_down[a] = action != glfw.RELEASE
            if action != glfw.RELEASE:
                self.axis_key_last_press[a] = glfw.get_time()
        if action != glfw.PRESS:
            return
        ctl = self.ctl
        locked = self.dragging >= 0 or ctl.moving_home()
        mode_keys = (glfw.KEY_1, glfw.KEY_2, glfw.KEY_3, glfw.KEY_4,
                     glfw.KEY_L, glfw.KEY_P, glfw.KEY_B, glfw.KEY_R)
        if locked and key not in (glfw.KEY_H, glfw.KEY_T, glfw.KEY_ESCAPE):
            if key in mode_keys:
                print("[key] ignored while dragging; release the mouse first" if self.dragging >= 0
                      else "[key] ignored while R moves home (moveJ)")
            return
        if key in (glfw.KEY_1, glfw.KEY_2, glfw.KEY_3, glfw.KEY_4):
            ctl.request_mode(mode_from_number(key - glfw.KEY_0))
        elif key == glfw.KEY_L:
            ctl.toggle_link()
        elif key == glfw.KEY_P:
            ctl.place_hands_forward()
        elif key == glfw.KEY_B:
            ctl.go_to_stress_point()
        elif key == glfw.KEY_R:
            ctl.return_home()
        elif key == glfw.KEY_H:
            self.show_help = not self.show_help
        elif key == glfw.KEY_T:
            self.robot_transparent = not self.robot_transparent
        elif key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)

    def on_mouse_button(self, window, button, action, mods):
        glfw = self.glfw
        if g_debug_input:
            print(f"[input] mouse button {button} {'press' if action == glfw.PRESS else 'release'} "
                  f"mods {mods}")
        self.last_x, self.last_y = glfw.get_cursor_pos(window)
        press = action == glfw.PRESS
        if button == glfw.MOUSE_BUTTON_LEFT:
            self.left_down = press
        if button == glfw.MOUSE_BUTTON_RIGHT:
            self.right_down = press
        if button == glfw.MOUSE_BUTTON_MIDDLE:
            self.middle_down = press
        if (button == glfw.MOUSE_BUTTON_LEFT and press and (mods & glfw.MOD_CONTROL)
                and not self.ctl.moving_home()):
            part = self.pick(self.last_x, self.last_y)
            if part >= 0:
                self.ctl.begin_drag(part)          # 状态机：拖手 / 拖躯干事件
                self.dragging = part
                self.drag_vertical = bool(mods & glfw.MOD_SHIFT)
                self.drag_axis = self.held_axis()
        if button == glfw.MOUSE_BUTTON_LEFT and not press and self.dragging >= 0:
            self.ctl.commit_drag(self.dragging)    # 松开即执行
            self.dragging = -1

    def on_cursor(self, window, x, y):
        px, py = self.last_x, self.last_y
        dx, dy = x - px, y - py
        self.last_x, self.last_y = x, y
        _, height = self.glfw.get_window_size(window)
        if height <= 0:
            return
        if self.dragging >= 0:
            # 增量拖动：上一位置和当前位置在同一约束下的位移，加到目标点上（切换轴不跳）
            axis = self.held_axis()
            if axis != self.drag_axis:
                print("[drag] axis " + {0: "X", 1: "Y", 2: "Z"}.get(axis, "free"))
            self.drag_axis = axis
            anchor = self.ctl.ghost(self.dragging)
            a = self.drag_hit(px, py, anchor, axis)
            b = self.drag_hit(x, y, anchor, axis)
            if a is not None and b is not None:
                self.ctl.drag_to(self.dragging, anchor + (b - a))
            return
        if not (self.left_down or self.right_down or self.middle_down):
            self.hover = self.pick(x, y)
            return
        action = mujoco.mjtMouse.mjMOUSE_MOVE_V if self.middle_down else mujoco.mjtMouse.mjMOUSE_ROTATE_V
        self._move_camera(action, dx / height, dy / height)

    def on_focus(self, window, focused):
        if focused:
            return
        if self.dragging >= 0:                     # 失焦可能收不到松开事件：取消拖动
            self.ctl.cancel_drag(self.dragging)
            self.dragging = -1
        self.left_down = self.right_down = self.middle_down = False
        self.axis_key_down = [False, False, False]
        self.axis_key_last_press = [-1e9, -1e9, -1e9]

    def on_scroll(self, window, xoffset, yoffset):
        self._move_camera(mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.05 * yoffset)

    # ---- 主循环
    def run(self, sim_per_frame=None):
        """窗口主循环。sim_per_frame=None：仿真跟随墙钟（交互）；给定秒数时每帧固定推进
        这么多仿真时间，与渲染快慢无关（录制视频用，见 tools/record_readme_videos.py）。"""
        glfw = self.glfw
        if not glfw.init():
            print("GLFW init failed (no display?). Use --demo for headless check.", file=sys.stderr)
            return 5
        self.window = glfw.create_window(1280, 800, "HQP interactive", None, None)
        if not self.window:
            glfw.terminate()
            print("GLFW window creation failed", file=sys.stderr)
            return 5
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        self.scene = mujoco.MjvScene(self.m, maxgeom=4000)
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.con = mujoco.MjrContext(self.m, mujoco.mjtFontScale.mjFONTSCALE_150)
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[:] = self.ctl.home(TORSO) + [0.3, 0.0, -0.1]
        self.cam.distance = 2.4
        self.cam.azimuth = 150.0
        self.cam.elevation = -15.0
        glfw.set_key_callback(self.window, self.on_key)
        glfw.set_mouse_button_callback(self.window, self.on_mouse_button)
        glfw.set_cursor_pos_callback(self.window, self.on_cursor)
        glfw.set_scroll_callback(self.window, self.on_scroll)
        glfw.set_window_focus_callback(self.window, self.on_focus)

        # 单线程：每帧先把仿真推进到墙钟时间，再渲染。一帧内追不上墙钟时丢弃积压。
        wall_start = time.perf_counter()
        sim_start = self.ctl.time()
        dropped = 0.0
        while not g_stop and not glfw.window_should_close(self.window):
            if sim_per_frame is not None:
                frame_end = self.ctl.time() + sim_per_frame - 1e-9
                while self.ctl.time() < frame_end:
                    self.ctl.step()
                self.render()
                glfw.poll_events()
                continue
            target = time.perf_counter() - wall_start - dropped
            steps = 0
            while self.ctl.time() - sim_start < target and steps < self.MAX_STEPS_PER_FRAME:
                self.ctl.step()
                steps += 1
            self.slower_than_realtime = (steps >= self.MAX_STEPS_PER_FRAME
                                         and self.ctl.time() - sim_start < target)
            if self.slower_than_realtime:
                dropped += target - (self.ctl.time() - sim_start)
            self.render()
            glfw.poll_events()
        self.con.free()
        glfw.destroy_window(self.window)
        glfw.terminate()
        return 0


# ---------------------------------------------------------------- main
def parse_args(argv):
    """返回 (运行方式选项, Options)。参数的优先级：命令行 > --config 文件 > 默认值。"""
    opt = Options()
    ap = argparse.ArgumentParser(description="HQP / WBC interactive simulation")
    ap.add_argument("--robot", default="r1",
                    help=f"内置机器人 {list(robots.ROBOTS)}，或 JSON 配置文件（见 robots.py）")
    ap.add_argument("--base", action="store_true", help="底盘也参与规划（机器人有底盘关节时）")
    ap.add_argument("--demo", action="store_true", help="无窗口自检")
    ap.add_argument("--data-record", action="store_true", help="记录数据")
    ap.add_argument("--trace-dir", default=None, help="记录目录（同时开启 --data-record）")
    ap.add_argument("--debug-input", action="store_true", help="打印键盘 / 鼠标事件")
    ap.add_argument("--config", default=None, help="参数文件（JSON），键为参数名")
    ap.add_argument("--print-params", action="store_true", help="打印当前生效的全部参数后退出")
    ap.add_argument("--kp", type=float, default=argparse.SUPPRESS,
                    help="同时设置 --kp-pos 和 --kp-rot")
    add_arguments(ap, opt)
    a = ap.parse_args(argv)

    if a.config:
        with open(a.config, encoding="utf-8") as f:
            apply_dict(opt, json.load(f))
    apply_arguments(opt, a)
    if hasattr(a, "kp"):
        opt.wbc.kp_pos = opt.wbc.kp_rot = a.kp

    w = opt.wbc
    if opt.solver not in ("hqp", "wqp", "wls"):
        ap.error("--solver 只能是 hqp / wqp / wls")
    if opt.command_base not in ("reference", "cmd", "measured"):
        ap.error("--command-base 只能是 reference / cmd / measured")
    positive = dict(kp_pos=w.kp_pos, kp_rot=w.kp_rot, lambda_reg=w.lambda_reg, dq_max=w.max_joint_vel,
                    control_dt=opt.control_dt, place_dx=opt.place_dx, hand_vmax=opt.hand_vmax,
                    torso_vmax=opt.torso_vmax, movej_vmax=opt.movej_vmax,
                    adaptive_lambda_sigma=opt.adaptive_lambda_sigma, torso_cost=opt.torso_cost,
                    arm_cost=opt.arm_cost, hand_weight=opt.hand_weight,
                    torso_weight=opt.torso_weight)
    for name, value in positive.items():
        if not value > 0:
            ap.error(f"{name} 必须大于 0")
    if min(opt.torso_leash, opt.adaptive_lambda, w.posture_k, w.limit_w) < 0:
        ap.error("torso_leash / adaptive_lambda / posture_k / limit_w 不能为负")
    if not 0.0 <= opt.torso_return_gain < 1.0:
        ap.error("torso_return_gain 必须在 [0, 1) 内")
    return a, opt


def main(argv=None):
    global g_debug_input
    a, opt = parse_args(sys.argv[1:] if argv is None else argv)
    g_debug_input = a.debug_input
    if a.print_params:
        print(json.dumps(to_dict(opt), indent=2, ensure_ascii=False))
        return 0
    robot = robots.get_robot(a.robot)
    model, data, substeps = load_sim(robot, opt.control_dt)
    set_initial_posture(robot, model, data)
    ctl = InteractiveController(model, data, substeps, opt, robot, use_base=a.base)
    ctl.initialize()

    trace_dir = a.trace_dir
    if a.data_record or trace_dir:
        if not trace_dir:
            trace_dir = os.path.join(HERE, "hqp_sim_trace", time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(trace_dir, exist_ok=True)
        ctl.enable_trace(os.path.join(trace_dir, "trace.csv"))
        print(f"[trace] recording to {trace_dir}/trace.csv\n"
              f"[trace] live view:  python3 tools/plot_trace.py {trace_dir}/trace.csv --live")

    def on_signal(*_):
        global g_stop
        g_stop = True
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    ret = run_demo(ctl, opt.place_dx) if a.demo else Viewer(ctl).run()
    if trace_dir:
        ctl.close_trace()
        print(f"[trace] data saved in {trace_dir}/trace.csv; plot it with:\n"
              f"  python3 tools/plot_trace.py {trace_dir}/trace.csv            # save PNGs\n"
              f"  python3 tools/plot_trace.py {trace_dir}/trace.csv --show     # interactive windows")
    return ret


if __name__ == "__main__":
    sys.exit(main())
