"""
wbc_controller.py —— 全身控制器（应用层）。

在 wbc_function 的基础组件之上，组装出可以同时跟踪末端、躯干、底盘等的
全身控制器，规划底盘（x, y, yaw）、躯干（4 自由度）和手臂（每侧 7 自由度）。

任务是常驻对象：构造时（或之后）注册，运行时随时修改目标、前馈速度、开关、
权重；每个控制周期调用一次 step() 求解。控制器只如实求解当前给定的目标，
不做任何平滑——目标轨迹由上层规划保证连续。

    wbc = WholeBodyController(model, data, joints, ["right_ee"], solver="wqp",
                              frames=R1_FRAMES)
    hand, torso = wbc.task("right_ee"), wbc.task("torso")

    hand.set_target(p, q, vel=v_ff)      # 目标 + 前馈速度（可选）
    hand.enabled = True
    torso.enabled = False                # 切换任务：改开关即可
    wbc.set_frozen(TORSO_JOINTS)         # 冻结一组关节（[] 表示全部解冻）

    qd, err = wbc.step()                 # err["right_ee"] = (位置误差, 姿态误差)

    # 快捷写法：设置目标并只启用这些坐标系任务，然后求解一步
    qd, err = wbc.compute({"right_ee": (p, q)})

三种求解方式（构造时用 solver 参数选择）：

    "wls"  加权最小二乘，与项目最初的实现完全一致：求解后逐自由度裁剪速度，
           底盘平移速度整体按比例缩放，积分时把关节角裁剪到限位内。
    "wqp"  加权 QP：目标函数相同，但关节位置、速度限位作为硬约束参与求解，
           速度饱和时仍保持运动方向（QP_WBC.md §7）。
    "hqp"  分层 QP：约束同 "wqp"，任务按 priority 分层（数字越小越优先），
           高层做到最好后锁定，低层只在剩余自由度里优化（QP_WBC.md §8）。
           同一层内的任务仍按权重加权；"wls" / "wqp" 忽略 priority。

原理与参数说明见 QP_WBC.md §1、§11。四元数使用 MuJoCo 的 [w, x, y, z]。
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .kinematics import MujocoKinematics
from .problem import Task, WBCProblem
from .tasks import joint_velocity_bounds, quat_error
from .wls_solver import WLSSolver
from .wqp_solver import WQPSolver
from .hqp_solver import HQPSolver

BASE_JOINTS = ("base_x", "base_y", "base_yaw")

# 坐标系跟踪任务可选的分量：位置 x/y/z，以及绕世界坐标系 x/y/z 轴的转动 rx/ry/rz
# （对应几何雅可比的 6 行；rz 对底盘就是 yaw）
COMPONENTS = ("x", "y", "z", "rx", "ry", "rz")


class FrameTask:
    """坐标系跟踪任务：让某个 site / body 跟踪目标位姿（可只跟踪部分分量）。

    期望速度 = 前馈速度 + Kp * 误差，只取 components 选中的分量。
    属性可以随时修改，下一次 step() 生效：
        enabled   是否参与求解
        weight    任务权重 w_i
        priority  所在层（只对 solver="hqp" 有效，数字越小越优先）
        kp_pos / kp_rot   比例增益；None 表示使用控制器的 Kp_pos / Kp_rot
    """

    def __init__(self, name, kind, ref, components, weight, kp_pos, kp_rot,
                 priority=0):
        self.name = name
        self.kind = kind                  # "site" 或 "body"
        self.ref = ref                    # site / body 名
        self.components = tuple(components)
        self.rows = np.array([COMPONENTS.index(c) for c in self.components])
        self.weight = float(weight)
        self.priority = int(priority)
        self.kp_pos = kp_pos
        self.kp_rot = kp_rot
        self.enabled = False
        self.target_pos = None
        self.target_quat = None
        self.vel = None                   # 6 维前馈速度 [线速度; 角速度]（世界坐标系）

    def set_target(self, pos=None, quat=None, vel=None):
        """设置目标位姿和前馈速度。

        pos   目标位置 (3,)；只跟踪姿态分量时可为 None
        quat  目标四元数 (4,)；只跟踪位置分量时可为 None
        vel   前馈速度 (6,)，[vx, vy, vz, wx, wy, wz]，世界坐标系；None 表示 0。
              跟踪运动目标时应给出目标的速度，否则误差会滞后约 v/Kp。
        """
        self.target_pos = None if pos is None else np.asarray(pos, float).copy()
        self.target_quat = None if quat is None else np.asarray(quat, float).copy()
        self.vel = None if vel is None else np.asarray(vel, float).reshape(6).copy()
        return self


@dataclass
class TaskOutput:
    """自定义任务函数每个周期的返回值。

    J       m x n 雅可比，列顺序同 controller.joint_names
    xdot    m 维期望速度
    w_diag  可选，m 维方向权重；None 表示全 1
    error   可选，这个任务的误差（数值或数组，含义由你定义）；给出时会出现在
            step() / compute() 返回的 errors[任务名] 中，不给则不出现
    """
    J: object
    xdot: object
    w_diag: object = None
    error: object = None


class CustomTask:
    """自定义任务：用户提供一个函数，描述“这个任务想要什么”。

    控制器每个周期调用一次 fn(controller)（controller 即 WholeBodyController
    本身），函数用它读取当前状态，返回本周期的 TaskOutput（J、期望速度，
    以及可选的方向权重、误差）。之所以传函数而不是传数值：J 和期望速度随
    关节角每个周期都在变，必须每个周期按当前状态重新计算。

    用于坐标系任务以外的需求，例如臂角、关节空间目标等。例如让 torso_joint1
    转到 q_goal：

        i = wbc.joint_names.index("torso_joint1")

        def torso1_task(c):                         # c 就是控制器
            J = np.zeros((1, c.n)); J[0, i] = 1.0   # 只选中这一个关节
            q_now = c.get_joint_positions()[i]
            return TaskOutput(J=J, xdot=[2.0 * (q_goal - q_now)],
                              error=abs(q_goal - q_now))

        wbc.add_task("torso1", torso1_task)         # 传函数本身（不带括号）
        qd, err = wbc.step()                        # err["torso1"] 即上面的 error
    """

    def __init__(self, name, fn, weight=1.0, enabled=True, priority=0):
        self.name = name
        self.fn = fn
        self.weight = float(weight)
        self.priority = int(priority)         # 所在层（只对 solver="hqp" 有效）
        self.enabled = enabled


class WholeBodyController:
    """全身控制器。

    model, data   MuJoCo 模型与状态。控制器只读写受控关节的 qpos，不推进仿真。
    joint_names   受控关节（按控制顺序），必须是单自由度关节
    ee_sites      末端 site 名列表；每个末端自动注册为同名的 6 分量坐标系任务
    base_free     False 时底盘不参与规划（直接从受控自由度中去掉）
    base_joints   joint_names 中哪些是底盘自由度（用于底盘权重和速度上限）
    solver        "wls"（默认，与原实现一致）、"wqp" 或 "hqp"
    frames        额外注册的坐标系任务，{名字: 设置字典}，键同 add_frame_task 的参数：
                  site / body、components、weight、priority、kp_pos、kp_rot。例如：
                    frames={"torso": {"body": "torso_link4"},
                            "base":  {"body": "base_link",
                                      "components": ("x", "y", "rz")}}

    所有注册的坐标系任务初始都是关闭的（enabled=False），给目标并打开后才参与求解；
    compute(targets) 会自动完成这两步。
    """

    def __init__(self, model, data, joint_names, ee_sites=(), base_free=True,
                 base_joints=BASE_JOINTS, solver="wls", frames=None, dt=None):
        self.m = model
        self.d = data
        # 控制周期：WQP / HQP 由关节位置限位换算速度上下界时用（QP_WBC.md §5.2）。
        # None 表示用模型的仿真步长；控制周期比仿真步长长时应显式给出。
        self.dt = float(model.opt.timestep if dt is None else dt)
        self.base_free = base_free

        self.ee_sites = [ee_sites] if isinstance(ee_sites, str) else list(ee_sites)
        self.ee_site = self.ee_sites[0] if self.ee_sites else None

        # 底盘被冻结时，直接从受控自由度中去掉，任何权重都不会让它动
        base_set = set(base_joints)
        names = [n for n in joint_names if base_free or n not in base_set]
        self.kin = MujocoKinematics(model, data, names)
        self.site_id = [self.kin._site_id(s) for s in self.ee_sites]

        # 常用属性（demo 和仿真脚本会直接读取）
        self.joint_names = self.kin.joint_names
        self.n = self.kin.n
        self.qadr = self.kin.qadr
        self.dofadr = self.kin.dofadr
        self.lower = self.kin.q_min
        self.upper = self.kin.q_max
        self.base_mask = self.kin.mask(base_set)
        self.base_xy_mask = self.kin.mask(("base_x", "base_y"))

        # 关节中位作为舒适姿态；底盘没有“中位”，不参与拉回
        self.q_nominal = 0.5 * (self.lower + self.upper)
        self.q_nominal[self.base_mask] = 0.0
        self.posture_mask = ~self.base_mask

        # --- 增益与权重（符号见 QP_WBC.md §1.3.2 引用块、§1.5.1）
        # 目标函数：1/2||J v - xdot_des||^2_W + λ/2||v||^2 + 1/2||v - v_posture||^2_Wp
        self.Kp_pos = 12.0       # 末端位置比例增益（1/s）
        self.Kp_rot = 12.0       # 末端姿态比例增益（1/s）
        self.task_w = np.array([1, 1, 1, 0.4, 0.4, 0.4])  # 位置权重高于姿态
        self.lambda_reg = 1e-4   # 全局阻尼 λ
        self.posture_w = 2e-3    # W_p 的关节部分：关节动起来的代价
        self.base_reg_w = 1e-4   # W_p 的底盘部分：很小，远处目标时底盘乐于移动
        self.posture_k = 2e-3    # v_posture 中的 k：拉向中位的增益（1/s）
        self.limit_w = 0.5       # v_posture 中的 k_l：限位斥力强度（距限位 10% 行程内生效）
        self.max_joint_vel = 2.0  # rad/s
        self.max_base_vel = 0.6   # m/s，底盘平移
        self.limit_beta = 0.5    # WQP / HQP 中关节位置限位的缓冲系数（QP_WBC.md §5.2）
        # 坐标系任务的误差限幅（None = 不限）：期望速度 = 前馈 + Kp * clip(误差)。
        # 目标够不到时，误差一直很大，Kp * 误差远超关节速度上限，接近奇异的手臂会在
        # 速度上限之间来回翻转（离散控制的极限环）；限幅后期望速度有界，就不会这样。
        # 正常跟踪（误差小于限幅）不受影响。
        self.max_pos_error = None  # m
        self.max_rot_error = None  # rad

        if solver == "wls":
            self.solver = WLSSolver(saturation=None)   # 饱和处理在 _saturate 中
        elif solver == "wqp":
            self.solver = WQPSolver()
        elif solver == "hqp":
            # 每层都带正则项（阻尼 + 姿态项）：目标够不到、接近奇异时不抖（QP_WBC.md §8.3 补充）
            # （见 hqp_solver.py 开头）
            self.solver = HQPSolver(regularization="all")
        else:
            raise ValueError("solver 只能是 'wls'、'wqp' 或 'hqp'")
        self.solver_kind = solver
        self.last_result = None  # 最近一次求解的 WBCResult（残差、耗时等）

        # --- 任务（常驻对象）与冻结
        self.tasks = {}                       # 名字 -> FrameTask / CustomTask，按注册顺序
        self.constraints = {}                 # 名字 -> 约束函数（见 add_constraint）
        self.frozen = np.zeros(self.n, bool)  # 被冻结的受控自由度
        self._warned_uncontrollable = set()   # 已警告过的（任务, 分量），只警告一次
        for site in self.ee_sites:
            self.add_frame_task(site, site=site)
        for name, spec in (frames or {}).items():
            self.add_frame_task(name, **spec)

    # ------------------------------------------------------------- 任务管理
    def add_frame_task(self, name, site=None, body=None, components=COMPONENTS,
                       weight=1.0, kp_pos=None, kp_rot=None, priority=0):
        """注册一个坐标系跟踪任务，返回 FrameTask（初始关闭）。

        site / body   二选一，坐标系所在的 site 或 body 名
        components    跟踪哪些分量，默认全部 6 个（见 COMPONENTS）
        weight        任务权重 w_i
        priority      所在层（只对 solver="hqp" 有效，数字越小越优先）
        kp_pos/kp_rot 比例增益，None 表示用控制器的 Kp_pos / Kp_rot
        """
        self._check_new_name(name)
        if (site is None) == (body is None):
            raise ValueError(f"任务 {name!r} 必须且只能指定 site 或 body 之一")
        kind, ref = ("site", site) if site is not None else ("body", body)
        if kind == "site":
            self.kin._site_id(ref)            # 名字不存在时尽早报错
        else:
            self.kin._body_id(ref)
        comps = tuple(components)
        bad = [c for c in comps if c not in COMPONENTS]
        if bad or not comps:
            raise ValueError(f"任务 {name!r} 的分量 {bad or comps} 无效，可选 {COMPONENTS}")
        task = FrameTask(name, kind, ref, comps, weight, kp_pos, kp_rot, priority)
        self.tasks[name] = task
        return task

    def add_task(self, name, fn, weight=1.0, enabled=True, priority=0):
        """注册一个自定义任务（见 CustomTask），返回 CustomTask（初始开启）。

        priority  所在层（只对 solver="hqp" 有效，数字越小越优先）
        """
        self._check_new_name(name)
        task = CustomTask(name, fn, weight, enabled, priority)
        self.tasks[name] = task
        return task

    def add_constraint(self, name, fn):
        """注册一个常驻约束：每个周期调用 fn(controller)，返回本周期的
        LinearConstraint（或 None 表示本周期不加）。WLS 会忽略不等式约束。

        例如躯干位置的安全盒（线性化，QP_WBC.md §5.3）：

            def torso_box(c):
                p, _ = c.get_frame_pose("torso")
                J = c.frame_jacobian("torso")[[0, 2]]          # x、z 两行
                return wf.linearized_constraint("torso_box", J, p[[0, 2]],
                                                [x_min, z_min], [x_max, z_max],
                                                c.dt, beta=0.5, soft=False)
            wbc.add_constraint("torso_box", torso_box)
        """
        if name in self.constraints:
            raise ValueError(f"约束名 {name!r} 已被占用")
        self.constraints[name] = fn

    def remove_constraint(self, name):
        del self.constraints[name]

    def task(self, name):
        """按名字取已注册的任务对象。"""
        if name not in self.tasks:
            raise KeyError(f"未注册的任务 {name!r}，已注册：{list(self.tasks)}")
        return self.tasks[name]

    def remove_task(self, name):
        self.task(name)
        del self.tasks[name]

    def _check_new_name(self, name):
        if name in self.tasks:
            raise ValueError(f"任务名 {name!r} 已被占用")

    def set_frozen(self, joint_names):
        """冻结一组受控关节（速度为 0），替换之前的冻结设置；传 [] 表示全部解冻。

        冻结不改变 QP 的维度，可以在运行中随时切换（WLS / WQP / HQP 都支持）。
        """
        names = list(joint_names)
        unknown = [n for n in names if n not in self.joint_names]
        if unknown:
            raise KeyError(f"不是受控关节：{unknown}")
        self.frozen = self.kin.mask(names)

    @property
    def frozen_joints(self):
        return [n for n, f in zip(self.joint_names, self.frozen) if f]

    # ------------------------------------------------------------- 状态读写
    def get_joint_positions(self):
        return self.kin.q()

    def set_joint_positions(self, q):
        """写入受控关节位置并刷新运动学。"""
        self.kin.set_q(q)

    def get_base_pose(self):
        """底盘 (x, y, yaw)，从底盘关节读取（模型中没有的关节记为 0）。"""
        import mujoco
        pose = []
        for name in BASE_JOINTS:
            jid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, name)
            pose.append(float(self.d.qpos[self.m.jnt_qposadr[jid]])
                        if jid >= 0 else 0.0)
        return np.array(pose)

    def _site(self, which):
        """which 可以是 ee_sites 中的下标，也可以直接是 site 名。"""
        return which if isinstance(which, str) else self.site_id[which]

    def get_ee_pose(self, which=0):
        """末端在世界坐标系中的 (位置, 四元数)。"""
        return self.kin.site_pose(self._site(which))

    def get_frame_pose(self, name):
        """坐标系任务所在 site / body 在世界坐标系中的 (位置, 四元数)。"""
        f = self._frame_task(name)
        if f.kind == "site":
            return self.kin.site_pose(f.ref)
        return self.kin.body_pose(f.ref)

    def frame_jacobian(self, name):
        """坐标系任务所在 site / body 的 6 x n 世界坐标系雅可比。"""
        f = self._frame_task(name)
        if f.kind == "site":
            return self.kin.site_jacobian(f.ref)
        return self.kin.body_jacobian(f.ref)

    def _frame_task(self, name):
        task = self.task(name)
        if not isinstance(task, FrameTask):
            raise TypeError(f"任务 {name!r} 不是坐标系任务")
        return task

    # ------------------------------------------------------------- 雅可比
    def site_jacobian(self, which=0):
        """6 x n 世界坐标系雅可比（前 3 行线速度，后 3 行角速度）。"""
        return self.kin.site_jacobian(self._site(which))

    def body_jacobian(self, body_name):
        """3 x n body 原点的位置雅可比（臂角 demo 用来取肘部雅可比）。"""
        return self.kin.body_jacobian(body_name)[:3]

    def check_jacobian(self, eps=1e-6):
        """用有限差分校验末端雅可比，返回最大误差（期望 1e-6 量级）。

        逐个扰动受控关节，用正运动学测出末端位姿变化，与雅可比对应列比较。
        列的选取（dofadr）和坐标系约定是最容易出隐蔽错误的地方，
        后面所有结果都建立在它之上。会恢复原状态。
        """
        q0 = self.get_joint_positions()
        max_err = 0.0
        for k in range(len(self.site_id)):
            self.set_joint_positions(q0)
            J = self.site_jacobian(k)
            pos0, quat0 = self.get_ee_pose(k)
            for c in range(self.n):
                q = q0.copy()
                q[c] += eps
                self.set_joint_positions(q)
                pos1, quat1 = self.get_ee_pose(k)
                measured = np.concatenate([(pos1 - pos0) / eps,
                                           quat_error(quat0, quat1) / eps])
                max_err = max(max_err, float(np.max(np.abs(measured - J[:, c]))))
        self.set_joint_positions(q0)
        return max_err

    # ------------------------------------------------------------- 姿态项
    def _posture_velocity(self, q):
        """姿态速度 v_posture（QP_WBC.md §1.3.2 引用块）：

            v_posture,i = k (q_mid,i - q_i)
                        + k_l [max(0, 1 - (q_i - q_min,i)/m_i) - max(0, 1 - (q_max,i - q_i)/m_i)]
            m_i = 0.1 (q_max,i - q_min,i)

        k = posture_k，k_l = limit_w。底盘的 k 取 0（它没有偏好的世界位置），
        但仍受限位斥力。
        """
        v = self.posture_k * (self.q_nominal - q) * self.posture_mask
        margin = 0.1 * (self.upper - self.lower) + 1e-6
        near_lower = q - self.lower
        near_upper = self.upper - q
        v += self.limit_w * np.where(near_lower < margin,
                                     (margin - near_lower) / margin, 0.0)
        v -= self.limit_w * np.where(near_upper < margin,
                                     (margin - near_upper) / margin, 0.0)
        return v

    # ------------------------------------------------------------- 主循环一步
    def step(self):
        """用当前已开启的任务和冻结设置求解一步。返回 (q_dot[n], errors)。

        q_dot 的顺序与 joint_names 相同；errors 包含本周期开启的任务：
          坐标系任务  {任务名: (位置误差, 姿态误差)}，只统计该任务跟踪的分量
          自定义任务  {任务名: 函数返回的 TaskOutput.error}（没给 error 则不出现）
        各任务的残差 ||J v - xdot|| 见 self.last_result.task_residuals。
        """
        problem, errors = self.build_problem()
        self.last_result = self.solver.solve(problem)
        qd = self.last_result.v
        if self.solver_kind == "wls":
            qd = self._saturate(qd)
        return qd, errors

    def compute(self, targets):
        """快捷写法：设置目标、只开启这些坐标系任务，然后求解一步。返回同 step()。

        targets   {坐标系任务名: (位置, 四元数)} 或 {名字: (位置, 四元数, 前馈速度)}。
                  本次没给的坐标系任务会被关闭；自定义任务的开关不受影响。

            qd, err = wbc.compute({"right_ee": (p_hand, q_hand),
                                   "torso":    (p_torso, q_torso)})
        """
        self._apply_targets(targets)
        return self.step()

    def _apply_targets(self, targets):
        """设置这些坐标系任务的目标并开启，其余坐标系任务关闭。"""
        if not isinstance(targets, dict):
            raise TypeError("targets 应为 {坐标系任务名: (位置, 四元数)} 字典，"
                            "例如 {\"right_ee\": (p, q)}")
        if not targets:
            raise ValueError("至少要给一个目标")
        for name in targets:
            self._frame_task(name)                # 名字检查
        for name, task in self.tasks.items():
            if isinstance(task, FrameTask):
                task.enabled = name in targets
                if task.enabled:
                    task.set_target(*targets[name])

    def build_problem(self, targets=None):
        """用当前已开启的任务和冻结设置构造本周期的 WBCProblem（不求解）。
        返回 (problem, errors)。可在此基础上再加任务或约束，然后交给 self.solver。

        给出 targets 时，先按 compute 的规则设置目标、开关坐标系任务。
        """
        if targets is not None:
            self._apply_targets(targets)
        q = self.get_joint_positions()
        # 姿态项 1/2||v - v_posture||^2_Wp 直接交给 WBCProblem（QP_WBC.md §1.3.2 引用块）
        problem = WBCProblem(self.n, lambda_reg=self.lambda_reg,
                             wp_diag=self._posture_weights(),
                             v_posture=self._posture_velocity(q))
        active = [t for t in self.tasks.values() if t.enabled]
        if not active:
            raise ValueError("没有开启的任务")

        errors = {}
        for task in active:
            if isinstance(task, FrameTask):
                # 坐标系任务：J 和期望速度由控制器按目标位姿、前馈速度算出
                errors[task.name] = self._add_frame_task_to(problem, task)
            else:
                # 自定义任务：J 和期望速度由用户函数算出。
                # task.fn 是 add_task 时传入的函数，这里每个周期调用一次，
                # 把控制器自己（self）传给它，函数借此读取当前状态（关节角、
                # 雅可比、位姿等），返回本周期的 TaskOutput。
                out = task.fn(self)
                if not isinstance(out, TaskOutput):
                    raise TypeError(f"自定义任务 {task.name!r} 的函数应返回 "
                                    f"TaskOutput(J, xdot, w_diag=None, error=None)，"
                                    f"实际返回了 {type(out).__name__}")
                # 包装成 Task 加入本周期的 QP，之后与坐标系任务一样参与求解，
                # 残差按任务名记录在 last_result.task_residuals 中
                problem.add_task(Task(task.name, out.J, out.xdot, task.weight, out.w_diag,
                                      task.priority))
                # 误差只有用户知道含义，函数给出了才报告
                if out.error is not None:
                    errors[task.name] = out.error

        for name, fn in self.constraints.items():
            c = fn(self)
            if c is not None:
                problem.add_constraint(c)

        if self.solver_kind in ("wqp", "hqp"):
            problem.set_bounds(*self._velocity_bounds(q))
        problem.freeze(self.frozen)               # WLS 也据此把冻结自由度固定为 0
        return problem, errors

    def _add_frame_task_to(self, problem, task):
        """把一个坐标系任务加入 problem，返回 (位置误差, 姿态误差)。"""
        rows = task.rows
        needs_pos = bool(np.any(rows < 3))
        needs_rot = bool(np.any(rows >= 3))
        if (needs_pos and task.target_pos is None) or (needs_rot and task.target_quat is None):
            raise ValueError(f"任务 {task.name!r} 已开启，但没有给出"
                             f"{'位置' if needs_pos and task.target_pos is None else '姿态'}目标")
        pos, quat = self.get_frame_pose(task.name)
        e_pos = (task.target_pos - pos) if needs_pos else np.zeros(3)
        e_rot = quat_error(quat, task.target_quat) if needs_rot else np.zeros(3)
        e = np.concatenate([e_pos, e_rot])        # 报告用真实误差，限幅只作用于期望速度
        e_pos = self._clip_norm(e_pos, self.max_pos_error)
        e_rot = self._clip_norm(e_rot, self.max_rot_error)
        kp_pos = self.Kp_pos if task.kp_pos is None else task.kp_pos
        kp_rot = self.Kp_rot if task.kp_rot is None else task.kp_rot
        xdot = np.concatenate([kp_pos * e_pos, kp_rot * e_rot])
        if task.vel is not None:
            xdot = xdot + task.vel                # 前馈
        J = self.frame_jacobian(task.name)[rows]
        self._check_controllable(task.name, rows, J)
        problem.add_task(Task(task.name, J, xdot[rows], task.weight, self.task_w[rows],
                              task.priority))
        return (float(np.linalg.norm(e[rows[rows < 3]])),
                float(np.linalg.norm(e[rows[rows >= 3]])))

    @staticmethod
    def _clip_norm(e, limit):
        if limit is None:
            return e
        norm = float(np.linalg.norm(e))
        return e if norm <= limit else e * (limit / norm)

    def _check_controllable(self, name, rows, J):
        """检查跟踪的分量是否至少有一个受控自由度能影响它（雅可比该行不全为 0）。

        全部分量都控制不了（如 base_free=False 时跟踪底盘）→ 报错：任务完全不起作用；
        部分分量控制不了（如底盘不参与时躯干的 y 平移）→ 警告一次，其余分量照常跟踪。
        """
        # 冻结的自由度不算“能影响它”
        dead = np.max(np.abs(J[:, ~self.frozen]), axis=1, initial=0.0) < 1e-12
        if not np.any(dead):
            return
        comps = [COMPONENTS[r] for r in rows[dead]]
        if np.all(dead):
            raise ValueError(
                f"坐标系 {name!r} 跟踪的分量 {comps} 都不受任何受控自由度影响，"
                f"这个任务不会起作用（例如底盘不参与或被冻结时跟踪底盘）")
        key = (name, tuple(comps))
        if key not in self._warned_uncontrollable:
            self._warned_uncontrollable.add(key)
            warnings.warn(
                f"坐标系 {name!r} 的分量 {comps} 不受任何受控自由度影响，这些分量的"
                f"目标不会被跟踪；如不需要，可在注册时的 components 中去掉它们",
                stacklevel=3)

    def integrate(self, qd, dt):
        """运动学积分一步：q + qd*dt，裁剪到关节限位内。返回新的关节位置。"""
        q_new = np.clip(self.get_joint_positions() + qd * dt,
                        self.lower, self.upper)
        self.set_joint_positions(q_new)
        return q_new

    # ------------------------------------------------------------- 内部
    def _posture_weights(self):
        """W_p 的对角：底盘 base_reg_w，关节 posture_w。"""
        return np.where(self.base_mask, self.base_reg_w, self.posture_w)

    def _saturate(self, qd):
        """WLS 的速度饱和：逐自由度裁剪；底盘平移作为二维向量整体缩放（保持方向）。"""
        qd = np.clip(qd, -self.max_joint_vel, self.max_joint_vel)
        if self.base_free:
            speed = np.linalg.norm(qd[self.base_xy_mask])
            if speed > self.max_base_vel:
                qd[self.base_xy_mask] *= self.max_base_vel / speed
        return qd

    def _velocity_bounds(self, q):
        """WQP / HQP 的速度上下界：关节位置限位 + 速度限位。

        底盘平移原本是圆形速度限制 |(vx, vy)| <= max_base_vel，这里用内接正方形
        |vx|, |vy| <= max_base_vel / sqrt(2) 近似（保证不超过原限制）。
        """
        qd_max = np.full(self.n, self.max_joint_vel)
        qd_max[self.base_xy_mask] = self.max_base_vel / np.sqrt(2.0)
        return joint_velocity_bounds(q, self.lower, self.upper, qd_max, self.dt,
                                     beta=self.limit_beta)
