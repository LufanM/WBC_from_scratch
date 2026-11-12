"""
tasks.py —— 常用任务、约束的构造函数，以及姿态误差等工具函数。

任务（QP_WBC.md §4）：
    pose_task            末端 6 维位姿
    position_task        末端 3 维位置
    joint_posture_task   把关节拉向参考姿态
    joint_velocity_task  让 v 接近某个参考速度
    scalar_task          标量指标（如臂角）
约束（QP_WBC.md §5）：
    joint_velocity_bounds   关节位置 + 速度限位 → v 的上下界
    linearized_constraint   非线性约束 c(q) 线性化后的一行
工具：
    quat_error、rpy_to_quat、integrate

四元数统一使用 MuJoCo 的 [w, x, y, z] 顺序。
"""
from __future__ import annotations

import numpy as np

from .problem import LinearConstraint, Task


# =============================================================================
# 工具函数
# =============================================================================
def quat_error(q_cur, q_des):
    """姿态误差：从 q_cur 转到 q_des 的旋转向量（世界坐标系，最短路径）。

    即 log(R_des R_cur^T)，与世界坐标系几何雅可比的角速度行对应。
    """
    w1, x1, y1, z1 = np.asarray(q_des, float)
    w0, x0, y0, z0 = np.asarray(q_cur, float)
    # 误差四元数 q_err = q_des * conj(q_cur)
    q_err = np.array([
        w1 * w0 + x1 * x0 + y1 * y0 + z1 * z0,
        -w1 * x0 + x1 * w0 - y1 * z0 + z1 * y0,
        -w1 * y0 + x1 * z0 + y1 * w0 - z1 * x0,
        -w1 * z0 - x1 * y0 + y1 * x0 + z1 * w0,
    ])
    if q_err[0] < 0.0:          # 取最短路径
        q_err = -q_err
    sin_half_norm = np.linalg.norm(q_err[1:])
    if sin_half_norm < 1e-12:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(sin_half_norm, q_err[0])
    return (angle / sin_half_norm) * q_err[1:]


def rpy_to_quat(rpy):
    """roll-pitch-yaw（绕固定轴 X、Y、Z，弧度）→ 四元数 [w, x, y, z]。"""
    r, p, y = (float(v) for v in rpy)
    cr, sr = np.cos(r / 2), np.sin(r / 2)
    cp, sp = np.cos(p / 2), np.sin(p / 2)
    cy, sy = np.cos(y / 2), np.sin(y / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def integrate(q, v, dt, q_min=None, q_max=None):
    """q + v * dt；给出 q_min、q_max 时裁剪到位置限位内。"""
    q_new = np.asarray(q, float) + np.asarray(v, float) * dt
    if q_min is not None and q_max is not None:
        q_new = np.clip(q_new, q_min, q_max)
    return q_new


# =============================================================================
# 任务
# =============================================================================
def pose_task(name, J, pos, quat, target_pos, target_quat,
              kp_pos=12.0, kp_rot=12.0, weight=1.0, w_pos=1.0, w_rot=0.4,
              xdot_ff=None, priority=0):
    """末端 6 维位姿任务：xdot_des = xdot_ff + Kp * [位置误差; 姿态误差]。

    J          6 x n 世界坐标系雅可比（前 3 行线速度，后 3 行角速度）
    pos, quat  当前位姿；target_pos, target_quat 为目标位姿
    w_pos, w_rot  位置 / 姿态的方向权重
    xdot_ff    目标的速度（前馈）；目标在动时一定要给，否则误差滞后 v/Kp
    """
    e_pos = np.asarray(target_pos, float) - np.asarray(pos, float)
    e_rot = quat_error(quat, target_quat)
    xdot = np.concatenate([kp_pos * e_pos, kp_rot * e_rot])
    if xdot_ff is not None:
        xdot = xdot + np.asarray(xdot_ff, float)
    w_diag = np.array([w_pos] * 3 + [w_rot] * 3, float)
    return Task(name, J, xdot, weight, w_diag, priority)


def position_task(name, Jp, pos, target_pos, kp=12.0, weight=1.0,
                  xdot_ff=None, priority=0):
    """末端 3 维位置任务（Jp 为 3 x n 位置雅可比）。"""
    xdot = kp * (np.asarray(target_pos, float) - np.asarray(pos, float))
    if xdot_ff is not None:
        xdot = xdot + np.asarray(xdot_ff, float)
    return Task(name, Jp, xdot, weight, None, priority)


def joint_posture_task(q, q_ref, kp=1.0, weight=1e-3, mask=None,
                       name="posture", priority=0):
    """关节姿态任务：把 mask 选中的关节拉向 q_ref（mask 为空表示全部）。

    权重要远小于主任务，只在冗余方向上起作用，防止构型漂移（QP_WBC.md §4.4）。
    """
    q = np.asarray(q, float)
    n = q.size
    idx = np.arange(n) if mask is None else np.flatnonzero(mask)
    S = np.eye(n)[idx]                       # 选择矩阵：每行选中一个关节
    xdot = kp * (np.asarray(q_ref, float)[idx] - q[idx])
    return Task(name, S, xdot, weight, None, priority)


def joint_velocity_task(v_ref, w_diag, name="velocity_ref", priority=0):
    """1/2 || v - v_ref ||^2_diag(w_diag)。

    J = I，它在 H 中贡献 diag(w)，在 g 中贡献 -diag(w) v_ref。
    （与 WBCProblem 的姿态项同形；需要多个这样的项或单独报告残差时用它。）
    """
    v_ref = np.asarray(v_ref, float)
    return Task(name, np.eye(v_ref.size), v_ref, 1.0, w_diag, priority)


def scalar_task(name, Jc, rate_cmd, weight=1.0, priority=0):
    """标量指标任务 Jc v = rate_cmd（如臂角：Jc 为 1 x n，rate_cmd 为期望 ψ̇）。"""
    return Task(name, np.reshape(Jc, (1, -1)), [rate_cmd], weight, None,
                priority)


# =============================================================================
# 约束
# =============================================================================
def joint_velocity_bounds(q, q_min, q_max, qd_max, dt, beta=1.0, frozen=None):
    """关节位置限位 + 速度限位 → v 的上下界（QP_WBC.md §5.2）。

        lb = clamp(beta * (q_min - q) / dt, -qd_max, +qd_max)
        ub = clamp(beta * (q_max - q) / dt, -qd_max, +qd_max)

    beta    限位缓冲系数 (0, 1]：越小越早减速
    frozen  布尔 mask，被选中的自由度上下界都设为 0
    关节已经超出限位时，会以最大速度退回，不会出现 lb > ub。
    """
    q = np.asarray(q, float)
    qd_max = np.broadcast_to(np.asarray(qd_max, float), q.shape)
    lb = np.clip(beta * (np.asarray(q_min, float) - q) / dt, -qd_max, qd_max)
    ub = np.clip(beta * (np.asarray(q_max, float) - q) / dt, -qd_max, qd_max)
    if frozen is not None:
        frozen = np.asarray(frozen, bool)
        lb[frozen] = 0.0
        ub[frozen] = 0.0
    return lb, ub


def linearized_constraint(name, Jc, c, c_min, c_max, dt, beta=1.0,
                          soft=True, w_lin=1e3, w_quad=1e1, priority=0):
    """把 c_min <= c(q) <= c_max 线性化成一行（或多行）线性约束（QP_WBC.md §5.3）：

        beta * (c_min - c) / dt  <=  Jc v  <=  beta * (c_max - c) / dt

    Jc     约束量对关节的雅可比（k x n）
    c      当前约束量（标量或 k 维）
    单边约束把另一侧设为 +/-inf。默认是软约束。
    priority  软约束在 HQP 中所在的层（见 LinearConstraint）
    """
    Jc = np.atleast_2d(np.asarray(Jc, float))
    c = np.atleast_1d(np.asarray(c, float))
    lb = beta * (np.asarray(c_min, float) - c) / dt
    ub = beta * (np.asarray(c_max, float) - c) / dt
    return LinearConstraint(name, Jc, lb, ub, soft, w_lin, w_quad, priority)
