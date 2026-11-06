"""
wbc_function —— 全身控制（WBC）求解器包。

文件组织：

    problem.py            问题描述：Task、LinearConstraint、WBCProblem、WBCResult
    tasks.py              任务 / 约束的构造函数，姿态误差等工具
    kinematics.py         MujocoKinematics：从 MuJoCo 取 q、限位、雅可比
    wls_solver.py         WLSSolver：加权最小二乘（闭式解，无不等式约束）
    wqp_solver.py         WQPSolver：加权 QP（qpOASES，支持约束）
    hqp_solver.py         HQPSolver：分层 QP（按 Task.priority 逐层求解，每层 qpOASES）
    config.py             WBCConfig、SolverConfig：所有参数和默认值
    wbc_controller.py     WholeBodyController：末端到达全身控制器（应用层，
                          组合上面的组件；可选 WLS / WQP / HQP）
    qpoases_interface.py  qpOASES 的 Python 封装（编译、加载桥接库）
    qpoases_bridge.cpp    qpOASES 的 C 接口（首次使用时自动编译到 _build/）

三个求解器共用同一个 WBCProblem，换求解器只需换一行。典型用法：

    import wbc_function as wf

    kin = wf.MujocoKinematics(model, data, joint_names)
    solver = wf.WQPSolver()                     # 或 wf.WLSSolver() / wf.HQPSolver()

    prob = wf.WBCProblem(kin.n, lambda_reg=1e-4, wp_diag=wp)
    prob.add_task(wf.site_pose_task(kin, "right_ee", target_pos, target_quat))
    prob.set_bounds(*wf.joint_velocity_bounds(q, kin.q_min, kin.q_max,
                                              qd_max, dt))
    result = solver.solve(prob)
    q = wf.integrate(q, result.v, dt)

原理与接口对应关系见 QP_WBC.md §11。
"""
from .config import SolverConfig, WBCConfig
from .kinematics import MujocoKinematics, site_pose_task
from .problem import LinearConstraint, Task, WBCProblem, WBCResult
from .qpoases_interface import QP_INF, QpOasesSQP
from .tasks import (integrate, joint_posture_task, joint_velocity_bounds,
                    joint_velocity_task, linearized_constraint, pose_task,
                    position_task, quat_error, rpy_to_quat, scalar_task)
from .wls_solver import WLSSolver
from .wqp_solver import WQPSolver
from .hqp_solver import HQPSolver
from .wbc_controller import TaskOutput, WholeBodyController


def make_solver(kind="wqp", **kwargs):
    """按名字创建求解器："wls" / "wqp" / "hqp"。"""
    kind = kind.lower()
    if kind == "wls":
        return WLSSolver(**kwargs)
    if kind == "wqp":
        return WQPSolver(**kwargs)
    if kind == "hqp":
        return HQPSolver(**kwargs)
    raise ValueError(f"未知的求解器类型 {kind!r}，可选 'wls' / 'wqp' / 'hqp'")


__all__ = [
    # 问题描述
    "Task", "LinearConstraint", "WBCProblem", "WBCResult",
    # 任务 / 约束构造
    "pose_task", "position_task", "joint_posture_task", "joint_velocity_task",
    "scalar_task", "joint_velocity_bounds", "linearized_constraint",
    # 工具
    "quat_error", "rpy_to_quat", "integrate",
    # MuJoCo
    "MujocoKinematics", "site_pose_task",
    # 求解器
    "WLSSolver", "WQPSolver", "HQPSolver", "make_solver", "QpOasesSQP", "QP_INF",
    # 应用层
    "WholeBodyController", "TaskOutput", "WBCConfig", "SolverConfig",
]
