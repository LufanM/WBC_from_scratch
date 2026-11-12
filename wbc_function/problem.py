"""
problem.py —— WBC 问题描述：任务、约束、问题、结果。

WLS、WQP、HQP 三个求解器共用这里的数据结构（原理见 QP_WBC.md §4~§8）：

    目标函数  sum_i  weight_i/2 * ||J_i v - xdot_i||^2_{W_i}
              + lambda/2 * ||v||^2  +  1/2 * ||v - v_posture||^2_{W_p}
    约束      lb <= v <= ub                （Box：关节限位、冻结）
              lbA <= A v <= ubA            （线性约束，可设为软约束）

决策变量 v = q_dot（受控自由度的速度），所有边界都是速度量纲。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# =============================================================================
# 任务
# =============================================================================
@dataclass
class Task:
    """一个最小二乘任务：weight/2 * || J v - xdot_des ||^2_diag(w_diag)。

    name      任务名（结果中按名字报告残差）
    J         m x n 雅可比
    xdot_des  m 维期望任务速度
    weight    任务权重（标量）
    w_diag    m 维方向权重，默认全 1（如位置 1、姿态 0.4）
    priority  所在层（HQP 使用，数字越小优先级越高）；WLS / WQP 忽略
    """
    name: str
    J: np.ndarray
    xdot_des: np.ndarray
    weight: float = 1.0
    w_diag: np.ndarray | None = None
    priority: int = 0

    def __post_init__(self):
        self.J = np.atleast_2d(np.asarray(self.J, float))
        self.xdot_des = np.atleast_1d(np.asarray(self.xdot_des, float))
        m = self.J.shape[0]
        if self.xdot_des.shape != (m,):
            raise ValueError(f"任务 {self.name!r}：xdot_des 形状为 "
                             f"{self.xdot_des.shape}，应为 ({m},)")
        if self.w_diag is None:
            self.w_diag = np.ones(m)
        else:
            self.w_diag = np.asarray(self.w_diag, float).reshape(m)

    def weights(self):
        """每一行的总权重 weight * w_diag。"""
        return self.weight * self.w_diag

    def residual(self, v):
        """任务残差 J v - xdot_des。"""
        return self.J @ v - self.xdot_des


# =============================================================================
# 线性约束
# =============================================================================
@dataclass
class LinearConstraint:
    """线性约束 lb <= A v <= ub（速度量纲）。

    soft=False：硬约束，必须满足。
    soft=True ：软约束，引入松弛变量 s >= 0：
                    lb - s <= A v <= ub + s
                代价 w_lin * sum(s) + w_quad/2 * ||s||^2。
                线性项保证“能满足时 s 严格为 0”（QP_WBC.md §6）。
    单边约束把另一侧设为 +/-inf 即可。
    priority    只对 HQP 的软约束有意义：它在第 priority 层被最小化违反量，
                之后各层带着求得的 s* 当作硬约束（QP_WBC.md §8.4）。
                硬约束不分层，每一层都满足。
    """
    name: str
    A: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    soft: bool = False
    w_lin: float = 1e3
    w_quad: float = 1e1
    priority: int = 0

    def __post_init__(self):
        self.A = np.atleast_2d(np.asarray(self.A, float))
        rows = self.A.shape[0]
        self.lb = np.broadcast_to(np.asarray(self.lb, float), (rows,)).copy()
        self.ub = np.broadcast_to(np.asarray(self.ub, float), (rows,)).copy()
        if np.any(self.lb > self.ub):
            raise ValueError(f"约束 {self.name!r}：存在 lb > ub")

    @property
    def rows(self):
        return self.A.shape[0]

    def relaxed(self, s):
        """违反量固定为 s 后的硬约束 lb - s <= A v <= ub + s（HQP 中下层使用，QP_WBC.md §8.4）。"""
        return LinearConstraint(self.name, self.A, self.lb - s, self.ub + s)


# =============================================================================
# 问题
# =============================================================================
class WBCProblem:
    """一个控制周期的完整问题：任务 + 正则化 + 约束。

    n           受控自由度个数
    lambda_reg  全局阻尼 λ（保证 H 正定、抑制奇异附近的跳变）
    wp_diag     姿态项权重 W_p 的对角：每个自由度动起来的代价（越大越“懒”），默认全 0
    v_posture   姿态速度：冗余方向上希望关节动的速度（如拉向中位、推离限位），默认全 0

    姿态项 1/2 ||v - v_posture||^2_{W_p} 见 QP_WBC.md §1.3.2 引用块。
    v_posture = 0 时它退化为正文的分组正则化 1/2 v^T R v（R = W_p）。
    """

    def __init__(self, n, lambda_reg=1e-4, wp_diag=None, v_posture=None):
        self.n = int(n)
        self.lambda_reg = float(lambda_reg)
        self.wp_diag = (np.zeros(self.n) if wp_diag is None
                        else np.asarray(wp_diag, float).reshape(self.n))
        self.v_posture = (np.zeros(self.n) if v_posture is None
                          else np.asarray(v_posture, float).reshape(self.n))
        self.tasks: list[Task] = []
        self.constraints: list[LinearConstraint] = []
        self.lb = np.full(self.n, -np.inf)
        self.ub = np.full(self.n, np.inf)

    # --- 添加任务、约束 --------------------------------------------------------
    def add_task(self, task: Task):
        if task.J.shape[1] != self.n:
            raise ValueError(f"任务 {task.name!r} 有 {task.J.shape[1]} 列，"
                             f"问题的自由度 n = {self.n}")
        self.tasks.append(task)
        return task

    def add_constraint(self, constraint: LinearConstraint):
        if constraint.A.shape[1] != self.n:
            raise ValueError(f"约束 {constraint.name!r} 有 "
                             f"{constraint.A.shape[1]} 列，问题的自由度 n = {self.n}")
        self.constraints.append(constraint)
        return constraint

    def set_bounds(self, lb, ub):
        """设置 v 的上下界（通常来自 joint_velocity_bounds）。"""
        lb = np.broadcast_to(np.asarray(lb, float), (self.n,)).copy()
        ub = np.broadcast_to(np.asarray(ub, float), (self.n,)).copy()
        if np.any(lb > ub):
            bad = np.flatnonzero(lb > ub)
            raise ValueError(f"上下界矛盾（lb > ub）的自由度：{bad.tolist()}")
        self.lb, self.ub = lb, ub

    def freeze(self, mask):
        """冻结 mask 选中的自由度（lb = ub = 0）。"""
        mask = np.asarray(mask, bool)
        self.lb[mask] = 0.0
        self.ub[mask] = 0.0

    # --- 目标函数 --------------------------------------------------------------
    def regularization(self, posture=True):
        """正则化部分 λ/2||v||^2 + 1/2||v - v_posture||^2_{W_p} 的 (H, g)。

        posture=False 时取 v_posture = 0，只剩权重正则 λ/2||v||^2 + 1/2 v^T W_p v。
        建议低层加 v_posture、高层不加：v_posture 会把 v 往回中 / 斥力方向拉，放在高层
        会让高优先级任务产生残差；只用权重正则则只起阻尼作用。HQP 按此处理
        （hqp_solver.py 开头）；WLS / WQP 只有一层，通过 objective() 使用完整形式。
        """
        H = self.lambda_reg * np.eye(self.n) + np.diag(self.wp_diag)
        g = -self.wp_diag * self.v_posture if posture else np.zeros(self.n)
        return H, g

    def task_objective(self, tasks=None):
        """任务部分 sum_i w_i/2 ||J_i v - xdot_i||^2_{W_i} 的 (H, g)。

        tasks  只累加这些任务（HQP 按层取）；None 表示全部任务。
        """
        H = np.zeros((self.n, self.n))
        g = np.zeros(self.n)
        for task in self.tasks if tasks is None else tasks:
            JtW = task.J.T * task.weights()      # 等价于 J^T @ diag(weights)
            H += JtW @ task.J
            g -= JtW @ task.xdot_des
        return H, g

    def objective(self):
        """返回目标函数 1/2 v^T H v + g^T v 中的 (H, g)。

            H = sum_i J_i^T W_i J_i + λI + W_p             （即 §1.4 的 A）
            g = -sum_i J_i^T W_i xdot_i - W_p v_posture    （即 §1.4 的 -b）
        """
        H_reg, g_reg = self.regularization()
        H_task, g_task = self.task_objective()
        return H_reg + H_task, g_reg + g_task


# =============================================================================
# 结果
# =============================================================================
@dataclass
class WBCResult:
    """求解结果，以及每个周期值得记录的诊断量（QP_WBC.md §11.4）。

    v                   最优速度；失败时为 0（等于保持上一周期指令）
    ok / status         是否成功；"solved" / "max_iterations" / "infeasible" ...
    solve_time          求解耗时（秒）
    n_iter              qpOASES 迭代次数（WLS 为 0）
    cold_start          本次是否冷启动（WQP）
    task_residuals      任务名 → ||J v - xdot_des||
    slack               软约束名 → 松弛量 s（WQP）
    active_bounds       每个自由度：-1 下界 active，+1 上界 active，0 inactive
    active_constraints  约束名 → 每一行的 active 标志（WQP / HQP 最后一层）
    degraded            HQP：某个低优先级层求解失败，只保留了更高层的解
    levels              HQP：每层的求解信息，见 HQPSolver
    """
    v: np.ndarray
    ok: bool
    status: str
    solve_time: float
    n_iter: int = 0
    cold_start: bool = False
    task_residuals: dict = field(default_factory=dict)
    slack: dict = field(default_factory=dict)
    active_bounds: np.ndarray | None = None
    active_constraints: dict = field(default_factory=dict)
    degraded: bool = False
    levels: list = field(default_factory=list)

    def fill_task_residuals(self, problem: WBCProblem):
        for task in problem.tasks:
            self.task_residuals[task.name] = float(
                np.linalg.norm(task.residual(self.v)))
        return self
