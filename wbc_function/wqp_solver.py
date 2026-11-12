"""
wqp_solver.py —— 加权 QP（WQP）求解器，用 qpOASES 求解。

原理见 QP_WBC.md §7：目标函数与加权最小二乘相同，额外加入不等式约束

    min  1/2 v^T H v + g^T v
    s.t. lb <= v <= ub                 关节位置 + 速度限位、冻结
         lbA <= A v <= ubA             硬线性约束
         lbA - s <= A v <= ubA + s     软线性约束（s >= 0 为松弛变量）

软约束的处理（QP_WBC.md §6.3）：决策变量扩展为 z = [v; s]，每一行软约束拆成两行
单边约束，并在目标函数中加上 w_lin * sum(s) + w_quad/2 * ||s||^2。组装细节见 qp_builder.py。
"""
from __future__ import annotations

import time

import numpy as np

from .problem import WBCProblem, WBCResult
from .qp_builder import QPCache, build_qp, read_back


class WQPSolver:
    """加权 QP 求解器（qpOASES SQProblem，每个周期自动热启动）。

    preset       qpOASES 选项："reliable"（稳健，默认）/ "mpc"（快）/ "default"
    max_iter     每次求解的最大迭代次数（qpOASES 中的 nWSR）
    max_cputime  每次求解的最长时间（秒），None 表示不限
    active_tol   判断约束 active 时，拉格朗日乘子的阈值

    使用注意：
    * 问题的“形状”（自由度数、约束行数、软约束行数）变化时，会重建 qpOASES
      对象并冷启动。想临时关掉某个约束，把它的边界设为 +/-inf，而不是删掉它；
      任务的增删不影响形状。
    * 求解失败时 result.ok = False、result.v = 0（保持上一周期指令），
      下一周期自动冷启动。
    """

    def __init__(self, preset="reliable", max_iter=100, max_cputime=None,
                 active_tol=1e-9):
        self.active_tol = active_tol
        self._qp = QPCache(preset, max_iter, max_cputime)

    def reset(self):
        """丢弃热启动信息，下一次求解为冷启动。"""
        self._qp.reset()

    def solve(self, problem: WBCProblem) -> WBCResult:
        t_start = time.perf_counter()
        n = problem.n

        H_v, g_v = problem.objective()
        qp = build_qp(n, H_v, g_v, problem.lb, problem.ub,
                      hard=[c for c in problem.constraints if not c.soft],
                      soft=[c for c in problem.constraints if c.soft])
        z, dual, status, n_iter, cold_start = self._qp.solve("wqp", qp)
        ok = status == "solved"

        result = WBCResult(v=z[:n] if ok else np.zeros(n), ok=ok, status=status,
                           solve_time=time.perf_counter() - t_start,
                           n_iter=n_iter, cold_start=cold_start)
        if ok:
            read_back(result, qp, z, dual, self.active_tol)
        return result.fill_task_residuals(problem)
