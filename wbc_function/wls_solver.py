"""
wls_solver.py —— 加权最小二乘（Weighted Damped Least Squares）求解器。

原理见 QP_WBC.md §1：没有不等式约束，目标函数的最优解有闭式解

    H v = -g     （H、g 由 WBCProblem.objective() 给出）

与项目最初的加权最小二乘实现完全等价（WholeBodyController 默认使用它）。

由于没有不等式约束，问题中的上下界（关节限位、速度限位）不参与求解，
只能在求解之后处理，见 saturation 参数。需要在求解时就满足约束，请用 WQPSolver。

例外：上下界相等的变量（lb = ub，如冻结的关节）是等式约束，可以精确处理——
把它固定为该值，只对其余变量求解（消元）：

    H_ff v_f = -(g_f + H_fx v_x)      f：自由变量，x：固定变量
"""
from __future__ import annotations

import time
import warnings

import numpy as np

from .problem import WBCProblem, WBCResult


class WLSSolver:
    """加权最小二乘求解器。

    saturation  求解后如何让 v 落在上下界内：
        "scale"  整体等比例缩小 v，直到所有分量都在界内。保持运动方向（默认）。
                 只有当 v = 0 在界内时才有效，否则退回 "clip"。
        "clip"   逐个自由度裁剪，与项目最初的实现一致。会改变运动方向。
        None     不处理，直接返回闭式解。
    """

    def __init__(self, saturation="scale"):
        if saturation not in ("scale", "clip", None):
            raise ValueError("saturation 只能是 'scale'、'clip' 或 None")
        self.saturation = saturation
        self._warned_constraints = False

    def solve(self, problem: WBCProblem) -> WBCResult:
        t_start = time.perf_counter()
        if problem.constraints and not self._warned_constraints:
            self._warned_constraints = True
            warnings.warn(f"WLSSolver 不处理线性约束，{[c.name for c in problem.constraints]} "
                          f"被忽略；需要约束请用 WQPSolver / HQPSolver", stacklevel=2)
        H, g = problem.objective()

        # 上下界相等的变量（冻结）固定下来，只解其余变量
        fixed = np.isfinite(problem.lb) & (problem.lb == problem.ub)
        free = ~fixed
        v = np.zeros(problem.n)
        v[fixed] = problem.lb[fixed]
        try:
            if np.any(free):
                rhs = -(g[free] + H[np.ix_(free, fixed)] @ v[fixed])
                v[free] = np.linalg.solve(H[np.ix_(free, free)], rhs)
            ok = bool(np.all(np.isfinite(v)))
            status = "solved" if ok else "error"
        except np.linalg.LinAlgError:     # H 奇异：通常是 lambda_reg = 0
            ok, status = False, "singular"

        if ok:
            v = self._saturate(v, problem.lb, problem.ub)
        else:
            v = np.zeros(problem.n)

        result = WBCResult(v=v, ok=ok, status=status,
                           solve_time=time.perf_counter() - t_start)
        result.active_bounds = (np.isclose(v, problem.ub).astype(int)
                                - np.isclose(v, problem.lb).astype(int))
        return result.fill_task_residuals(problem)

    def _saturate(self, v, lb, ub):
        if self.saturation is None:
            return v
        if self.saturation == "clip":
            return np.clip(v, lb, ub)

        # "scale"：找最大的 s <= 1，使 s * v 在 [lb, ub] 内。
        # 只有 0 在界内时才可能（s 缩到 0 总能满足）；否则（如关节已越过限位、
        # 必须往回走）缩放解决不了，退回裁剪。
        if np.any(lb > 0.0) or np.any(ub < 0.0):
            return np.clip(v, lb, ub)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_upper = np.where(v > ub, ub / v, 1.0)
            ratio_lower = np.where(v < lb, lb / v, 1.0)
        s = float(min(1.0, ratio_upper.min(), ratio_lower.min()))
        return s * v
