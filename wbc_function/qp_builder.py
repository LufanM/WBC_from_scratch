"""
qp_builder.py —— 把目标函数和约束组装成 qpOASES 的标准形式（WQP、HQP 共用）。

决策变量 z = [v; s]，s 为软约束的松弛量（QP_WBC.md §6.3）：

    min  1/2 z^T H z + g^T z
    s.t. lb  <= z   <= ub          v 的上下界；s >= 0
         lbA <= A z <= ubA

约束矩阵 A 的行依次为：

    硬约束          [A_c  0]                    lb_c <= A_c v <= ub_c
    等式块（可选）  [A_eq 0]                    b_eq - tol <= A_eq v <= b_eq + tol
                                                （HQP 的上层锁定，QP_WBC.md §8.2）
    软约束（每行拆成两行）
                    [A_s  +I]  >= lb_s          A_s v + s >= lb_s
                    [A_s  -I]  <= ub_s          A_s v - s <= ub_s
                    目标函数加 w_lin * sum(s) + w_quad/2 * ||s||^2
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .qpoases_interface import QP_INF, QpOasesSQP


def finite(a):
    """把 +/-inf 换成 qpOASES 的“无穷”（1e20）。"""
    return np.clip(np.asarray(a, float), -QP_INF, QP_INF)


@dataclass
class QPData:
    """组装好、可以直接交给 qpOASES 的 QP。"""
    n: int                 # v 的维数（z 的前 n 个分量）
    H: np.ndarray
    g: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    A: np.ndarray
    lbA: np.ndarray
    ubA: np.ndarray
    hard_rows: list        # [(约束, A 中的起始行)]
    soft_rows: list        # [(约束, A 中的起始行, s 在 z 中的起始下标)]
    hessian_posdef: bool   # 软约束的 w_quad 全为正时 H 正定

    def slacks(self, z):
        """[(软约束, 它的松弛量 s)]。"""
        return [(c, z[s_col:s_col + c.rows].copy()) for c, _, s_col in self.soft_rows]


def build_qp(n, H_v, g_v, lb_v, ub_v, hard=(), soft=(), A_eq=None, b_eq=None,
             eq_tol=0.0) -> QPData:
    """组装 QP（行的排列见文件开头）。

    H_v, g_v      v 部分的目标函数
    lb_v, ub_v    v 的上下界
    hard / soft   LinearConstraint 列表
    A_eq, b_eq    可选的等式块 A_eq v = b_eq，放宽为 ±eq_tol 的窄区间
    """
    hard, soft = list(hard), list(soft)
    m_eq = 0 if A_eq is None else A_eq.shape[0]
    n_slack = sum(c.rows for c in soft)
    nV = n + n_slack
    nC = sum(c.rows for c in hard) + m_eq + 2 * n_slack

    H = np.zeros((nV, nV))
    g = np.zeros(nV)
    H[:n, :n] = H_v
    g[:n] = g_v
    lb = np.concatenate([finite(lb_v), np.zeros(n_slack)])
    ub = np.concatenate([finite(ub_v), np.full(n_slack, QP_INF)])
    A = np.zeros((nC, nV))
    lbA = np.full(nC, -QP_INF)
    ubA = np.full(nC, QP_INF)

    row = 0
    hard_rows = []
    for c in hard:
        A[row:row + c.rows, :n] = c.A
        lbA[row:row + c.rows] = finite(c.lb)
        ubA[row:row + c.rows] = finite(c.ub)
        hard_rows.append((c, row))
        row += c.rows

    if m_eq:
        A[row:row + m_eq, :n] = A_eq
        lbA[row:row + m_eq] = b_eq - eq_tol
        ubA[row:row + m_eq] = b_eq + eq_tol
        row += m_eq

    s_col = n
    soft_rows = []
    for c in soft:
        k = c.rows
        s_idx = s_col + np.arange(k)
        H[s_idx, s_idx] = c.w_quad                   # 二次罚
        g[s_idx] = c.w_lin                           # 线性罚
        lower = row + np.arange(k)                   # A v + s >= lb
        A[lower, :n] = c.A
        A[lower, s_idx] = 1.0
        lbA[lower] = finite(c.lb)
        upper = row + k + np.arange(k)               # A v - s <= ub
        A[upper, :n] = c.A
        A[upper, s_idx] = -1.0
        ubA[upper] = finite(c.ub)
        soft_rows.append((c, row, s_col))
        row += 2 * k
        s_col += k

    posdef = all(c.w_quad > 0 for c in soft)
    return QPData(n, H, g, lb, ub, A, lbA, ubA, hard_rows, soft_rows, posdef)


def read_back(result, qp: QPData, z, dual, active_tol):
    """把解中的松弛量和 active 标志写进 WBCResult。

    qpOASES 对偶解的符号：> 0 下界 active，< 0 上界 active。
    这里统一转成：-1 下界 active，+1 上界 active，0 inactive。
    """
    def flags(y):
        return (y < -active_tol).astype(int) - (y > active_tol).astype(int)

    nV = qp.H.shape[0]
    result.active_bounds = flags(dual[:qp.n])
    dual_rows = dual[nV:]                            # 线性约束部分的乘子
    for c, row in qp.hard_rows:
        result.active_constraints[c.name] = flags(dual_rows[row:row + c.rows])
    for c, row, s_col in qp.soft_rows:
        lower_active = dual_rows[row:row + c.rows] > active_tol
        upper_active = dual_rows[row + c.rows:row + 2 * c.rows] < -active_tol
        result.active_constraints[c.name] = upper_active.astype(int) - lower_active.astype(int)
        result.slack[c.name] = z[s_col:s_col + c.rows].copy()


class QPCache:
    """按名字缓存 qpOASES 对象，以便逐周期热启动。

    同一个名字下 QP 的形状（变量数、约束数）或 Hessian 类型变了，就重建对象（冷启动）。
    WQP 只用一个名字；HQP 每层一个名字（层的 priority），某一层暂时不存在时，
    其他层的热启动不受影响。
    """

    def __init__(self, preset="reliable", max_iter=100, max_cputime=None):
        self.preset = preset
        self.max_iter = max_iter
        self.max_cputime = max_cputime
        self._qps = {}           # 名字 -> ((nV, nC, posdef), QpOasesSQP)

    def reset(self):
        self._qps = {}

    def solve(self, key, qp: QPData):
        """返回 (z, dual, status, n_iter, cold_start)，含义同 QpOasesSQP.solve。"""
        nC, nV = qp.A.shape
        shape = (nV, nC, qp.hessian_posdef)
        cached = self._qps.get(key)
        if cached is None or cached[0] != shape:
            cached = (shape, QpOasesSQP(nV, nC, self.preset, qp.hessian_posdef))
            self._qps[key] = cached
        return cached[1].solve(qp.H, qp.g, qp.lb, qp.ub,
                               qp.A if nC > 0 else None, qp.lbA, qp.ubA,
                               self.max_iter, self.max_cputime)
