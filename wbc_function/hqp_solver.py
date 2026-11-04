"""
hqp_solver.py —— 分层 QP（HQP，字典序优化）求解器，每层用 qpOASES 求解。

原理见 QP_WBC.md §8，完整例子见 §8.7。任务按 Task.priority 分层（数字越小优先级越高），
软约束按 LinearConstraint.priority 放进对应的层；从高到低逐层求解，第 k 层为：

    min  1/2 sum_{i in 第 k 层} w_i ||J_i v - xdot_i||^2_{W_i} + (本层正则项)
         + 本层软约束的罚函数
    s.t. lb <= v <= ub                         关节限位、冻结（每层都有）
         lbA <= A v <= ubA                     硬线性约束（每层都有）
         J_j v = J_j v*          j 在上面各层   上层锁定：上层能做到的程度不能被破坏
         lb_S - s* <= S v <= ub_S + s*         上层软约束，违反量固定为该层求得的 s*

本层正则项（QP_WBC.md §8.3），由 regularization 参数选择：
    "last"（默认，严格分层）
        高层（不是最后一层）  eps/2 ||v||^2，eps 取本层任务量级的 1e-6，只为保证 H 正定
        最后一层              WBCProblem.regularization()：λ/2||v||^2 + 1/2||v - v_posture||^2_{W_p}
    "all"（带阻尼的分层，实际控制推荐）
        每一层都加 WBCProblem.regularization()（高层另加 eps）。高层任务接近奇异时，
        只有 eps 的高层相当于不带阻尼的伪逆：目标够不到时解会在速度上限之间逐周期翻转；
        每层带上 λ 和姿态项（含限位斥力）后不再翻转。代价是高层任务多了一点由正则项
        引起的残差（与 WQP 中正则项的影响同量级），优先级仍然成立。

实现要点（与主流 HQP 框架的对应）：
  * 逐层求解、上层任务作为下层的等式约束（OpenSoT iHQP 的做法）；
  * 锁定的行先做 SVD，只保留线性无关的单位行，锁定值取上一层的解（各行严格自洽），
    并放宽为 ±lock_tol 的窄区间，避免退化的等式约束（LexLS / Escande 等用 QR / COD
    做同样的事，QP_WBC.md §8.6）；只锁定权重 > 0 的行；
  * 每层一个 qpOASES 对象，按层的 priority 缓存，逐周期热启动；
  * 上层锁定后已没有剩余自由度时，跳过下面所有层；
  * 低优先级层求解失败时，保留上一层的解并标记 degraded（它满足所有硬约束和更高层任务）。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .problem import Task, WBCProblem, WBCResult
from .qp_builder import QPCache, build_qp, read_back


@dataclass
class Level:
    """一层：同一 priority 的任务和软约束。"""
    priority: int
    tasks: list
    soft: list


@dataclass
class LevelInfo:
    """一层的求解信息（WBCResult.levels 的元素）。

    status      "solved" / "skipped"（上层已无剩余自由度）/ qpOASES 的失败状态
    residual    本层任务残差 ||J v - xdot||（本层求解后），即 QP_WBC.md 中的 |r_k*|
    n_locked    本层求解时，来自上层的独立锁定行数
    eps         本层的高层阻尼（最后一层为 0）
    """
    priority: int
    tasks: list = field(default_factory=list)
    status: str = ""
    residual: float = 0.0
    n_locked: int = 0
    eps: float = 0.0
    n_iter: int = 0
    cold_start: bool = False

    @property
    def skipped(self):
        return self.status == "skipped"


class HQPSolver:
    """分层 QP 求解器。

    preset      qpOASES 选项："reliable"（稳健，默认）/ "mpc"（快）/ "default"
    max_iter    每层每次求解的最大迭代次数（nWSR）
    max_cputime 每层每次求解的最长时间（秒），None 表示不限
    regularization  "last"：正则项只放最后一层；"all"：每层都放（见文件开头）
    eps         高层阻尼的相对大小：eps_k = eps * max(1, 本层 J^T W J 的最大对角元)
    lock_tol    上层锁定的放宽量（速度量纲）：|行 · v - 锁定值| <= lock_tol
    rank_tol    SVD 压缩锁定行时的相对奇异值阈值：比最大奇异值小这么多倍的方向
                不锁定（上层任务几乎不受它影响，锁定它只会白白占用下层的自由度）
    active_tol  判断约束 active 时，拉格朗日乘子的阈值

    结果中：
      result.levels   每层一个 LevelInfo
      result.degraded 某个低优先级层失败，只保留了更高层的解（result.ok 仍为 True）
      result.slack    每个软约束在它所在层求得的 s*
      active 标志取自最后一个求解成功的层
    """

    def __init__(self, preset="reliable", max_iter=100, max_cputime=None,
                 regularization="last", eps=1e-6, lock_tol=1e-9, rank_tol=1e-6,
                 active_tol=1e-9):
        if regularization not in ("last", "all"):
            raise ValueError("regularization 只能是 'last' 或 'all'")
        self.regularization = regularization
        self.eps = eps
        self.lock_tol = lock_tol
        self.rank_tol = rank_tol
        self.active_tol = active_tol
        self._qp = QPCache(preset, max_iter, max_cputime)

    def reset(self):
        """丢弃所有层的热启动信息。"""
        self._qp.reset()

    # -------------------------------------------------------------------------
    def solve(self, problem: WBCProblem) -> WBCResult:
        t_start = time.perf_counter()
        n = problem.n
        levels = self._split_levels(problem)
        hard = [c for c in problem.constraints if not c.soft]
        free = ~(np.isfinite(problem.lb) & (problem.lb == problem.ub))   # 未冻结的自由度

        infos = [LevelInfo(lv.priority, [t.name for t in lv.tasks]) for lv in levels]
        locked = []          # 上面各层要锁定的行（雅可比）
        fixed_soft = []      # 上面各层的软约束，已按 s* 放宽成硬约束
        slack = {}
        v = np.zeros(n)      # 最近一个成功层的解
        last = None          # 最近一个成功层的 (qp, z, dual)
        degraded = False

        for k, (level, info) in enumerate(zip(levels, infos)):
            is_last = k == len(levels) - 1

            # 1. 上层锁定：压缩成独立的单位行；已没有剩余自由度时，下面的层都跳过
            A_lock = self._independent_rows(locked, n)
            info.n_locked = A_lock.shape[0]
            if k > 0 and self._rank(A_lock[:, free]) >= free.sum():
                for rest in infos[k:]:
                    rest.status = "skipped"
                break

            # 2. 本层目标函数 + 组装 + 求解
            H_v, g_v = self._objective(problem, level, is_last, info)
            qp = build_qp(n, H_v, g_v, problem.lb, problem.ub,
                          hard=hard + fixed_soft, soft=level.soft,
                          A_eq=A_lock, b_eq=A_lock @ v, eq_tol=self.lock_tol)
            z, dual, info.status, info.n_iter, info.cold_start = \
                self._qp.solve(level.priority, qp)
            if info.status != "solved":
                degraded = k > 0     # 低层失败：保留上一层的解；最高层失败：整体失败
                break

            v, last = z[:n].copy(), (qp, z, dual)
            info.residual = _residual_norm(level.tasks, v)

            # 3. 锁定本层：任务的 J v 在下层固定为当前值（只锁权重 > 0 的行）；
            #    软约束的违反量固定为 s*，之后当作硬约束
            locked += [t.J[t.weights() > 0] for t in level.tasks]
            for c, s in qp.slacks(z):
                slack[c.name] = s
                fixed_soft.append(c.relaxed(s))

        ok = last is not None
        if ok:
            status = "degraded" if degraded else "solved"
        else:
            status = infos[0].status
        result = WBCResult(v=v if ok else np.zeros(n), ok=ok, status=status,
                           solve_time=time.perf_counter() - t_start,
                           n_iter=sum(i.n_iter for i in infos),
                           cold_start=any(i.cold_start for i in infos),
                           degraded=degraded, levels=infos)
        if ok:
            read_back(result, *last, self.active_tol)
            result.slack = slack
        return result.fill_task_residuals(problem)

    # -------------------------------------------------------------------------
    @staticmethod
    def _split_levels(problem: WBCProblem):
        """按 priority 从小到大分层。没有任何任务和软约束时，只有一层（只含正则项）。"""
        soft = [c for c in problem.constraints if c.soft]
        priorities = sorted({t.priority for t in problem.tasks} | {c.priority for c in soft})
        return [Level(p, [t for t in problem.tasks if t.priority == p],
                      [c for c in soft if c.priority == p])
                for p in priorities or [0]]

    def _objective(self, problem: WBCProblem, level: Level, is_last, info: LevelInfo):
        """本层的 (H, g)：本层任务 + 正则项（见文件开头）+ 高层的 eps。"""
        H, g = problem.task_objective(level.tasks)
        if is_last or self.regularization == "all":
            H_reg, g_reg = problem.regularization()
            H, g = H + H_reg, g + g_reg
        if not is_last:
            info.eps = self.eps * max(1.0, float(np.max(np.diag(H), initial=0.0)))
            H = H + info.eps * np.eye(problem.n)
        return H, g

    def _independent_rows(self, locked, n):
        """把上层锁定的行压缩成线性无关的单位行 V_r^T（SVD：J = U S V^T）。

        锁定 J v = J v_prev 与锁定 V_r^T v = V_r^T v_prev 等价（只差被丢掉的、
        奇异值很小的方向），后者的行线性无关、条件数为 1。
        """
        J = np.vstack(locked) if locked else np.zeros((0, n))
        if J.shape[0] == 0:
            return J
        _, S, Vt = np.linalg.svd(J, full_matrices=False)
        return Vt[:self._count_significant(S)]

    def _rank(self, M):
        if M.size == 0:
            return 0
        return self._count_significant(np.linalg.svd(M, compute_uv=False))

    def _count_significant(self, S):
        """奇异值中不可忽略的个数（相对最大奇异值，且不小于 rank_tol 的绝对量）。"""
        return int(np.sum(S > self.rank_tol * max(1.0, S[0]))) if S.size else 0


def _residual_norm(tasks: list[Task], v):
    if not tasks:
        return 0.0
    return float(np.linalg.norm(np.concatenate([t.residual(v) for t in tasks])))
