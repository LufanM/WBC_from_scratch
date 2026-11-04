#!/usr/bin/env python3
"""
active_set_demo.py —— 最简单的 Active Set 求解器（教学用），打印每一步迭代。

求解
    min  1/2 x^T H x + g^T x
    s.t. A x <= b

配合 QP_KKT_GUIDE.md 阅读。算法（原始 active set 法）：

    从一个满足所有约束的点 x 和一个“贴墙集合” W 出发，循环：
      1. 把 W 里的约束当作等式，解等式约束 QP（一个线性方程组）→ x_W 和 μ
      2. 如果 x_W != x：从 x 朝 x_W 走，碰到第一堵墙就停，把它加入 W
      3. 如果 x_W == x：检查 μ；有 μ < 0 就把那堵墙移出 W（球想离开它）
      4. 没有墙可加、也没有 μ < 0 → 满足 KKT 条件，得到最优解

只用于理解原理；实际控制中用 wbc_function.WQPSolver（qpOASES）。

运行：python3 tools/active_set_demo.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

TOL = 1e-9


def solve_equality_qp(H, g, A_W, b_W):
    """把 W 中的约束当作等式，解 KKT 线性方程组：

        [ H   A_W^T ] [ x ]   [ -g  ]
        [ A_W   0   ] [ μ ] = [ b_W ]
    """
    n, m = H.shape[0], A_W.shape[0]
    K = np.zeros((n + m, n + m))
    K[:n, :n] = H
    K[:n, n:] = A_W.T
    K[n:, :n] = A_W
    rhs = np.concatenate([-g, b_W])
    sol = np.linalg.solve(K, rhs)
    return sol[:n], sol[n:]


def active_set_qp(H, g, A, b, x0, W0=(), names=None, max_iter=20):
    """原始 active set 法。x0 必须满足 A x0 <= b。返回 (x, W, μ)。"""
    H, g, A, b = (np.asarray(v, float) for v in (H, g, A, b))
    x = np.asarray(x0, float).copy()
    W = list(W0)
    names = names or [f"约束{i}" for i in range(len(b))]
    fmt = lambda v: "[" + ", ".join(f"{e:.2f}" for e in v) + "]"
    wnames = lambda: "{" + ", ".join(names[i] for i in W) + "}"

    print(f"  起点 x = {fmt(x)}，W = {wnames()}")
    for it in range(1, max_iter + 1):
        x_W, mu = solve_equality_qp(H, g, A[W], b[W])
        step = x_W - x
        print(f"  迭代 {it}：把 W = {wnames()} 当作等式求解 → x_W = {fmt(x_W)}"
              + (f"，μ = {fmt(mu)}" if W else ""))

        if np.linalg.norm(step) > TOL:
            # 第 2 步：朝 x_W 走，碰到第一堵墙就停（比值检验）
            alpha, blocking = 1.0, None
            for i in range(len(b)):
                if i in W:
                    continue
                a_p = A[i] @ step
                if a_p > TOL:                       # 朝这堵墙的外侧走
                    alpha_i = (b[i] - A[i] @ x) / a_p
                    if alpha_i < alpha:
                        alpha, blocking = alpha_i, i
            x = x + alpha * step
            if blocking is not None:
                W.append(blocking)
                print(f"           朝 x_W 走到 {alpha:.0%} 时碰到 {names[blocking]}，"
                      f"停在 x = {fmt(x)}，把它加入 W")
            else:
                print(f"           一路没碰墙，走到 x = {fmt(x)}")
            continue

        # 第 3 步：x 已是当前 W 下的最优点，检查 μ 的符号
        if len(W) == 0 or mu.min() >= -TOL:
            print(f"  → 所有 μ ≥ 0，也没有穿墙：最优解 x = {fmt(x)}，"
                  f"贴着的墙 W = {wnames()}")
            return x, W, mu
        k = int(np.argmin(mu))
        print(f"           {names[W[k]]} 的 μ = {mu[k]:.2f} < 0：球想离开这堵墙，"
              f"把它移出 W")
        W.pop(k)
    raise RuntimeError("超过最大迭代次数")


def main():
    H = np.eye(2)

    # 例 1：目标 [2, 2]，三堵墙；从原点冷启动
    A = np.array([[1, 0], [0, 1], [1, 1]], float)
    b = np.array([1.5, 1.5, 2.0])
    names = ["①Δq1≤1.5", "②Δq2≤1.5", "③Δq1+Δq2≤2"]
    print("例 1：目标 [2, 2]，冷启动（从原点、W 为空开始）")
    x, W, _ = active_set_qp(H, -np.array([2.0, 2.0]), A, b, [0, 0], (), names)

    # 例 2：下一周期目标变成 [0.5, 0.5]；用例 1 的结果热启动
    print("\n例 2：目标变成 [0.5, 0.5]，热启动（从上一周期的 x 和 W 开始）")
    active_set_qp(H, -np.array([0.5, 0.5]), A, b, x, W, names)

    # 例 3：目标 [3, 1]，两堵墙，最优解在角点
    A2 = np.array([[1, 0], [1, 1]], float)
    b2 = np.array([1.5, 2.0])
    names2 = ["①Δq1≤1.5", "③Δq1+Δq2≤2"]
    print("\n例 3：目标 [3, 1]，两堵墙，冷启动")
    x3, _, _ = active_set_qp(H, -np.array([3.0, 1.0]), A2, b2, [0, 0], (),
                             names2)

    # 与 qpOASES 对照
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        import wbc_function as wf
    except Exception as exc:                       # qpOASES 不可用时跳过
        print(f"\n（跳过 qpOASES 对照：{exc}）")
        return
    prob = wf.WBCProblem(2, lambda_reg=0.0)
    prob.add_task(wf.Task("target", np.eye(2), [3.0, 1.0]))
    prob.add_constraint(wf.LinearConstraint("walls", A2, -np.inf, b2))
    res = wf.WQPSolver().solve(prob)
    print(f"\n对照：qpOASES 求解例 3 → {np.round(res.v, 4)}，"
          f"与上面的结果相差 {np.max(np.abs(res.v - x3)):.1e}")


if __name__ == "__main__":
    main()
