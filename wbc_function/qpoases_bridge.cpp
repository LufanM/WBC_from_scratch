// =============================================================================
// qpoases_bridge.cpp —— 让 Python 能调用 qpOASES 的“转接头”
// =============================================================================
//
// 为什么需要这个文件
// ------------------
// qpOASES 是一个 C++ 库，求解器是 C++ 类 `qpOASES::SQProblem`。
// Python 自带的 ctypes 只能调用 **C 函数**，不能直接创建 C++ 对象、调用成员函数。
// 所以这里写了几个 C 函数（`extern "C"`），每个函数内部去调用 SQProblem 的对应方法：
//
//     Python (qpoases_interface.py)
//         │  ctypes 调用 C 函数
//         ▼
//     本文件：qp_create / qp_init / qp_hotstart / qp_get_primal ...
//         │  调用 C++ 成员函数
//         ▼
//     libqpOASES.so：qpOASES::SQProblem
//
// 这个文件不需要手动编译：第一次使用 WQPSolver 时，qpoases_interface.py
// 会自动用 g++ 把它编译成 .so，放在 wbc_function/_build/ 下。
//
// 求解的 QP 形式（与 QP_WBC.md §3.1 一致）
// ----------------------------------------
//     min  1/2 x^T H x + g^T x
//     s.t. lb  <=  x  <= ub          （Box 约束，nV 个）
//          lbA <= A x <= ubA         （线性约束，nC 个）
//
// 数据约定
// --------
// * 所有矩阵都是 **行优先** 存储的 double 数组（numpy 默认就是行优先）：
//   H 为 nV x nV，A 为 nC x nV。
// * 没有线性约束（nC = 0）时，A / lbA / ubA 传空指针。
// * 返回值是 qpOASES 的 returnValue 枚举（转成 int），0 表示成功；
//   可用 qp_simple_status() 转成简单状态码。
// =============================================================================

#include <qpOASES.hpp>

using qpOASES::SQProblem;

extern "C" {

// -----------------------------------------------------------------------------
// 创建求解器
//   nV      变量个数
//   nC      线性约束个数（可以为 0）
//   preset  选项预设：0 = 默认，1 = Reliable（稳健优先），2 = MPC（速度优先）
//   posdef  1 表示 H 一定正定（WBC 中有阻尼项 λI，总是成立），可跳过一些检查
// 返回：求解器句柄（不透明指针），之后所有函数都传这个句柄
// -----------------------------------------------------------------------------
void* qp_create(int nV, int nC, int preset, int posdef) {
    qpOASES::HessianType hessian_type =
        posdef ? qpOASES::HST_POSDEF : qpOASES::HST_UNKNOWN;
    SQProblem* solver = new SQProblem(nV, nC, hessian_type);

    qpOASES::Options options;
    if (preset == 1) {
        options.setToReliable();
    } else if (preset == 2) {
        options.setToMPC();
    } else {
        options.setToDefault();
    }
    options.printLevel = qpOASES::PL_NONE;  // 不往终端打印求解日志
    solver->setOptions(options);

    return solver;
}

// -----------------------------------------------------------------------------
// 释放求解器
// -----------------------------------------------------------------------------
void qp_destroy(void* handle) {
    delete static_cast<SQProblem*>(handle);
}

// -----------------------------------------------------------------------------
// 冷启动：第一次求解，或者上一次求解失败后重新开始
//   max_iter  输入：最大迭代次数（qpOASES 中叫 nWSR）
//             输出：实际用掉的迭代次数
//   cputime   可以为空指针；非空时，输入为最长允许时间（秒），输出为实际耗时
// -----------------------------------------------------------------------------
int qp_init(void* handle,
            const double* H, const double* g, const double* A,
            const double* lb, const double* ub,
            const double* lbA, const double* ubA,
            int* max_iter, double* cputime) {
    SQProblem* solver = static_cast<SQProblem*>(handle);
    qpOASES::int_t n_iter = *max_iter;
    qpOASES::returnValue ret =
        solver->init(H, g, A, lb, ub, lbA, ubA, n_iter, cputime);
    *max_iter = static_cast<int>(n_iter);
    return static_cast<int>(ret);
}

// -----------------------------------------------------------------------------
// 热启动：从上一次的解（以及哪些约束 active）出发求解新的 QP。
// 相邻控制周期的问题很接近，热启动通常只需 1~2 次迭代。
// SQProblem 允许 H 和 A 也随周期变化（WBC 中雅可比每个周期都在变）。
// 参数含义与 qp_init 相同。
// -----------------------------------------------------------------------------
int qp_hotstart(void* handle,
                const double* H, const double* g, const double* A,
                const double* lb, const double* ub,
                const double* lbA, const double* ubA,
                int* max_iter, double* cputime) {
    SQProblem* solver = static_cast<SQProblem*>(handle);
    qpOASES::int_t n_iter = *max_iter;
    qpOASES::returnValue ret =
        solver->hotstart(H, g, A, lb, ub, lbA, ubA, n_iter, cputime);
    *max_iter = static_cast<int>(n_iter);
    return static_cast<int>(ret);
}

// -----------------------------------------------------------------------------
// 取最优解 x*（长度 nV），写入调用方提供的数组
// -----------------------------------------------------------------------------
int qp_get_primal(void* handle, double* x_out) {
    SQProblem* solver = static_cast<SQProblem*>(handle);
    return static_cast<int>(solver->getPrimalSolution(x_out));
}

// -----------------------------------------------------------------------------
// 取拉格朗日乘子（长度 nV + nC）：前 nV 个对应 Box 约束，后 nC 个对应线性约束。
// 符号约定：> 0 表示下界 active，< 0 表示上界 active，= 0 表示 inactive。
// -----------------------------------------------------------------------------
int qp_get_dual(void* handle, double* y_out) {
    SQProblem* solver = static_cast<SQProblem*>(handle);
    return static_cast<int>(solver->getDualSolution(y_out));
}

// -----------------------------------------------------------------------------
// 把 qp_init / qp_hotstart 返回的详细错误码转成简单状态码：
//    0 成功
//    1 达到最大迭代次数
//   -1 其他内部错误
//   -2 不可行（约束互相矛盾）
//   -3 无界
// -----------------------------------------------------------------------------
int qp_simple_status(int return_value) {
    return static_cast<int>(qpOASES::getSimpleStatus(
        static_cast<qpOASES::returnValue>(return_value), qpOASES::BT_FALSE));
}

}  // extern "C"
