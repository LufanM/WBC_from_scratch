"""
qpoases_interface.py —— 在 Python 里使用 qpOASES 求解 QP。

分三步：
1. 找到 qpOASES 的安装位置（头文件 + 动态库）；
2. 第一次使用时，用 g++ 把 qpoases_bridge.cpp 编译成 .so（之后直接复用）；
3. 用 ctypes 加载 .so，封装成 Python 类 ``QpOasesSQP``。

一般不需要直接使用本文件：QP 的组装和对象缓存在 qp_builder.py，WQPSolver / HQPSolver 都通过它使用 qpOASES。

qpOASES 的位置按以下顺序查找：
    环境变量 QPOASES_ROOT → 当前 conda 环境 → ~/miniconda3 等常见位置 → /usr/local、/usr
每个位置都要求同时有 include/qpOASES.hpp 和 lib/libqpOASES.so。
"""
from __future__ import annotations

import ctypes
import glob
import hashlib
import os
import subprocess

import numpy as np

# qpOASES 用一个很大的数表示“无穷”（qpOASES::INFTY）
QP_INF = 1e20

_HERE = os.path.dirname(os.path.abspath(__file__))
_BRIDGE_CPP = os.path.join(_HERE, "qpoases_bridge.cpp")
_BUILD_DIR = os.path.join(_HERE, "_build")

# qp_simple_status() 的返回值 → 可读的状态名
_STATUS_NAMES = {
    0: "solved",
    1: "max_iterations",
    -1: "error",
    -2: "infeasible",
    -3: "unbounded",
}

# 选项预设名 → qp_create() 的 preset 参数
_PRESETS = {"default": 0, "reliable": 1, "mpc": 2}


# =============================================================================
# 第 1 步：找到 qpOASES
# =============================================================================
def find_qpoases_root():
    """返回 qpOASES 的安装前缀（含 include/ 和 lib/ 的目录）。"""
    candidates = []
    if os.environ.get("QPOASES_ROOT"):
        candidates.append(os.environ["QPOASES_ROOT"])
    if os.environ.get("CONDA_PREFIX"):
        candidates.append(os.environ["CONDA_PREFIX"])

    home = os.path.expanduser("~")
    for conda in ("miniconda3", "anaconda3", "miniforge3", "mambaforge"):
        base = os.path.join(home, conda)
        candidates.append(base)                                        # base 环境
        candidates += sorted(glob.glob(os.path.join(base, "envs", "*")))   # 其他环境
        candidates += sorted(glob.glob(os.path.join(base, "pkgs", "qpoases-*")),
                             reverse=True)                             # 包缓存
    candidates += ["/usr/local", "/usr", "/opt/qpOASES"]

    for root in candidates:
        has_header = os.path.isfile(os.path.join(root, "include", "qpOASES.hpp"))
        has_lib = glob.glob(os.path.join(root, "lib", "libqpOASES.so*"))
        if has_header and has_lib:
            return root

    raise RuntimeError(
        "找不到 qpOASES。请设置环境变量 QPOASES_ROOT，指向包含 "
        "include/qpOASES.hpp 和 lib/libqpOASES.so 的目录"
        "（例如 `conda install -c conda-forge qpoases` 后的 conda 环境目录）。")


# =============================================================================
# 第 2 步：编译桥接库
# =============================================================================
def build_bridge(qpoases_root):
    """把 qpoases_bridge.cpp 编译成 .so，返回 .so 的路径。

    文件名里带有“源码 + qpOASES 路径”的哈希：源码改了或 qpOASES 换了位置，
    会自动重新编译；否则直接复用已有的 .so。
    """
    with open(_BRIDGE_CPP, "r", encoding="utf-8") as f:
        source = f.read()
    tag = hashlib.sha1((source + qpoases_root).encode()).hexdigest()[:10]
    so_path = os.path.join(_BUILD_DIR, f"libqpoases_bridge_{tag}.so")
    if os.path.exists(so_path):
        return so_path

    os.makedirs(_BUILD_DIR, exist_ok=True)
    lib_dir = os.path.join(qpoases_root, "lib")
    cmd = [
        "g++", "-O2", "-shared", "-fPIC", "-std=c++11",
        _BRIDGE_CPP,
        "-I", os.path.join(qpoases_root, "include"),   # 头文件
        "-L", lib_dir, "-lqpOASES",                    # 链接 libqpOASES.so
        f"-Wl,-rpath,{lib_dir}",                       # 运行时到这里找 libqpOASES.so
        "-o", so_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError("编译 qpOASES 桥接库失败：\n" + " ".join(cmd)
                           + "\n" + proc.stderr)
    return so_path


# =============================================================================
# 第 3 步：加载 .so，声明每个 C 函数的参数类型
# =============================================================================
_LIB = None


def load_bridge():
    """返回加载好的桥接库（每个进程只编译、加载一次）。"""
    global _LIB
    if _LIB is not None:
        return _LIB

    lib = ctypes.CDLL(build_bridge(find_qpoases_root()))

    c_int, c_void_p = ctypes.c_int, ctypes.c_void_p
    double_ptr = ctypes.POINTER(ctypes.c_double)
    int_ptr = ctypes.POINTER(ctypes.c_int)

    # void* qp_create(int nV, int nC, int preset, int posdef)
    lib.qp_create.restype = c_void_p
    lib.qp_create.argtypes = [c_int, c_int, c_int, c_int]

    # void qp_destroy(void* handle)
    lib.qp_destroy.argtypes = [c_void_p]

    # int qp_init / qp_hotstart(handle, H, g, A, lb, ub, lbA, ubA, max_iter*, cputime*)
    for fn in (lib.qp_init, lib.qp_hotstart):
        fn.restype = c_int
        fn.argtypes = [c_void_p] + [double_ptr] * 7 + [int_ptr, double_ptr]

    # int qp_get_primal / qp_get_dual(handle, double* out)
    for fn in (lib.qp_get_primal, lib.qp_get_dual):
        fn.restype = c_int
        fn.argtypes = [c_void_p, double_ptr]

    # int qp_simple_status(int return_value)
    lib.qp_simple_status.restype = c_int
    lib.qp_simple_status.argtypes = [c_int]

    _LIB = lib
    return lib


def _as_c_array(a):
    """numpy 数组 → C 的 double*。None 表示空指针（例如没有线性约束时的 A）。"""
    if a is None:
        return None
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


# =============================================================================
# 封装：一个 qpOASES SQProblem
# =============================================================================
class QpOasesSQP:
    """持有一个 qpOASES ``SQProblem``，变量数 nV 和约束数 nC 固定。

    用法::

        qp = QpOasesSQP(nV=3, nC=1)
        x, dual, status, n_iter, cold = qp.solve(H, g, lb, ub, A, lbA, ubA)

    第一次 ``solve`` 冷启动，之后自动热启动；热启动失败会自动退回冷启动。
    所有数组按 numpy 默认的行优先传入即可。
    """

    def __init__(self, nV, nC, preset="reliable", hessian_posdef=True):
        self.lib = load_bridge()
        self.nV = int(nV)
        self.nC = int(nC)
        self._handle = self.lib.qp_create(self.nV, self.nC, _PRESETS[preset],
                                          int(bool(hessian_posdef)))
        self.initialized = False   # 是否已有可用于热启动的上一次解

    def __del__(self):
        handle = getattr(self, "_handle", None)
        if handle:
            self.lib.qp_destroy(handle)
            self._handle = None

    def solve(self, H, g, lb, ub, A=None, lbA=None, ubA=None,
              max_iter=100, max_cputime=None):
        """求解一次 QP。

        返回 ``(x, dual, status, n_iter, cold_start)``：
            x           最优解（失败时为 0）
            dual        拉格朗日乘子，长度 nV + nC（失败时为 0）
            status      "solved" / "max_iterations" / "infeasible" / ...
            n_iter      实际迭代次数
            cold_start  本次是否为冷启动
        """
        def as_c_contiguous(a):
            return None if a is None else np.ascontiguousarray(a, dtype=np.float64)

        H, g = as_c_contiguous(H), as_c_contiguous(g)
        lb, ub = as_c_contiguous(lb), as_c_contiguous(ub)
        if self.nC > 0:
            A, lbA, ubA = (as_c_contiguous(A), as_c_contiguous(lbA),
                           as_c_contiguous(ubA))
        else:
            A = lbA = ubA = None

        def call(c_function):
            n_iter = ctypes.c_int(max_iter)
            cputime = None if max_cputime is None else ctypes.c_double(max_cputime)
            ret = c_function(
                self._handle,
                _as_c_array(H), _as_c_array(g), _as_c_array(A),
                _as_c_array(lb), _as_c_array(ub),
                _as_c_array(lbA), _as_c_array(ubA),
                ctypes.byref(n_iter),
                None if cputime is None else ctypes.byref(cputime))
            return self.lib.qp_simple_status(ret), n_iter.value

        cold_start = not self.initialized
        if self.initialized:
            status, n_iter = call(self.lib.qp_hotstart)
            if status != 0:          # 热启动失败 → 冷启动重来
                cold_start = True
        if cold_start:
            status, n_iter = call(self.lib.qp_init)
        self.initialized = (status == 0)

        x = np.zeros(self.nV)
        dual = np.zeros(self.nV + self.nC)
        if status == 0:
            self.lib.qp_get_primal(self._handle, _as_c_array(x))
            self.lib.qp_get_dual(self._handle, _as_c_array(dual))
        return (x, dual, _STATUS_NAMES.get(status, f"status_{status}"),
                n_iter, cold_start)
