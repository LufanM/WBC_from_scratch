"""
config.py —— 全身控制的参数：所有参数和默认值都定义在这里。

    SolverConfig   QP 求解器的参数
    WBCConfig      控制器的参数（任务跟踪、正则与姿态、限位），含一个 SolverConfig

用法：

    cfg = wf.WBCConfig(kp_pos=20.0, posture_k=0.0)
    cfg.solver.max_iter = 200
    wbc = wf.WholeBodyController(model, data, joints, sites, solver="hqp", config=cfg)

不传 config 时用这里的默认值。参数的含义和调法见 HQP_GUIDE.md。

每个参数用 param() 定义，自带说明，所以应用程序可以用 add_arguments() 自动生成命令行选项
（见 demo_hqp_interactive.py），新增参数时不需要再手写一遍。
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, fields, is_dataclass


def param(default, doc, group="", flag=None, nargs=None, type=float):
    """定义一个参数。doc 为说明；group 为命令行帮助里的分组；flag 为命令行选项名（默认
    --参数名）；nargs、type 用于默认值为 None、无法推断类型的参数。"""
    return field(default=default, metadata=dict(doc=doc, group=group, flag=flag, nargs=nargs,
                                                type=type))


@dataclass
class SolverConfig:
    """QP 求解器的参数。regularization、eps、lock_tol、rank_tol 只有 HQP 使用。"""
    preset: str = param("reliable", "qpOASES 选项：reliable / mpc / default", "求解器", type=str)
    max_iter: int = param(100, "每次（HQP 为每层）求解的最大迭代次数", "求解器", type=int)
    max_cputime: float = param(None, "每次（HQP 为每层）求解的最长时间 (s)，不给表示不限", "求解器")
    active_tol: float = param(1e-9, "判断约束 active 的乘子阈值", "求解器")
    regularization: str = param("all", "HQP 正则项放哪里：all 每层都放 / last 只放最后一层",
                                "求解器", type=str)
    eps: float = param(1e-6, "HQP 高层阻尼的相对大小", "求解器")
    lock_tol: float = param(1e-9, "HQP 上层锁定的放宽量", "求解器")
    rank_tol: float = param(1e-6, "HQP 压缩锁定行时的相对奇异值阈值", "求解器")


@dataclass
class WBCConfig:
    """WholeBodyController 的参数。"""
    # 任务跟踪
    kp_pos: float = param(12.0, "位置比例增益 (1/s)", "任务跟踪")
    kp_rot: float = param(12.0, "姿态比例增益 (1/s)", "任务跟踪")
    pos_weight: float = param(1.0, "方向权重：位置", "任务跟踪")
    rot_weight: float = param(0.4, "方向权重：姿态", "任务跟踪")
    max_pos_error: float = param(None, "位置误差限幅 (m)，不给或 0 表示不限", "任务跟踪")
    max_rot_error: float = param(None, "姿态误差限幅 (rad)，不给或 0 表示不限", "任务跟踪")
    # 正则与姿态
    lambda_reg: float = param(1e-4, "全局阻尼 λ", "正则与姿态", flag="--lambda")
    posture_w: float = param(2e-3, "W_p：每个自由度动起来的代价，分组设置用 set_posture",
                             "正则与姿态")
    posture_k: float = param(2e-3, "回中增益 k (1/s)，0 表示不回中", "正则与姿态")
    posture_kl: float = param(0.5, "限位斥力强度 k_l，0 表示不要斥力", "正则与姿态")
    limit_zone: float = param(0.1, "限位斥力区：离限位多少比例的行程内生效", "正则与姿态")
    # 限位
    max_joint_vel: float = param(2.0, "关节速度上限 (rad/s)", "限位", flag="--dq-max")
    max_base_vel: float = param(0.6, "底盘平移速度上限 (m/s)", "限位")
    limit_beta: float = param(0.5, "关节位置限位的缓冲系数 β，越小越早减速", "限位")
    solver: SolverConfig = field(default_factory=SolverConfig)


# =============================================================================
# 命令行 / 字典 <-> 参数
# =============================================================================
def _param_fields(cfg):
    return [f for f in fields(cfg) if "doc" in f.metadata]


def add_arguments(parser, cfg):
    """按 cfg（参数数据类的实例）的字段生成命令行选项，帮助里显示 cfg 中的当前值作为默认值。
    嵌套的参数数据类也会展开。没有在命令行给出的选项不会出现在解析结果里。"""
    groups = {g.title: g for g in parser._action_groups}
    for f in fields(cfg):
        value = getattr(cfg, f.name)
        if is_dataclass(value):
            add_arguments(parser, value)
            continue
        m = f.metadata
        if "doc" not in m:
            continue
        title = m["group"]
        if title not in groups:
            groups[title] = parser.add_argument_group(title)
        flag = m["flag"] or "--" + f.name.replace("_", "-")
        kw = dict(dest=f.name, default=argparse.SUPPRESS, help=f"{m['doc']} [默认 {value}]")
        if isinstance(value, bool):
            kw["action"] = argparse.BooleanOptionalAction
        else:
            kw["type"] = m["type"]
            if m["nargs"]:
                kw["nargs"] = m["nargs"]
        groups[title].add_argument(flag, **kw)


def apply_arguments(cfg, namespace):
    """把 parse_args 的结果写回 cfg（只改命令行给出的参数）。"""
    for f in fields(cfg):
        value = getattr(cfg, f.name)
        if is_dataclass(value):
            apply_arguments(value, namespace)
        elif hasattr(namespace, f.name):
            new = getattr(namespace, f.name)
            setattr(cfg, f.name, tuple(new) if isinstance(new, list) else new)


def apply_dict(cfg, values):
    """用字典（如 JSON 配置文件）覆盖 cfg，可以按字段嵌套，也可以不分层直接写参数名。"""
    names = {f.name: f for f in fields(cfg)}
    nested = [getattr(cfg, f.name) for f in fields(cfg) if is_dataclass(getattr(cfg, f.name))]
    for key, value in values.items():
        if key in names and is_dataclass(getattr(cfg, key)):
            apply_dict(getattr(cfg, key), value)
        elif key in names:
            setattr(cfg, key, tuple(value) if isinstance(value, list) else value)
        else:
            for sub in nested:
                try:
                    apply_dict(sub, {key: value})
                    break
                except KeyError:
                    pass
            else:
                raise KeyError(f"未知参数 {key!r}")


def to_dict(cfg):
    return asdict(cfg)
