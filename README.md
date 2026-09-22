# WBC

轮式双臂机器人（Galaxea R1 Pro）全身控制（Whole-Body Control）实现与原理研究，基于 MuJoCo。

## 本仓库做什么

让机械臂末端精确到达指定空间位姿，同时利用冗余自由度（零空间）完成次任务（姿态优化、臂角摆动等）。目标可能超出手臂可达范围，需要底盘、躯干、手臂协同运动来完成任务。

冗余机器人全身控制主要有三类方法，本仓库的实现情况：

- [x] **方法1：加权最小二乘（Weighted Damped Least Squares）** — 闭式解，主次任务混合优化。核心实现见 `wbc_controller.py`
- [x] **方法2：零空间投影（Nullspace Projection）** — 严格分层，次任务不影响主任务。演示见 `demo_nullspace_scenarios.py`、`demo_nullspace_real.py`、`demo_arm_angle_swivel.py`
- [ ] **方法3：分层QP（Hierarchical QP）** — 需要QP求解器（如OSQP、qpOASES）叠加多层任务约束；`nullspace_example.py` 中仅作教学示意（用方法2的解析解近似），未接入真正的QP求解器

原理推导、公式证明、方法对比详见 [WBC_PRINCIPLES.md](WBC_PRINCIPLES.md)。

## 演示：标准臂角（swivel angle）零空间控制

双手末端固定为主任务，臂角 ψ（肘部绕肩-腕轴自转的相位）按正弦轨迹摆动为次任务，验证零空间投影不影响末端精度。原理见 [WBC_PRINCIPLES.md 第三章 3.7 节](WBC_PRINCIPLES.md#37-案例用标准臂角-psi-驱动零空间速度)，代码见 `demo_arm_angle_swivel.py`。

示例视频：`assets/videos/arm_angle_control.mp4`（未包含在本仓库中，运行 `demo_arm_angle_swivel.py` 可自行生成）。

## 快速开始

```bash
# 安装依赖
pip install mujoco numpy

# 首次运行会自动转换URDF为MJCF
python3 demo_nullspace_scenarios.py 1

# 标准臂角零空间控制演示
python3 demo_arm_angle_swivel.py
```

更多运行示例（远处目标、动态模式、双臂够地面等）见 [WBC_PRINCIPLES.md 1.6 节](WBC_PRINCIPLES.md#16-实现与验证)。

## 文件说明

| 文件 | 说明 |
|---|---|
| `wbc_controller.py` | WBC 核心：加权最小二乘求解器、雅可比、有限差分校验 |
| `robot_model.py` | 加载/编译 MuJoCo 模型，关节名常量 |
| `convert_urdf.py` | URDF → MJCF 转换（自动调谐伺服增益） |
| `demo_nullspace_scenarios.py` | 零空间投影演示：双手固定 + 肘部/底盘协同运动 |
| `demo_nullspace_real.py` | 零空间投影演示：末端固定 + 肘部/躯干自由运动 |
| `demo_arm_angle_swivel.py` | 标准臂角 ψ 零空间控制演示（解析雅可比 + 有限差分校验） |
| `nullspace_example.py` | 三种零空间方法的教学对比（数值示例，非真实机器人） |
| `test_dual_arm_ground.py` | 双臂够地面压力测试 |
| `simulate.py` / `visualize_nullspace.py` | 其他仿真/可视化脚本 |
