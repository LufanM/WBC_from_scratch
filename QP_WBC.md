# 运动学 QP-WBC：加权最小二乘、加权 QP（WQP）与分层 QP（HQP）

> 本文讲运动学 WBC 中按“加权”或“分层 QP”求解的三种方法：本项目实现的**加权最小二乘**（§1），以及带不等式约束的**加权 QP（WQP）**（§7）和**分层 QP（HQP）**（§8）。
> 方法分类、零空间投影的推导见 [WBC_PRINCIPLES.md](WBC_PRINCIPLES.md)，本文不再重复，只在用到时给出链接。运行示例与验证结果见 [README.md](README.md#实验与验证)。

---

## 目录

0. [方法分类与本文范围](#0-方法分类与本文范围)
1. [加权最小二乘（本项目实现）](#1-加权最小二乘本项目实现)
2. [为什么需要 QP](#2-为什么需要-qp)
3. [QP 基础](#3-qp-基础)
4. [任务与目标函数](#4-任务与目标函数)
5. [约束](#5-约束)
6. [软约束与松弛变量](#6-软约束与松弛变量)
7. [加权 QP（WQP）](#7-加权-qpwqp)
8. [分层 QP（HQP）](#8-分层-qphqp)
9. [WQP 与 HQP 的区别](#9-wqp-与-hqp-的区别)
10. [求解器：qpOASES](#10-求解器qpoases)
11. [代码结构（wbc_function/）](#11-代码结构wbc_function)
12. [调参与诊断](#12-调参与诊断)
- [附录 A 公式速查](#附录-a-公式速查)
- [附录 B 符号与单位](#附录-b-符号与单位)
- [附录 C 加权最小二乘的闭式解推导](#附录-c-加权最小二乘的闭式解推导)

---

## 0. 方法分类与本文范围

### 0.1 四种方法的位置

冗余机器人的全身控制方法可以按两个维度分类：**多任务的优先级怎么体现**、**有没有不等式约束**（详见 [WBC_PRINCIPLES.md 第一章](WBC_PRINCIPLES.md#第一章-wbc方法分类与对比)）。

| | 无不等式约束（闭式解） | 有不等式约束（QP 求解器） |
|---|---|---|
| **加权**（软优先级） | **加权最小二乘** → 本文 §1 | **加权 QP（WQP）** → 本文 §7 |
| **严格分层** | 零空间投影 → [WBC_PRINCIPLES.md 第二章](WBC_PRINCIPLES.md#第二章-零空间投影的数学推导) | **分层 QP（HQP）** → 本文 §8 |

同一行的两种方法是**同一个优化问题**，区别只在于有没有不等式约束：

- WQP 去掉不等式约束，就是 §1 的加权最小二乘；
- HQP 去掉不等式约束，就是 [WBC_PRINCIPLES.md 第二章](WBC_PRINCIPLES.md#第二章-零空间投影的数学推导)的零空间投影（§8.5）。

所以 WQP、HQP 可以看作两种闭式方法的“带约束版本”。

### 0.2 符号约定

全部文档统一使用以下符号：

| 符号 | 含义 |
|---|---|
| $\mathbf{v} = \dot{q}$ | 决策变量：受控自由度的速度（rad/s 或 m/s） |
| $J_i$、$\dot{x}_{d,i}$ | 第 $i$ 个任务的雅可比、期望任务速度 |
| $w_i$、$W_i$ | 第 $i$ 个任务的任务权重（标量）、方向权重（对角阵，不含 $w_i$），见 §4.1 |
| $A$、$S$ | 硬线性约束的矩阵、软约束的矩阵（§5、§6） |
| $\lambda$ | 阻尼（Tikhonov 正则化）系数 |
| $dt$ | 控制周期；每周期更新 $q_{\text{cmd}} \leftarrow q_{\text{cmd}} + \mathbf{v}\,dt$ |

---

## 1. 加权最小二乘（本项目实现）

> **方法类型**：加权阻尼最小二乘（Weighted Damped Least Squares）
> **特点**：闭式解、计算快、主次任务按权重折中
> **在方法分类中的位置**：“加权 × 无不等式约束”（见 [WBC_PRINCIPLES.md 1.2 节](WBC_PRINCIPLES.md#12-方法分类两个维度)）。加入不等式约束后即为加权 QP，见 [§7](#7-加权-qpwqp)

---

### 1.1 核心思想

把机器人看作一个**整体**，联合规划：
- **底盘**：3 个自由度（前后 $x$、左右 $y$、转向 yaw）
- **躯干**：4 个关节
- **手臂**：左右各 7 个关节（单臂任务时只规划一侧）

每个控制周期，计算所有自由度的**速度**，使末端尽快接近目标。

---

### 1.2 数学建模

#### 1.2.1 状态向量

全身速度向量（单臂任务 $n = 3 + 4 + 7 = 14$ 维；双臂任务 $n = 3 + 4 + 14 = 21$ 维）：

$$
\mathbf{v} = \begin{bmatrix} \dot{\mathbf{q}}_{\text{base}} \\ \dot{\mathbf{q}}_{\text{torso}} \\ \dot{\mathbf{q}}_{\text{arms}} \end{bmatrix} \in \mathbb{R}^n
$$

- $\dot{\mathbf{q}}_{\text{base}} = [\dot{x}, \dot{y}, \dot{\theta}_{\text{yaw}}]^T$：底盘在世界坐标系中的速度
- $\dot{\mathbf{q}}_{\text{torso}}$：躯干 4 个关节的角速度
- $\dot{\mathbf{q}}_{\text{arms}}$：手臂关节角速度

#### 1.2.2 末端速度与全身速度的关系

末端 6 维速度（3 维线速度 + 3 维角速度）：

$$
\dot{\mathbf{x}}_{\text{ee}} = \mathbf{J}\,\mathbf{v}
$$

$\mathbf{J}$ 是**任务雅可比**：单臂为 $6 \times 14$，双臂把两只手的雅可比上下堆叠，为 $12 \times 21$。它的第 $i$ 列表示“只有第 $i$ 个自由度以单位速度运动时，末端的速度”。

**构造方式**：MuJoCo 模型中底盘被建模为三个真实关节（`base_x`、`base_y`、`base_yaw`），所以 $\mathbf{J}$ 由 `mj_jacSite()` 一步算出，不需要手工拼接底盘项。

##### 手工推导参考：底盘部分 $\mathbf{J}_{\text{base}}$（$6 \times 3$）

刚体速度公式：

$$
\mathbf{v}_{\text{ee}} = \mathbf{v}_{\text{base}} + \boldsymbol{\omega}_{\text{base}} \times \mathbf{r}, \qquad
\boldsymbol{\omega}_{\text{ee}} = \boldsymbol{\omega}_{\text{base}}
$$

$\mathbf{r}$ 是从底盘 yaw 转轴到末端的位置向量。底盘只做平面运动：$\mathbf{v}_{\text{base}} = [v_x, v_y, 0]^T$，$\boldsymbol{\omega}_{\text{base}} = [0, 0, \omega_{\text{yaw}}]^T$，于是

$$
\boldsymbol{\omega}_{\text{base}} \times \mathbf{r} = [-r_y\,\omega_{\text{yaw}},\; r_x\,\omega_{\text{yaw}},\; 0]^T
$$

$$
\mathbf{J}_{\text{base}} = \begin{bmatrix}
1 & 0 & -r_y \\
0 & 1 & r_x \\
0 & 0 & 0 \\
0 & 0 & 0 \\
0 & 0 & 0 \\
0 & 0 & 1
\end{bmatrix}
\begin{matrix}
\leftarrow x \text{ 方向线速度} \\
\leftarrow y \text{ 方向线速度} \\
\leftarrow z \text{ 方向线速度（底盘不能升降）} \\
\leftarrow \text{roll 角速度（底盘不能侧倾）} \\
\leftarrow \text{pitch 角速度（底盘不能俯仰）} \\
\leftarrow \text{yaw 角速度}
\end{matrix}
$$

##### 实际实现：MuJoCo 自动计算

```python
# wbc_function/kinematics.py: MujocoKinematics.site_jacobian()
jacp = np.zeros((3, model.nv))   # 线速度雅可比
jacr = np.zeros((3, model.nv))   # 角速度雅可比
mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
J = np.vstack([jacp[:, dofadr], jacr[:, dofadr]])   # 只取受控自由度的列
```

MuJoCo 返回的雅可比在**世界坐标系**下表达，已包含底盘三列，数值上与手工推导等价，避免了手工推导时的坐标系错误。列的选取（`dofadr`）是最容易出错的地方，所以启动时用有限差分校验（`check_jacobian()`，结果见 [README.md](README.md#质量保证)）。

---

### 1.3 目标函数

#### 1.3.1 主任务：跟踪末端目标

末端位姿误差 $\mathbf{e}_{\text{pose}}$（6 维）：位置误差 $\mathbf{p}_d - \mathbf{p}$，姿态误差取旋转向量形式（`quat_error()`，即 $\log(R_d R^T)^\vee$，与雅可比的角速度行匹配）。

用比例控制得到期望末端速度：

$$
\dot{\mathbf{x}}_{\text{des}} = K\,\mathbf{e}_{\text{pose}}
$$

$K$ 的单位是 1/s，含义是**误差收敛的时间常数** $\tau = 1/K$：理想情况下误差按 $e(t) = e_0\, e^{-Kt}$ 衰减。本项目 $K = 12$，$\tau \approx 83$ ms。（$K$ 即 §4.2 通用写法中的 $K_p$。）

> **注意**：目标在运动时，只靠比例项会有稳态滞后 $v_{\text{target}}/K$，需要加前馈 $\dot{\mathbf{x}}_{\text{des}} = \dot{\mathbf{x}}_{\text{ff}} + K\mathbf{e}$，详见 [§4.2](#42-期望任务速度前馈--反馈)。本项目的目标都是静止点，所以没有前馈项。

我们希望 $\mathbf{J}\mathbf{v} \approx \dot{\mathbf{x}}_{\text{des}}$。但这是**欠定系统**（方程数 6 < 变量数 14），有无穷多解，需要额外的准则来挑一个。

#### 1.3.2 正则项与阻尼

**分组正则化** $\frac12\mathbf{v}^T R\,\mathbf{v}$：$R$ 是对角阵，给每个自由度定一个“动起来的代价”，决定“谁来动”。本项目 $R$ 中底盘小、关节大，所以目标远时底盘乐于移动，目标近时主要靠手臂。

> **另一种正则方式**：$\frac12\|\mathbf{v} - \mathbf{v}_{\text{posture}}\|^2_{\mathbf{W}_p}$，其中姿态速度逐自由度为
>
> $$
> v_{\text{posture},i} = \underbrace{k\,(q_{\text{mid},i} - q_i)}_{\text{拉向中点}} + \underbrace{k_l\left[\max\!\Big(0,\ 1 - \frac{q_i - q_{\min,i}}{m_i}\Big) - \max\!\Big(0,\ 1 - \frac{q_{\max,i} - q_i}{m_i}\Big)\right]}_{\text{限位斥力}}, \qquad m_i = 0.1\,(q_{\max,i} - q_{\min,i})
> $$
>
> 限位斥力只在离限位不到 $m_i$（10% 行程）时非零：离下限的距离 $q_i - q_{\min,i}$ 从 $m_i$ 减到 0，第一项从 0 线性增到 1，产生正速度（往上推）；靠近上限时第二项同理产生负速度（往下推）；两头都不靠近时为 0。$k$、$k_l$ 对应代码的 `posture_k`、`limit_w`；底盘的 $k$ 取 0（不拉向中点，只受斥力）。
>
> 它除了起分组正则化的作用（$\mathbf{W}_p$ 相当于 $R$），还多了**姿态拉回**：在冗余方向上把关节拉向中位、推离限位，防止构型漂移。代价是两件事共用一个 $\mathbf{W}_p$，无法分开调节。**本项目代码用的是这种方式**，求解结果见 1.4 节。

**阻尼项** $\frac{\lambda}{2}\|\mathbf{v}\|^2$：防止雅可比接近奇异时速度爆炸（原理见 [WBC_PRINCIPLES.md 2.6 节](WBC_PRINCIPLES.md#26-阻尼伪逆数值稳定性与精度的权衡)）。

#### 1.3.3 完整目标函数

$$
\mathcal{L}(\mathbf{v}) = \frac12\|\mathbf{J}\mathbf{v} - \dot{\mathbf{x}}_{\text{des}}\|^2_{\mathbf{W}} + \frac{\lambda}{2}\|\mathbf{v}\|^2 + \frac12\mathbf{v}^T R\,\mathbf{v}
$$

| 符号 | 含义 |
|---|---|
| $\mathbf{W}$ | 末端任务的方向权重，$\text{diag}(1,1,1,\,0.4,0.4,0.4)$ |
| $\lambda$ | 全局阻尼 |
| $R$ | 分组正则化，对角阵，底盘小、关节大 |

这是一个**无约束、严格凸**的二次规划，所以有闭式解。

---

### 1.4 闭式解

**① 展开目标函数**（$\|\mathbf{x}\|^2_{\mathbf{W}} = \mathbf{x}^T\mathbf{W}\mathbf{x}$；$\mathbf{W}$ 对称，两个交叉项相等，合并为一项）：

$$
\mathcal{L}(\mathbf{v}) = \frac12\mathbf{v}^T\mathbf{J}^T\mathbf{W}\mathbf{J}\mathbf{v}
\;-\; \dot{\mathbf{x}}_{\text{des}}^T\mathbf{W}\mathbf{J}\mathbf{v}
\;+\; \frac{\lambda}{2}\mathbf{v}^T\mathbf{v}
\;+\; \frac12\mathbf{v}^T R\,\mathbf{v}
\;+\; \underbrace{\frac12\dot{\mathbf{x}}_{\text{des}}^T\mathbf{W}\dot{\mathbf{x}}_{\text{des}}}_{\text{与 } \mathbf{v} \text{ 无关}}
$$

**② 逐项求导**，用两条规则（$\mathbf{M}$ 对称）：

$$
\frac{\partial}{\partial\mathbf{v}}\Big(\tfrac12\mathbf{v}^T\mathbf{M}\mathbf{v}\Big) = \mathbf{M}\mathbf{v}, \qquad
\frac{\partial}{\partial\mathbf{v}}\big(\mathbf{c}^T\mathbf{v}\big) = \mathbf{c}
$$

第二项写成 $\mathbf{c}^T\mathbf{v}$ 时 $\mathbf{c} = \mathbf{J}^T\mathbf{W}\dot{\mathbf{x}}_{\text{des}}$，于是

$$
\nabla\mathcal{L} = \mathbf{J}^T\mathbf{W}\mathbf{J}\,\mathbf{v} \;-\; \mathbf{J}^T\mathbf{W}\dot{\mathbf{x}}_{\text{des}} \;+\; \lambda\mathbf{v} \;+\; R\,\mathbf{v}
$$

**③ 令 $\nabla\mathcal{L} = 0$**，含 $\mathbf{v}$ 的项合并到左边：

$$
\boxed{\;\mathbf{A}\mathbf{v} = \mathbf{b}, \qquad
\mathbf{A} = \mathbf{J}^T\mathbf{W}\mathbf{J} + \lambda\mathbf{I} + R, \qquad
\mathbf{b} = \mathbf{J}^T\mathbf{W}\dot{\mathbf{x}}_{\text{des}}\;}
\qquad\Longrightarrow\qquad
\mathbf{v}^* = \mathbf{A}^{-1}\mathbf{b}
$$

$\mathbf{A}$ 一定可逆：$\mathbf{J}^T\mathbf{W}\mathbf{J}$ 和 $R$ 都半正定，加上 $\lambda\mathbf{I}$（$\lambda > 0$）后正定；正定也保证了这个驻点是最小值。$\mathbf{A}$ 是 $n \times n$ 对称正定矩阵，用 `np.linalg.solve(A, b)` 求解即可，不需要外部 QP 求解器。加上不等式约束，就是 §7 的 WQP。

> **用 $\mathbf{W}_p$ 正则方式时**（本项目代码），结果为
> $$
> \mathbf{A} = \mathbf{J}^T\mathbf{W}\mathbf{J} + \lambda\mathbf{I} + \mathbf{W}_p, \qquad
> \mathbf{b} = \mathbf{J}^T\mathbf{W}\dot{\mathbf{x}}_{\text{des}} + \mathbf{W}_p\mathbf{v}_{\text{posture}}
> $$
> 即 $\mathbf{A}$ 中 $R$ 换成 $\mathbf{W}_p$，$\mathbf{b}$ 多出姿态拉回项 $\mathbf{W}_p\mathbf{v}_{\text{posture}}$（逐项展开见[附录 C](#附录-c-加权最小二乘的闭式解推导)）。

---

### 1.5 关键设计与参数

#### 1.5.1 实际参数（`wbc_function/wbc_controller.py`）

| 参数 | 代码变量 | 值 | 作用 |
|---|---|---|---|
| $K$ | `Kp_pos`、`Kp_rot` | 12 | 位置、姿态的比例增益 |
| $\mathbf{W}$ | `task_w` | diag(1, 1, 1, 0.4, 0.4, 0.4) | 位置权重 1，姿态权重 0.4 |
| $\lambda$ | `lambda_reg` | $10^{-4}$ | 全局阻尼 |
| $\mathbf{W}_p$（关节） | `posture_w` | $2 \times 10^{-3}$ | 关节的分组正则化 |
| $\mathbf{W}_p$（底盘） | `base_reg_w` | $10^{-4}$ | 底盘的分组正则化 |
| $k$ | `posture_k` | $2 \times 10^{-3}$ | 中位拉回增益（与 $\mathbf{W}_p$ 独立） |
| $k_l$ | `limit_w` | 0.5 | 限位斥力强度，距限位 10% 行程内生效 |
| 关节速度上限 | `max_joint_vel` | 2.0 rad/s | 逐自由度裁剪 |
| 底盘平移速度上限 | `max_base_vel` | 0.6 m/s | $(v_x, v_y)$ 整体按比例缩放 |

#### 1.5.2 让底盘“按需”参与

代码用的是 $\mathbf{W}_p$ 正则方式（1.4 节引用块），$\mathbf{A}$ 的对角正则化部分为 $\lambda + W_{p,i}$（用 $R$ 方式时即 $\lambda + R_i$）：

| 自由度 | $\lambda + W_{p,i}$ |
|---|---|
| 底盘 | $10^{-4} + 10^{-4} = 2 \times 10^{-4}$ |
| 躯干、手臂关节 | $10^{-4} + 2 \times 10^{-3} = 2.1 \times 10^{-3}$ |

底盘运动的“代价”约为关节的 1/10，所以优化器乐于使用底盘。

**效果**：目标远时，底盘主动移动；目标近时，手臂承担主要运动，底盘只做很小的辅助。

> **注意，这是按比例分担，不是开关**：二次代价下，只要任务需要运动，底盘就会按代价比例分担一部分，不会严格为 0。“手臂够不着才动底盘”的明显先后，主要发生在手臂接近限位或奇异时。如果需要严格先后，应使用分层方法（[零空间投影](WBC_PRINCIPLES.md#第二章-零空间投影的数学推导) / [HQP](#8-分层-qphqp)）或直接冻结底盘（`base_free=False`）。
>
> 另外，真正决定比例的是 $\lambda + W_{p,i}$ 的**相对大小**。如果 $\lambda$ 比底盘的 $W_p$ 大很多，再把底盘 $W_p$ 调小也没有作用。

#### 1.5.3 位置优先，姿态次要

$\mathbf{W}$ 中位置权重为 1、姿态权重为 0.4。

**效果**：位置和姿态**同时**收敛，但当两者不能同时满足时（如接近工作空间边界），优化器优先保证位置。它不是“先位置、后姿态”的先后顺序。

位置误差单位是 m，姿态误差单位是 rad，权重同时起到了**单位换算**的作用，调参时要结合两者的数值量级考虑（[§4.2](#42-期望任务速度前馈--反馈)）。

#### 1.5.4 姿态项与限位处理的实际效果

- $\mathbf{b}$ 中姿态项的贡献为 $\mathbf{W}_p\mathbf{v}_{\text{posture}}$。关节的中位拉回量级为 $W_p \cdot k = 2 \times 10^{-3} \times 2 \times 10^{-3} = 4 \times 10^{-6}$ 乘以偏差，远小于任务项，**几乎只在冗余方向上起微弱作用**；
- 限位斥力的量级约为 $2 \times 10^{-3} \times 0.5 = 10^{-3}$，同样较弱。真正保证不越限的是运动学积分时的**裁剪**（`integrate()` 中的 `np.clip`）；
- 速度上限也是**求解后**逐自由度裁剪的，会改变速度方向，导致末端偏离期望方向。

这三点是闭式方法的固有局限：无法在求解时考虑不等式约束。改进方法见 [1.6 节](#16-已知限制)。

---

### 1.6 已知限制

1. **底盘模型简化**：当前假设底盘可横向平移（理想平面关节）。差速轮底盘不能侧向移动，真机部署时需替换为差速驱动模型（非完整约束）。
2. **关节限位与速度限位是“事后”处理的**：限位靠斥力 + 积分裁剪，速度靠求解后裁剪（§1.5.4）。改为求解时的硬约束只需要**普通 QP**（Box 约束），不需要二次锥规划，见 [§5.2](#52-关节限位box)。
3. **碰撞约束**：自碰撞检测在物理引擎中已启用，但尚未作为不等式约束加入控制器，同样可以线性化后放进 QP（[§5.3](#53-非线性约束的线性化通用方法)）。
4. **主次任务是软优先级**：姿态项会轻微影响末端任务。需要严格优先级时，见[零空间投影](WBC_PRINCIPLES.md#第二章-零空间投影的数学推导)或 [HQP](#8-分层-qphqp)。

---

## 2. 为什么需要 QP

§1 的加权最小二乘和 [WBC_PRINCIPLES.md 第二章](WBC_PRINCIPLES.md#第二章-零空间投影的数学推导)的零空间投影这两种闭式方法，都无法表达不等式约束，只能用变通办法：

| 需求 | 闭式方法的做法 | 问题 |
|---|---|---|
| 关节位置限位 | 靠近限位时加“斥力”速度；积分后再把关节角裁剪到限位内 | 斥力只是软性倾向；裁剪发生在求解之后，求解器“不知道”关节已经到头，末端误差无法被其他关节补偿 |
| 关节速度限位 | 求解后逐个关节裁剪（clip） | 逐个裁剪会**改变速度方向**，末端不再沿期望方向运动 |
| 躯干姿态、活动范围 | 无法表达 | — |
| 自碰撞 | 无法表达 | — |

QP 把这些限制直接写进优化问题，求解器在满足约束的前提下找最优解。例如某个关节到了限位，求解器会自动让其他关节多承担一些，末端仍沿正确方向运动。

---

## 3. QP 基础

### 3.1 标准形式

$$
\begin{aligned}
\min_{x} \quad & \frac{1}{2}x^T H x + g^T x \\
\text{s.t.} \quad & lb \leq x \leq ub && \text{（Box 约束：逐变量上下界）} \\
& lb_A \leq A x \leq ub_A && \text{（线性约束：变量的线性组合）}
\end{aligned}
$$

### 3.2 凸性

| $H$ | 性质 | 解 |
|---|---|---|
| 正定 $H \succ 0$ | 严格凸 | 唯一全局最优 |
| 半正定 $H \succeq 0$ | 凸 | 全局最优，但可能不唯一 |
| 不定 | 非凸 | 局部极小，一般难解 |

WBC 必须保证 $H \succ 0$，方法见 §4.4。

### 3.3 KKT 条件与 Active Set（直观版）

> 本节是简要版。从无约束 QP 出发、逐步推导 KKT 条件和 Active Set 迭代的完整讲解（含数值例子和可运行脚本），见 [QP_KKT_GUIDE.md](QP_KKT_GUIDE.md)。

凸 QP 的最优解处：
- 没碰到边界的约束（**inactive**）不起作用；
- 碰到边界的约束（**active**）产生“推力”（拉格朗日乘子），把解挡在边界上。

```
  v2 ↑
     │   ┌────────────┐
     │   │            │      ● 无约束最优点（在可行域外）
     │   │   可行域   │     /
     │   │            ●────     ← 约束最优点：落在右边界上（右边界 active）
     │   └────────────┘
     └──────────────────→ v1
```

知道哪些约束 active 后，把它们当作等式约束，问题就退化为解一个线性方程组。**Active Set 方法**就是不断猜测并修正 active 集合。实时控制中相邻周期的问题几乎一样，用上一周期的结果作为起点（**热启动**），通常一两次迭代就能收敛。

---

## 4. 任务与目标函数

> 本节内容 WQP 和 HQP 通用。在 HQP 中，§4.3 的叠加只发生在**同一层内部**。

### 4.1 统一的任务描述

所有任务都用同一种形式描述：

$$
\text{任务}_i = (J_i,\; \dot{x}_{d,i},\; w_i,\; W_i), \qquad
\text{代价} = w_i\, f_i(\mathbf{v}), \qquad
f_i(\mathbf{v}) = \frac{1}{2}\big\|J_i\mathbf{v} - \dot{x}_{d,i}\big\|^2_{W_i}
$$

两个权重分工不同，**各只乘一次**：

| 权重 | 形式 | 作用 | 代码 |
|---|---|---|---|
| 任务权重 $w_i$ | 标量 | 这个任务和其他任务比，有多重要 | `Task.weight` |
| 方向权重 $W_i$ | 对角阵 | 同一个任务内部，各分量（如位置 / 姿态）的相对重要性，见 §4.2 | `Task.w_diag` |

两者相乘就是该任务每一行的总权重 $w_i W_i$（代码中 `Task.weights() = weight * w_diag`）。

| 任务 | $J_i$ | $\dot{x}_{d,i}$ |
|---|---|---|
| 末端位姿 | 末端 site 的雅可比（6 行） | $\dot{x}_{\text{ff}} + K_p\, e$ |
| 躯干位姿 | 躯干 body 的雅可比（6 行或其子集） | $\dot{x}_{\text{ff}} + K_p\, e_{\text{torso}}$ |
| 关节姿态（posture） | 选择矩阵 $S$（每行在一个选中关节处为 1） | $K_p\, S(q_{\text{ref}} - q)$ |
| 臂角等标量指标 | $1 \times n$ 的指标雅可比（如 [WBC_PRINCIPLES.md 2.7 节](WBC_PRINCIPLES.md#27-案例用标准臂角-psi-驱动零空间速度)的 $\mathbf{J}_c$） | $\dot\psi_{\text{cmd}}$ |

求解器只需要处理一种数据结构。“躯干 / 手臂末端任务随意切换”，本质上就是启用或禁用不同的任务项。

### 4.2 期望任务速度：前馈 + 反馈

$$
\dot{x}_d = \underbrace{\dot{x}_{\text{ff}}}_{\text{前馈}} + \underbrace{K_p\, e}_{\text{反馈}}
$$

- **误差** $e$：位置为 $p_d - p$；姿态为 $\log(R_d R^T)^\vee$（旋转向量），与几何雅可比的角速度行匹配。实现见 `wbc_function/tasks.py` 的 `quat_error()`（§1.3.1）。
- **前馈** $\dot{x}_{\text{ff}}$：目标本身的速度，目标静止时为 0。
- **增益** $K_p$（单位 1/s）：误差的收敛时间常数 $\tau = 1/K_p$。离散实现下要满足 $K_p\,dt \ll 1$，否则会和底层关节控制器的滞后耦合，引起振荡。

> **为什么跟踪轨迹时要加前馈**：只靠反馈时，误差必须不为 0 才能产生运动，所以会有稳态滞后误差
> $$
> e_{\text{lag}} = \frac{v_{\text{target}}}{K_p}
> $$
> 例如目标以 0.1 m/s 运动、$K_p = 10$，滞后 1 cm。在“末端沿直线前伸”“躯干运动、末端保持不动”这类实验中，如果不加前馈，测到的误差大部分是这个滞后，而不是 WBC 能力不足。

**方向权重**：位置误差单位是 m，姿态误差单位是 rad，用对角权重平衡：

$$
W_i = \text{diag}(1, 1, 1,\; \rho, \rho, \rho)
$$

$W_i$ 只描述任务内部位置与姿态的比例，不含任务权重 $w_i$。

例如希望“1 cm 位置误差”和“0.05 rad 姿态误差”同等重要，则 $\rho \approx (0.01/0.05)^2 = 0.04$。

### 4.3 多任务叠加

$$
\sum_i w_i\, f_i(\mathbf{v}) \;\Longrightarrow\; H = \sum_i w_i J_i^T W_i J_i, \qquad g = -\sum_i w_i J_i^T W_i\, \dot{x}_{d,i}
$$

（展开 $w_i f_i$：$\tfrac12 w_i\,\mathbf{v}^T J_i^T W_i J_i \mathbf{v} - w_i\,\dot{x}_{d,i}^T W_i J_i \mathbf{v} + \text{常数}$，二次项系数给出 $H$，一次项系数给出 $g$。）

### 4.4 正则化

**全局阻尼 $\lambda I$：必需项**

$J_i^T W_i J_i$ 的秩最多为任务维度 $m_i$。只要关节数 $n > \sum m_i$（R1 Pro 单臂 14 个自由度对 6 维末端任务），$H$ **在任何姿态下都奇异**，不只在奇异位形才奇异。所以必须加 $\lambda I$，使 $H \succ 0$。它同时抑制奇异附近的关节跳变，原理见 [WBC_PRINCIPLES.md 2.6 节](WBC_PRINCIPLES.md#26-阻尼伪逆数值稳定性与精度的权衡)。

**分组正则化：控制“谁先动”**

$$
R = \text{diag}(\underbrace{r_b, \ldots}_{\text{底盘}},\; \underbrace{r_t, \ldots}_{\text{躯干}},\; \underbrace{r_a, \ldots}_{\text{手臂}}), \qquad H \mathrel{+}= R
$$

正则化越大，该组自由度越“懒”。§1 中的 $\mathbf{W}_p$ 起的就是这个作用。

**正则项符号对照**（全文统一）：

| 符号 | 名称 | 形式 | 作用 | 出现在 |
|---|---|---|---|---|
| $\lambda I$ | 全局阻尼 | 所有自由度相同 | 保证 $H \succ 0$，抑制奇异附近的跳变 | §1、§4.4、§7；HQP 最低层 |
| $R$ | 分组正则化 | 对角阵，按自由度分组 | 决定“谁先动” | §4.4、§7；HQP 最低层 |
| $\mathbf{W}_p$ | §1 的另一种正则方式 | 对角阵 | 分组正则化 + 姿态拉回（§1.3.2） | §1 |
| $\epsilon I$ | HQP 高层阻尼 | 所有自由度相同，极小 | 只保证高层 $H_k \succ 0$（§8.3） | HQP 第 1 ~ $K-1$ 层 |

$\lambda$ 和 $\epsilon$ 都是阻尼，区别只在大小和用途：$\lambda$ 是 WQP 和 HQP 最低层的正常阻尼；$\epsilon$ 是 HQP 高层为了数值上可解而加的极小阻尼，要尽量不影响本层任务。

> **注意，这是按比例分担，不是“先手臂、后躯干”的开关**：二次代价下，只要任务需要运动，各组都会按代价比例分担一部分。“手臂够不着才动底盘”这种明显的先后关系，通常出现在手臂接近限位（Box 约束 active）或接近奇异时。如果需要严格的先后，应该用 HQP 或关节冻结。

**冗余漂移与姿态任务**：只有末端任务时，冗余自由度不会被拉回任何位置，反复运动后构型会漂移。加一个**低权重**（或最低层）的关节姿态任务把构型拉回参考姿态。$\lambda I$ 让关节“少动”，姿态任务让关节“回到某处”，两者作用不同。

### 4.5 关节冻结 vs 不参与计算

让某些自由度不动，有两种做法：

| 做法 | 怎么做 | 结果 |
|---|---|---|
| **不参与计算**（去掉列） | 不把它放进受控自由度（`joint_names`），$J$、$H$ 都不含这一列 | 变量数变少；该关节保持当前位置 |
| **冻结**（Box 约束） | 保留这一列，令 $lb_i = ub_i = 0$ | 变量数不变；求解器保证 $\mathbf{v}_i = 0$ |

两者的最优解**完全相同**：对剩下的自由度，问题是一样的。

**冻结会增加求解难度吗？基本不会。**
- 上下界相等的变量，qpOASES 会识别为“固定变量”，不参与 active set 的迭代，相当于在内部把它消掉了；
- 多出来的只是 $H$ 中几行几列的矩阵运算，求解时间与直接去掉这些自由度基本相同，结果完全一致。

**怎么选**：

| 情况 | 推荐 | 原因 |
|---|---|---|
| 某自由度在这个应用里**始终**不用（如单臂任务的另一只手臂、车轮） | 不参与计算 | 最简单；变量更少 |
| 运行时要**切换**参与 / 不参与（如“躯干冻结 / 全身协同”模式切换） | 冻结 | QP 维度不变，切换时可以继续热启动（§10.2）；还能把速度上限在几百毫秒内平滑降到 0（或反过来），避免切换瞬间跳变 |
| 加权最小二乘（§1，没有约束） | 只能不参与计算 | 闭式解没有 Box 约束可用；§1 的 `base_free=False` 就是去掉底盘列 |

不推荐用超大正则化“近似冻结”：它不能严格冻结，还会让 $H$ 病态。

---

## 5. 约束

> 本节内容 WQP 和 HQP 通用。HQP 中物理约束出现在**每一层**。

### 5.1 需要哪些约束

| 约束 | 原因 | 类型 |
|---|---|---|
| 关节位置限位 | 机械限位 | Box（硬） |
| 关节速度、底盘速度限位 | 执行器能力 | Box（硬） |
| 躯干姿态范围（如 pitch） | 稳定性、视野 | 线性化（建议软） |
| 躯干 / 末端活动范围 | 安全区域 | 线性化（建议软） |
| 自碰撞距离 | 不自撞 | 线性化（建议软） |

### 5.2 关节限位（Box）

位置限位要换算成速度：下一周期 $q + \mathbf{v}\,dt$ 不能越限。与速度限位合并：

$$
lb_i = \text{clamp}\!\left(\beta\,\frac{q_{\min,i} - q_i}{dt},\; -\dot{q}_{\max,i},\; +\dot{q}_{\max,i}\right), \qquad
ub_i = \text{clamp}\!\left(\beta\,\frac{q_{\max,i} - q_i}{dt},\; -\dot{q}_{\max,i},\; +\dot{q}_{\max,i}\right)
$$

$\beta \in (0, 1]$ 为限位缓冲系数。

| 情况 | 结果 |
|---|---|
| 关节在限位内 | 等价于“位置限位 ∩ 速度限位” |
| 关节已超出限位 | $lb = ub = \mp\dot{q}_{\max}$，以最大速度退回，**永远不会出现 $lb > ub$** |
| $\beta < 1$ | 每周期最多走剩余距离的 $\beta$ 倍，越靠近限位越慢，平滑减速 |

这一条直接替代了 §1 中的“斥力 + 积分后裁剪”。

**底盘平移速度**：§1 的实现中是对 $(v_x, v_y)$ 整体按比例缩放，这是一个圆形约束 $v_x^2 + v_y^2 \leq v_{\max}^2$，不是线性约束。在 QP 中可以用外接或内接多边形近似（如 8 条线性约束），或者简单地用 Box $|v_x|, |v_y| \leq v_{\max}/\sqrt{2}$。

### 5.3 非线性约束的线性化（通用方法）

约束量 $c(q)$ 是关节角的非线性函数。下一周期一阶近似为：

$$
c(q + \mathbf{v}\,dt) \approx c(q) + J_c\,\mathbf{v}\,dt, \qquad J_c = \frac{\partial c}{\partial q}
$$

代入 $c_{\min} \leq c \leq c_{\max}$：

$$
\boxed{\;\frac{c_{\min} - c(q)}{dt} \;\leq\; J_c\,\mathbf{v} \;\leq\; \frac{c_{\max} - c(q)}{dt}\;}
$$

即 QP 中的一行：$A = J_c$，边界为“剩余余量 / $dt$”。同样可以乘缓冲系数 $\beta$ 实现平滑减速。

**$J_c$ 的计算**：位置、姿态类约束优先从 site / body 雅可比中取行（`mj_jacSite`、`mj_jacBody`）；其他标量用中心差分兜底，并像 §1 的 `check_jacobian()` 一样做一次差分校验。

**例：躯干 pitch**。欧拉角导数**不等于**角速度的某一分量。设 $J_\omega$ 为躯干在底座坐标系下的角速度雅可比，$\psi$ 为当前 yaw，对 ZYX 欧拉角有：

$$
J_{\text{pitch}} = \cos\psi\, J_{\omega,y} - \sin\psi\, J_{\omega,x}
$$

只有 yaw ≈ 0 时才能直接取 $y$ 行。也可以改为约束“躯干 $z$ 轴与竖直方向的夹角”，它与 yaw 无关。

**例：躯干位置**。$c = x_{\text{torso}}$ 或 $z_{\text{torso}}$，$J_c$ 取躯干位置雅可比的对应行。

### 5.4 量纲检查规则

**每一行约束 $a^T\mathbf{v}$ 的单位，必须和该行边界的单位一致。** 决策变量是速度，所以所有边界都是速度量纲：

| 约束 | 边界 | 单位 |
|---|---|---|
| 关节限位 | $(q_{\min} - q)/dt$ 与 $\dot{q}_{\max}$ 的交集 | rad/s |
| 姿态类线性约束 | $(\theta_{\min} - \theta)/dt$ | rad/s |
| 位置类线性约束 | $(x_{\min} - x)/dt$ | m/s |

> **最常见的错误**：漏掉 $1/dt$。边界会被缩小 $dt$ 倍（$dt = 2$ ms 时为 500 倍），约束过严，机器人几乎不动。
> 反过来，如果实现中决策变量取的是单周期增量 $\Delta q = \mathbf{v}\,dt$，则所有边界都**不**除 $dt$。两种写法都对，但必须全程统一。

---

## 6. 软约束与松弛变量

> 本节的罚函数写法用于 WQP。HQP 中推荐把软约束作为单独一层（§8.4）。

### 6.1 为什么需要

硬约束可能互相矛盾，使 QP **无解**，这一帧没有输出。

对于某条约束 $lb_S \leq S\mathbf{v} \leq ub_S$，有时我们希望求解器不要这么“死板”：既要保证一定有解，又要尽量少违反约束。做法是引入一个松弛变量 $s$，把它变成**软约束**。

（本文用 $S$ 表示软约束的矩阵，用 $A$ 表示硬线性约束的矩阵，以免混淆。代码中两者都是 `LinearConstraint`，用 `soft=True/False` 区分。）

### 6.2 做法

$$
lb_S - s \;\leq\; S\mathbf{v} \;\leq\; ub_S + s, \qquad s \geq 0
$$

$$
\text{代价} \mathrel{+}= \underbrace{w_1 \textstyle\sum s}_{\text{线性项}} + \underbrace{\frac{w_2}{2}\|s\|^2}_{\text{二次项}}
$$

- $s = 0$：约束完全满足，没有越界。
- $s > 0$：约束被违反了 $s$ 这么多。
- $s$ 可以任意大，所以这个 QP 永远有解（最坏情况是约束被违反，但至少能算出一个 $\mathbf{v}$）。

**代价函数里惩罚**：**在满足任务目标的同时，尽量少违反约束**，这比“无解直接罢工”好得多。

**为什么要加线性项**：只有二次惩罚时，它在 $s = 0$ 处斜率为 0。只要有任务往约束方向“推”，即使约束本来能满足，求解器也会让它**轻微违反**。加上线性项后，只要 $w_1$ 大于这股推力（对应的拉格朗日乘子），能满足时 $s$ 就严格为 0，这就是**精确罚函数**。

| 参数 | 作用 | 取值建议 |
|---|---|---|
| $w_1$（线性） | 可行时严格满足 | 较大，如任务权重的 $10^2 \sim 10^3$ 倍 |
| $w_2$（二次） | 保证正定；违反量平滑变化 | 适中，过大会让 $H$ 病态 |

### 6.3 写成 QP 标准形式

决策变量扩展为 $z = [\mathbf{v};\, s]$。$H$、$g$ 是任务和正则化给出的部分（§4.3、§4.4），松弛变量的罚函数加在右下块：

$$
H_z = \begin{bmatrix} H & 0 \\ 0 & w_2 I \end{bmatrix}, \qquad
g_z = \begin{bmatrix} g \\ w_1 \mathbf{1} \end{bmatrix}, \qquad
\begin{bmatrix} S & I \\ S & -I \end{bmatrix} z \;\;
\begin{matrix} \geq lb_S \\ \leq ub_S \end{matrix}
$$

第一行上界设为 $+\infty$，第二行下界设为 $-\infty$（qpOASES 中为 `qpOASES::INFTY`），Box 中令 $s \in [0, +\infty)$。上下界共用同一个 $s$，因为任一时刻最多只有一侧被违反。

### 6.4 哪些用硬约束，哪些用软约束

| 约束 | 建议 | 原因 |
|---|---|---|
| 关节位置、速度限位 | 硬（配合 §5.2 的 clamp 写法） | 安全底线；clamp 写法本身保证可行 |
| 关节冻结 | 硬 | 需要严格不动 |
| 躯干姿态、活动范围 | 软 | “期望区域”，极端情况下允许轻微越界 |
| 自碰撞 | 软，留较大余量 | 避免与其他约束冲突导致无解 |

这样设计后，**QP 在结构上总是可行**，求解失败只可能来自数值问题或迭代次数不够。

---

## 7. 加权 QP（WQP）

### 7.1 问题形式

所有任务按权重加成**一个**目标函数，再加上不等式约束：

$$
\boxed{\;\min_{\mathbf{v}} \;\; \sum_{i} w_i\, f_i(\mathbf{v}) + \frac{\lambda}{2}\|\mathbf{v}\|^2 + \frac{1}{2}\mathbf{v}^T R\, \mathbf{v}
\qquad \text{s.t.} \qquad lb \le \mathbf{v} \le ub,\quad lb_A \le A\mathbf{v} \le ub_A\;}
$$

- 目标函数：§4 的任务和正则化，与 §1 的加权最小二乘相同；
- $lb \le \mathbf{v} \le ub$：Box 约束，即关节位置 + 速度限位、冻结（§5.2、§4.5）；
- $lb_A \le A\mathbf{v} \le ub_A$：线性约束，如躯干姿态、活动范围（§5.3）。

下文把这两类约束合起来记为 $\mathcal{C}$，即 $\mathbf{v} \in \mathcal{C}$ 表示满足所有硬约束（§8 的 HQP 也用这个记号）。

这就是 WQP 的全部：**加权最小二乘 + 不等式约束**。（§6 的软约束是可选扩展，只是多一组松弛变量和罚函数，见 §7.2.2。）

**去掉不等式约束时的闭式解**：令目标函数对 $\mathbf{v}$ 的导数为零，得到

$$
\mathbf{v}^* = \Big(\sum_i w_i J_i^T W_i J_i + \lambda I + R\Big)^{-1} \sum_i w_i J_i^T W_i\, \dot{x}_{d,i}
$$

**推导**：对目标函数逐项求导（不含约束），$f_i = \frac12\|J_i\mathbf{v} - \dot{x}_{d,i}\|^2_{W_i}$。

① 展开一个任务项（$W_i$ 对称，两个交叉项相等，合并为一项）：

$$
w_i f_i = \frac{w_i}{2}(J_i\mathbf{v} - \dot{x}_{d,i})^T W_i (J_i\mathbf{v} - \dot{x}_{d,i})
= \frac{w_i}{2}\,\mathbf{v}^T J_i^T W_i J_i\,\mathbf{v} \;-\; w_i\,\dot{x}_{d,i}^T W_i J_i\,\mathbf{v} \;+\; \underbrace{\frac{w_i}{2}\dot{x}_{d,i}^T W_i \dot{x}_{d,i}}_{\text{与 } \mathbf{v} \text{ 无关}}
$$

② 用两条求导规则（$M$ 对称）：

$$
\frac{\partial}{\partial\mathbf{v}}\Big(\tfrac12\mathbf{v}^T M\mathbf{v}\Big) = M\mathbf{v}, \qquad
\frac{\partial}{\partial\mathbf{v}}\big(c^T\mathbf{v}\big) = c
$$

逐项求导（第二项中 $c^T = w_i\dot{x}_{d,i}^T W_i J_i$，所以 $c = w_i J_i^T W_i \dot{x}_{d,i}$）：

$$
\nabla L = \sum_i w_i J_i^T W_i J_i\,\mathbf{v} \;-\; \sum_i w_i J_i^T W_i\,\dot{x}_{d,i} \;+\; \lambda\mathbf{v} \;+\; R\,\mathbf{v}
$$

③ 令 $\nabla L = 0$，把含 $\mathbf{v}$ 的项合并到左边：

$$
\underbrace{\Big(\sum_i w_i J_i^T W_i J_i + \lambda I + R\Big)}_{H}\,\mathbf{v} = \underbrace{\sum_i w_i J_i^T W_i\,\dot{x}_{d,i}}_{-g}
$$

左乘 $H^{-1}$ 即得上式，也就是 QP 的无约束解 $H\mathbf{v} = -g$。$H$ 可逆，因为 $J_i^T W_i J_i$ 和 $R$ 都半正定，加上 $\lambda I$（$\lambda > 0$）后正定；$H$ 正定也保证了这个驻点是最小值而不是最大值。

**最简单的检验**：一个任务、$J = 1$、$W = 1$、$w = 1$、$R = 0$ 时，$\mathbf{v}^* = \dot{x}_d / (1 + \lambda)$。$\lambda = 0$ 时正好等于期望速度；$\lambda > 0$ 时略小，这就是阻尼让关节“少动一点”的效果。

**与 §1 加权最小二乘的关系**：§1 是这里只有末端一个任务（$w = 1$）的情形，代入上式即得 §1.4 的 $\mathbf{A}\mathbf{v} = \mathbf{b}$。§1.4 引用块中的 $\mathbf{W}_p$ 方式，相当于再加一个 $J = I$、$\dot{x}_d = \mathbf{v}_{\text{posture}}$、$W = \mathbf{W}_p$ 的姿态任务，并取 $R = 0$。

**含义**：所有任务的需求按权重叠加，解是各方的**加权平衡点**。权重决定“谁的话语权大”，不决定“谁必须被满足”。

### 7.2 写成求解器的标准形式

#### 7.2.1 基本形式

§7.1 的问题和 qpOASES 的标准形式（§3.1）结构完全相同，直接对号入座即可，变量就是 $z = \mathbf{v}$：

| 标准形式 | 填入 | 来源 |
|---|---|---|
| $H$ | $\sum_i w_i J_i^T W_i J_i + \lambda I + R$ | 目标函数的二次项（推导见 §7.1） |
| $g$ | $-\sum_i w_i J_i^T W_i\,\dot{x}_{d,i}$ | 目标函数的一次项 |
| `lb`、`ub` | 关节位置 + 速度限位；冻结自由度为 $[0, 0]$ | §5.2、§4.5 |
| `A`、`lbA`、`ubA` | 线性约束的矩阵和上下界 | §5.3 |

求解得到 $\mathbf{v}^*$，积分输出 $q_{\text{cmd}} \leftarrow q_{\text{cmd}} + \mathbf{v}^*\,dt$。

#### 7.2.2 加入软约束时

如果还有 §6 的软约束 $lb_S - s \le S\mathbf{v} \le ub_S + s$，就多出一组变量 $s$。qpOASES 只接受一个变量向量，所以把两组变量**堆叠**成 $z = [\mathbf{v};\, s]$，在 7.2.1 的基础上做三处扩展：

| 扩展 | 做法 |
|---|---|
| 目标函数 | 罚函数 $w_1\sum s + \frac{w_2}{2}\lVert s\rVert^2$ 加在右下块：$H_z = \begin{bmatrix} H & 0 \\ 0 & w_2 I \end{bmatrix}$，$g_z = \begin{bmatrix} g \\ w_1\mathbf{1} \end{bmatrix}$ |
| Box | 在 `lb`、`ub` 后面接上 $s \in [0, +\infty)$ |
| 线性约束 | 原有的硬约束行补零列变成 $[A\;\; 0]$；每条软约束拆成 $[S\;\; I]\,z \ge lb_S$ 和 $[S\;\; {-I}]\,z \le ub_S$ 两行（§6.3） |

求解后，$z^*$ 的前 $n$ 个元素是 $\mathbf{v}^*$，其余是各软约束的松弛量 $s^*$。

代码中 7.2.1 和 7.2.2 都在 `qp_builder.py` 的 `build_qp()` 里完成（WQP、HQP 共用）：没有软约束时 $z$ 就是 $\mathbf{v}$。

---

## 8. 分层 QP（HQP）

### 8.1 问题形式

**字典序（lexicographic）优化**：先把第 1 层做到最好，在不破坏第 1 层的前提下把第 2 层做到最好，以此类推：

$$
\boxed{\;\operatorname*{lex\,min}_{\mathbf{v} \in \mathcal{C}} \;\; \big(f_1(\mathbf{v}),\; f_2(\mathbf{v}),\; \ldots,\; f_K(\mathbf{v})\big)\;}
$$

### 8.2 逐层求解

引入每层的残差 $r_k = J_k\mathbf{v} - \dot{x}_{d,k}$（用 $r$ 而不是 $v$，避免与决策变量 $\mathbf{v}$ 混淆），第 $k$ 层为：

$$
\begin{aligned}
\min_{\mathbf{v}} \quad & \frac{1}{2}\big\|J_k\mathbf{v} - \dot{x}_{d,k}\big\|_{W_k}^2 + (\text{本层正则项，见 §8.3}) \\
\text{s.t.} \quad & J_j\mathbf{v} - \dot{x}_{d,j} = r_j^*, \quad j = 1, \ldots, k-1 && \text{（上层残差锁定为已求得的最优值）} \\
& \mathbf{v} \in \mathcal{C} && \text{（物理约束）}
\end{aligned}
$$

第 2 条约束是 HQP 的核心：**上层能做到的程度被“锁死”，下层只能在剩下的自由度里优化**。求解后计算 $r_k^* = J_k\mathbf{v}^* - \dot{x}_{d,k}$，传给下一层。

这里的 $W_k$ 是第 $k$ 层的总权重。一层只有一个任务时，任务权重 $w_k$ 乘不乘都不影响该层的最优解；一层内有多个任务时，把它们堆叠起来，$W_k$ 取各任务的 $w_i W_i$ 组成的块对角阵。

可行解集逐层收缩、层层嵌套：

$$
\mathcal{S}_0 = \mathcal{C} \;\supseteq\; \mathcal{S}_1 \;\supseteq\; \mathcal{S}_2 \;\supseteq\; \cdots, \qquad
\mathcal{S}_k = \operatorname*{arg\,min}_{\mathbf{v} \in \mathcal{S}_{k-1}} f_k(\mathbf{v})
$$

对应的 QP 标准形式（正则项按 §8.3 分层取）：

$$
H_k = J_k^T W_k J_k + \begin{cases} \epsilon I, & k < K \;\text{（高层）} \\ \lambda I + R, & k = K \;\text{（最低层）} \end{cases}, \qquad
g_k = -J_k^T W_k\, \dot{x}_{d,k}, \qquad
A_k = \begin{bmatrix} J_1 \\ \vdots \\ J_{k-1} \\ A_{\mathcal{C}} \end{bmatrix}
$$

其中 $J_j$ 对应行的上下界相等，都为 $\dot{x}_{d,j} + r_j^*$。

完整例子（安全盒、双手、躯干、底盘四层）见 [§8.7](#87-例子双手躯干底盘分层求解)。

### 8.3 正则项放在哪一层

- 第 1 ~ $K-1$ 层（高层）：只加极小的阻尼 $\frac{\epsilon}{2}\|\mathbf{v}\|^2$，$\epsilon$ 很小（如本层任务量级的 $10^{-6}$），只为保证 $H_k \succ 0$；
- 最低层 $K$：加正常的阻尼和分组正则化 $\frac{\lambda}{2}\|\mathbf{v}\|^2 + \frac12\mathbf{v}^T R\,\mathbf{v}$（与 WQP 相同），决定剩余冗余自由度怎么分配。

> **为什么分组正则化只放在最低层**：在高层放较大的正则化，高层任务就会为了“少动躯干”牺牲精度，违背了“高层优先”。放在最低层，它只影响上面所有任务都不关心的那部分自由度。

**ε 是什么、为什么必须很小**：$\epsilon$ 是高层的微小阻尼，目标函数中加 $\frac{\epsilon}{2}\|\mathbf{v}\|^2$。

- **为什么要加**：一层的任务维度通常远小于关节数，$J_k^T W_k J_k$ 奇异，这一层有无穷多个解。例如第 1 层只有任务 $v_1 + v_2 = 1$（$J = [1\;\;1]$），直线 $v_1 + v_2 = 1$ 上每个点都是最优的，$H = \begin{bmatrix}1&1\\1&1\end{bmatrix}$ 不可逆。加 $\epsilon I$ 后 $H \succ 0$，解唯一（挑范数最小的那个）。
- **为什么要小**：$\epsilon$ 会让本层任务稍微做不到位。上例的解为 $v_1 = v_2 = \frac{1}{2+\epsilon}$，残差 $\frac{\epsilon}{2+\epsilon}$：$\epsilon = 10^{-6}$ 时残差约 $5\times10^{-7}$，可以忽略；$\epsilon = 1$ 时残差 0.33，第 1 层有三分之一没做到。而这个残差会被**锁定**传给下层（§8.2），下层无法再改善，等于高层优先级被破坏。
- **它不替下层做决定**：传给下层的只是残差 $r_k^*$，不是本层挑出的那个 $\mathbf{v}$。下层仍可在“本层残差不变”的范围内（上例中整条直线）重新选择。冗余自由度最终怎么分配，由最低层的 $\lambda I + R$ 决定。

“本层任务量级的 $10^{-6}$”指与本层 $J_k^T W_k J_k$ 的数值大小相比，例如本层任务权重为 100 时取 $\epsilon \approx 10^{-4}$。

> **补充：高层接近奇异时只有 $\epsilon$ 不够**。上面的论证只考虑了“秩亏、需要 $H_k \succ 0$”，没有考虑“接近奇异”。设本层 $J_k$ 的某个奇异值 $\sigma$ 很小（例如手臂接近伸直），沿对应的方向，本层给出的关节速度为
>
> $$
> v_\sigma = \frac{\sigma}{\sigma^2 + d}\,\dot{x}_\sigma
> $$
>
> 其中 $d$ 是本层的阻尼。只有 $\epsilon$ 时 $d$ 极小，$v_\sigma \approx \dot{x}_\sigma / \sigma$ 非常大，只能靠速度上下界截住；而雅可比每个周期都在变，解就会在上下界之间逐周期翻转。WQP 中同一个式子的 $d = \lambda + R_i$，所以没有这个问题（阻尼的作用见 [WBC_PRINCIPLES.md 2.6 节](WBC_PRINCIPLES.md#26-阻尼伪逆数值稳定性与精度的权衡)）。
>
> $v_\sigma$ 在 $\sigma = \sqrt{d}$ 时最大，为 $\dot{x}_\sigma / (2\sqrt{d})$。要让它不超过关节速度上限，可以从分母和分子两头入手：
>
> 1. **每层都带正则项**（带阻尼的分层）：$H_k$ 中都加上 $\lambda I + \mathbf{W}_p$，$g_k$ 中都加上姿态项（含限位斥力），让高层的 $d$ 不再只是 $\epsilon$。高层多出的残差与 WQP 中正则项的影响同量级，优先级仍然成立。对应代码 `HQPSolver(regularization="all")`。
> 2. **任务误差限幅**：$\dot{x}_d = \dot{x}_{\text{ff}} + K_p \operatorname{clip}(e)$，让分子 $\dot{x}_\sigma$ 有界。对应 `WholeBodyController.max_pos_error` / `max_rot_error`。
> 3. **随 $\sigma_{\min}$ 自适应增大的阻尼**：$\lambda = \lambda_0 + \lambda_{\max}(1 - \sigma_{\min}/\sigma_0)^2$（$\sigma_{\min} < \sigma_0$ 时），只在接近奇异时加大 $d$，不影响正常跟踪的精度。有了第 1 条，它对高层也起作用。
>
> 三条要一起用。只做第 1 条不够：为了正常跟踪的精度，$\lambda$ 通常很小（如 $10^{-4}$，$2\sqrt{\lambda} = 0.02$），$\dot{x}_\sigma = 0.24$ m/s 时 $\dot{x}_\sigma / (2\sqrt{\lambda}) = 12$ rad/s，仍远超速度上限。
>
> 代码中两种写法都保留：`HQPSolver()` 默认 `regularization="last"`，即上文的严格写法；`WholeBodyController(solver="hqp")` 内部用 `regularization="all"`。

### 8.4 不等式任务与软约束

不等式约束也可以作为一层，例如把躯干范围放在第 1 层：

$$
\min_{\mathbf{v},\, s} \; \frac{1}{2}\|s\|^2 + \frac{\epsilon}{2}\|\mathbf{v}\|^2
\quad \text{s.t.} \quad
lb_S - s \leq S\mathbf{v} \leq ub_S + s,\;\; s \geq 0,\;\; \mathbf{v} \in \mathcal{C}
$$

得到最小违反量 $s^*$（能满足时为 0），之后各层都带着约束 $lb_S - s^* \leq S\mathbf{v} \leq ub_S + s^*$。与 WQP 的罚函数（§6）相比，这里**不需要调 $w_1$、$w_2$**，“能满足就严格满足”由分层结构保证。

**分层示例**（“躯干运动、双手末端保持不动”实验）：

| 层 | 内容 | 类型 |
|---|---|---|
| 0 | 关节位置、速度限位，冻结 | 硬约束，出现在每一层 |
| 1 | 躯干姿态、活动范围 | 不等式任务 |
| 2 | 双手末端位姿 | 等式任务 |
| 3 | 躯干轨迹 | 等式任务 |
| 4 | 关节姿态 + 分组正则化 | 等式任务（最低层） |

### 8.5 与零空间投影的关系

如果没有不等式约束、$\epsilon \to 0$，HQP 有闭式递推解，就是 [WBC_PRINCIPLES.md 第二章](WBC_PRINCIPLES.md#第二章-零空间投影的数学推导)的**零空间投影**推广到多层：

$$
\mathbf{v}_1 = J_1^\dagger\, \dot{x}_{d,1}, \qquad
\mathbf{v}_k = \mathbf{v}_{k-1} + \big(J_k N_{k-1}\big)^\dagger \big(\dot{x}_{d,k} - J_k\mathbf{v}_{k-1}\big), \qquad
N_{k-1} = I - \bar{J}_{k-1}^\dagger \bar{J}_{k-1}
$$

其中 $\bar{J}_{k-1} = [J_1;\, \ldots;\, J_{k-1}]$。$K = 2$ 且第 2 层任务为“$\mathbf{v}$ 接近 $\mathbf{v}_{\text{post}}$”（$J_2 = I$）时，退化为 [WBC_PRINCIPLES.md 2.5 节](WBC_PRINCIPLES.md#25-次任务投影完整流程)的 $\mathbf{v}^* = J^\dagger\dot{x}_{\text{des}} + N\mathbf{v}_{\text{post}}$。

所以 HQP 可以理解为**能处理不等式约束的零空间投影**。

### 8.6 实现注意

- 锁定约束 $J_j\mathbf{v} - \dot{x}_{d,j} = r_j^*$ 写成 $lb_A = ub_A$ 的线性约束。
- 上面各层的 $J_j$ 堆叠后，行之间常常**线性相关**（例如两个任务约束了同一方向）。退化的等式约束容易引起数值问题，可以放宽成 $|J_j\mathbf{v} - \dot{x}_{d,j} - r_j^*| \leq \delta$（$\delta$ 很小），或者先做 SVD/QR，只保留独立的行。
- 上层的松弛量 $s^*$ 也要固定，不能让下层“借用”上层的松弛来完成自己的任务。
- 每层的 $\epsilon$ 要远小于本层任务的量级，否则正则化本身会让本层偏离最优，并影响下层。

### 8.7 例子：双手、躯干、底盘分层求解

把 §8.2 的逐层求解代入一个具体问题，一步步走一遍。

#### 8.7.1 场景与分层

R1 的底盘、躯干、双臂都参与规划，受控自由度共 $n = 21$：

$$
\mathbf{v} = \dot{\mathbf{q}} = \begin{bmatrix} \mathbf{v}_B \\ \mathbf{v}_T \\ \mathbf{v}_L \\ \mathbf{v}_R \end{bmatrix} \in \mathbb{R}^{21}
\qquad (\text{底盘 } x, y, \text{yaw } 3,\ \text{躯干 } 4,\ \text{左臂 } 7,\ \text{右臂 } 7)
$$

要求：双手末端保持（或沿轨迹走到）目标位姿，这是主任务；躯干沿一条轨迹运动（如下蹲、前倾），次于双手；底盘走向目标位置，次于躯干；躯干始终不能离开安全范围。分为四层：

| 层 | 内容 | 类型 | 维度 |
|---|---|---|---|
| 1 | 躯干安全盒：躯干位置 $x$、$z$ 在范围内 | 不等式（软约束，§8.4） | 2 |
| 2 | 双手位姿 | 等式任务 | 12 |
| 3 | 躯干轨迹：位置 $x$、$z$ 和俯仰 ry | 等式任务 | 3 |
| 4 | 底盘位姿：$x$、$y$、yaw | 等式任务（最低层） | 3 |

关节位置、速度限位和冻结是硬约束 $\mathcal{C}$，每一层都要满足。

**自由度够不够**：三个等式任务共 $12 + 3 + 3 = 18$ 维，少于 21，余下 3 个自由度交给正则项。所以在没有碰到限位、安全盒时，所有任务都能完全满足。

#### 8.7.2 每个周期先准备的量

**硬约束 $\mathcal{C}$**：关节位置 + 速度限位（§5.2），冻结的关节 $lb = ub = 0$（§4.5）：

$$
lb \le \mathbf{v} \le ub
$$

**安全盒**：躯干位置 $c(\mathbf{q}) = [x_t,\ z_t]$ 要在 $[c_{\min}, c_{\max}]$ 内。按 §5.3 线性化成关节速度的约束：

$$
\underbrace{\beta\,\frac{c_{\min} - c}{dt}}_{lb_S} \;\le\; S\,\mathbf{v} \;\le\; \underbrace{\beta\,\frac{c_{\max} - c}{dt}}_{ub_S},
\qquad S = \begin{bmatrix} S_B & S_T & 0 & 0 \end{bmatrix}
$$

$S$ 是躯干雅可比的 $x$、$z$ 两行。躯干位置受底盘和躯干关节影响，与手臂无关。

**任务 2：双手**（每只手 6 行）

$$
J_2 = \begin{bmatrix} J_{L,B} & J_{L,T} & J_{L,L} & 0 \\ J_{R,B} & J_{R,T} & 0 & J_{R,R} \end{bmatrix}, \qquad
\dot{x}_{d,2} = \begin{bmatrix} \dot{x}_{\text{ff},L} + K_p\, e_L \\ \dot{x}_{\text{ff},R} + K_p\, e_R \end{bmatrix}, \qquad
W_2 = \text{diag}(W_L, W_R)
$$

- $J_{L,B}$、$J_{L,T}$、$J_{L,L}$ 分别是左手对底盘、躯干、左臂关节的雅可比；左手与右臂无关，那一块为 0。**底盘和躯干的列在两只手的行里都不为 0**：底盘或躯干一动，两只手都跟着动。
- $e_L$、$e_R$ 为位姿误差（位置 + 旋转向量，§4.2），$W_L = W_R = \text{diag}(1, 1, 1, \rho, \rho, \rho)$。双手保持不动时 $\dot{x}_{\text{ff}} = 0$。

**任务 3：躯干轨迹**

$$
J_3 = \begin{bmatrix} J_{t,B} & J_{t,T} & 0 & 0 \end{bmatrix}, \qquad
\dot{x}_{d,3} = \dot{x}_t(t) + K_p\, e_t, \qquad e_t = x_t(t) - x_{\text{torso}}
$$

轨迹 $x_t(t)$ 每个周期给出当前点，它的导数 $\dot{x}_t(t)$ 作为前馈（否则滞后 $v/K_p$，见 §4.2）。

**任务 4：底盘位姿**。底盘的 $x$、$y$、yaw 就是三个底盘关节本身，所以雅可比是选择矩阵：

$$
J_4 = \begin{bmatrix} I_3 & 0 & 0 & 0 \end{bmatrix}, \qquad
\dot{x}_{d,4} = \dot{x}_{\text{ff},B} + K_p\, e_B
$$

#### 8.7.3 第 1 层：安全盒（软约束，引入松弛变量）

为什么不直接把安全盒当硬约束放进 $\mathcal{C}$：躯干有跟踪误差，可能已经稍微出了盒子。此时线性化的约束要求躯干在一个周期内退回超出量的 $\beta$ 倍，再加上速度上限，可能根本做不到，整个 QP 就会不可行，所有任务一起失败。作为软约束，就变成“能满足就严格满足，满足不了就违反得最少”。

**引入松弛变量** $s \in \mathbb{R}^2$（每行约束一个），决策变量扩展为 $z = [\mathbf{v};\ s]$：

$$
\begin{aligned}
\min_{\mathbf{v},\, s} \quad & \frac12\|s\|^2 + \frac{\epsilon}{2}\|\mathbf{v}\|^2 \\
\text{s.t.} \quad & lb_S - s \le S\,\mathbf{v} \le ub_S + s, \qquad s \ge 0 \\
& lb \le \mathbf{v} \le ub
\end{aligned}
$$

- $s$ 是每行允许超出边界的量。目标函数让它尽量小：约束能满足时 $s^* = 0$；满足不了时，$s^*$ 是在 $\mathcal{C}$ 内能做到的最小违反量。
- 一行约束 $lb_S - s \le S\mathbf{v} \le ub_S + s$ 在交给 QP 求解器时拆成两行单边约束（§6.3）：$S\mathbf{v} + s \ge lb_S$ 和 $S\mathbf{v} - s \le ub_S$。
- 这一层没有任务，$\epsilon$ 只是为了让 $\mathbf{v}$ 部分的 $H$ 可逆。

**求解后固定 $s^*$**：之后的每一层都带着

$$
lb_S - s^* \;\le\; S\,\mathbf{v} \;\le\; ub_S + s^*
$$

作为**硬约束**，不再有松弛变量。这样下层不能为了自己的任务而多违反安全盒（§8.6），也不需要像 WQP 那样调罚函数权重（§6），“能满足就严格满足”由分层本身保证。

> 代码中软约束的罚函数是 $w_{\text{lin}}\sum s + \frac{w_{\text{quad}}}{2}\|s\|^2$（与 WQP 共用，§6）。这一层只有松弛变量时，它和 $\frac12\|s\|^2$ 的最优解相同：能满足时都是 $s^* = 0$，满足不了时都是最小违反量。

#### 8.7.4 第 2 层：双手

$$
\begin{aligned}
\min_{\mathbf{v}} \quad & \frac12\big\|J_2\mathbf{v} - \dot{x}_{d,2}\big\|^2_{W_2} + \frac{\epsilon}{2}\|\mathbf{v}\|^2 \\
\text{s.t.} \quad & lb_S - s^* \le S\,\mathbf{v} \le ub_S + s^* \\
& lb \le \mathbf{v} \le ub
\end{aligned}
$$

求得 $\mathbf{v}_2^*$，以及双手做到的程度（残差）$r_2^* = J_2\mathbf{v}_2^* - \dot{x}_{d,2}$：

- 双手目标够得到：$r_2^* \approx 0$；
- 够不到，或被限位、安全盒卡住：$r_2^* \ne 0$，它是在这些约束下能做到的最好结果。

这一层的解不唯一（还有 $21 - 12 = 9$ 维冗余），$\epsilon$ 只是从中挑一个范数最小的（§8.3）。**传给下层的只有 $r_2^*$**，底盘、躯干怎么动由下面的层重新决定。

#### 8.7.5 第 3 层：躯干轨迹

$$
\begin{aligned}
\min_{\mathbf{v}} \quad & \frac12\big\|J_3\mathbf{v} - \dot{x}_{d,3}\big\|^2_{W_3} + \frac{\epsilon}{2}\|\mathbf{v}\|^2 \\
\text{s.t.} \quad & J_2\mathbf{v} = \dot{x}_{d,2} + r_2^* && \text{（双手锁定，12 个等式）} \\
& lb_S - s^* \le S\,\mathbf{v} \le ub_S + s^* \\
& lb \le \mathbf{v} \le ub
\end{aligned}
$$

求得 $\mathbf{v}_3^*$ 和躯干残差 $r_3^* = J_3\mathbf{v}_3^* - \dot{x}_{d,3}$。$r_3^* \ne 0$ 表示躯干为了双手让步了。

#### 8.7.6 第 4 层：底盘（最低层）

$$
\begin{aligned}
\min_{\mathbf{v}} \quad & \frac12\big\|J_4\mathbf{v} - \dot{x}_{d,4}\big\|^2_{W_4} + \frac{\lambda}{2}\|\mathbf{v}\|^2 + \frac12\mathbf{v}^T R\,\mathbf{v} \\
\text{s.t.} \quad & J_2\mathbf{v} = \dot{x}_{d,2} + r_2^* && \text{（双手锁定）} \\
& J_3\mathbf{v} = \dot{x}_{d,3} + r_3^* && \text{（躯干锁定，3 个等式）} \\
& lb_S - s^* \le S\,\mathbf{v} \le ub_S + s^* \\
& lb \le \mathbf{v} \le ub
\end{aligned}
$$

- 这是最后一层，带正常的阻尼 $\lambda$ 和分组正则化 $R$（§8.3），由它决定剩下 3 个自由度怎么分配；
- 解 $\mathbf{v}_4^*$ 就是本周期的输出 $\mathbf{v}^*$；底盘残差 $r_4^* \ne 0$ 表示底盘让步了。

#### 8.7.7 四层交给 QP 求解器的标准形式

| | 第 1 层（安全盒） | 第 2 层（双手） | 第 3 层（躯干） | 第 4 层（底盘） |
|---|---|---|---|---|
| 变量 | $z = [\mathbf{v};\ s]$ | $\mathbf{v}$ | $\mathbf{v}$ | $\mathbf{v}$ |
| $H$ | $\text{diag}(\epsilon I,\ I)$ | $J_2^T W_2 J_2 + \epsilon I$ | $J_3^T W_3 J_3 + \epsilon I$ | $J_4^T W_4 J_4 + \lambda I + R$ |
| $g$ | $0$ | $-J_2^T W_2\,\dot{x}_{d,2}$ | $-J_3^T W_3\,\dot{x}_{d,3}$ | $-J_4^T W_4\,\dot{x}_{d,4}$ |
| Box | $lb \le \mathbf{v} \le ub$，$s \ge 0$ | $lb \le \mathbf{v} \le ub$ | 同左 | 同左 |
| 线性约束 | $[S\ \ I]\,z \ge lb_S$，$[S\ \ {-I}]\,z \le ub_S$ | 安全盒（固定 $s^*$） | 安全盒 + 双手锁定 | 安全盒 + 双手锁定 + 躯干锁定 |

锁定行的上下界相等：$J_j\mathbf{v} = \dot{x}_{d,j} + r_j^* = J_j\mathbf{v}_j^*$。层数越往下，线性约束越多，可行域越小（§8.2 的 $\mathcal{S}_0 \supseteq \mathcal{S}_1 \supseteq \cdots$）。

#### 8.7.8 为什么底盘、躯干动时双手不动

把锁定的等式拆开看（双手、躯干都要保持不动、误差为 0 时，右端为 0）。

躯干锁定（第 4 层的约束）：

$$
J_{t,B}\,\mathbf{v}_B + J_{t,T}\,\mathbf{v}_T = 0
\quad\Longrightarrow\quad
\mathbf{v}_T = -J_{t,T}^{\dagger}\, J_{t,B}\,\mathbf{v}_B \;(+\ \text{躯干自身零空间中的运动})
$$

左手锁定（第 3、4 层的约束）：

$$
J_{L,B}\,\mathbf{v}_B + J_{L,T}\,\mathbf{v}_T + J_{L,L}\,\mathbf{v}_L = 0
\quad\Longrightarrow\quad
\mathbf{v}_L = -J_{L,L}^{\dagger}\big(J_{L,B}\,\mathbf{v}_B + J_{L,T}\,\mathbf{v}_T\big) \;(+\ \cdots)
$$

所以底盘前进一点，躯干关节先反向补偿，让躯干在世界坐标系中不动；手臂再补偿底盘和躯干两者的运动，让手不动。这一串补偿不需要单独写，是“锁定上层的 $J_j\mathbf{v}$”自动带来的。

#### 8.7.9 几种情况

| 情况 | 结果 |
|---|---|
| 都能做到 | $s^* = 0$，$r_2^* = r_3^* = r_4^* = 0$：双手、躯干按要求，底盘走到目标 |
| 底盘目标太远，躯干、手臂补偿不了 | $r_2^* = r_3^* = 0$，$r_4^* \ne 0$：双手、躯干保持，**底盘停在能到的地方** |
| 躯干轨迹走得太远，手臂补偿不了 | $r_2^* = 0$，$r_3^* \ne 0$：双手保持，**躯干让步**；底盘只在剩余自由度里动 |
| 躯干轨迹要出安全盒 | 躯干停在盒子边界（安全盒在第 1 层，躯干任务在第 3 层） |
| 躯干已经出了安全盒 | $s^* > 0$：本周期以最小的违反量往回走，下层都不能让违反变大 |
| 双手目标本身够不到 | $r_2^* \ne 0$（已是最好）；躯干、底盘不会为了自己的目标把手拉得更远 |
| 某关节到限位 | 每层都满足 $\mathcal{C}$，限位从不被突破，由其他关节补偿 |

“底盘让步”“躯干让步”正是 HQP 和 WQP 的区别（§9）：WQP 把所有任务放在同一个目标函数里，底盘、躯干的目标会按权重比把双手拉离目标；HQP 中下层根本改变不了上层的 $J_j\mathbf{v}$。

#### 8.7.10 没有不等式约束时：闭式解

去掉 $\mathcal{C}$ 和安全盒、$\epsilon \to 0$ 时，三个等式任务就是 §8.5 的零空间投影（$J^\dagger$ 为伪逆）：

$$
\begin{aligned}
\mathbf{v}_2 &= J_2^\dagger\,\dot{x}_{d,2} \\
\mathbf{v}_3 &= \mathbf{v}_2 + (J_3 N_2)^\dagger\big(\dot{x}_{d,3} - J_3\mathbf{v}_2\big), & N_2 &= I - J_2^\dagger J_2 \\
\mathbf{v}^* &= \mathbf{v}_3 + (J_4 N_{23})^\dagger\big(\dot{x}_{d,4} - J_4\mathbf{v}_3\big), & N_{23} &= I - \bar{J}^\dagger \bar{J},\quad \bar{J} = \begin{bmatrix} J_2 \\ J_3 \end{bmatrix}
\end{aligned}
$$

- $N_2$ 把速度投影到“不影响双手”的方向，$N_{23}$ 投影到“既不影响双手、也不影响躯干”的方向；
- 每一项都只用不影响上层的方向，去补本层还差的部分（$\dot{x}_{d,k} - J_k\mathbf{v}_{k-1}$）。

HQP 做的事情完全一样，只是用 QP 代替伪逆，从而能同时满足限位、安全盒等不等式约束。

#### 8.7.11 对应代码

层号用 `priority` 表示，数字越小越优先：本节的第 1~4 层依次为 `priority = -1, 0, 1, 2`。

```python
wbc = wf.WholeBodyController(
    model, data, BASE_JOINTS + TORSO_JOINTS + LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS,
    ["left_ee", "right_ee"],                       # 双手：默认 priority=0（第 2 层）
    solver="hqp", dt=0.01,
    frames={"torso": {"body": "torso_link4", "components": ("x", "z", "ry"),
                      "priority": 1},               # 躯干：第 3 层
            "base":  {"body": "base_link", "components": ("x", "y", "rz"),
                      "priority": 2}})              # 底盘：第 4 层

def torso_box(c):                                   # 安全盒：第 1 层（软约束）
    p, _ = c.get_frame_pose("torso")
    S = c.frame_jacobian("torso")[[0, 2]]           # x、z 两行
    return wf.linearized_constraint("torso_box", S, p[[0, 2]], [x_min, z_min], [x_max, z_max],
                                    c.dt, beta=0.5, soft=True, priority=-1)
wbc.add_constraint("torso_box", torso_box)

# 每个控制周期
wbc.task("left_ee").set_target(p_L, q_L)            # 双手保持
wbc.task("right_ee").set_target(p_R, q_R)
wbc.task("torso").set_target(p_t, q_t, vel=v_t)     # 躯干轨迹当前点 + 前馈速度
wbc.task("base").set_target(p_b, q_b)               # 底盘目标（z 不跟踪，给任意值）
for name in ("left_ee", "right_ee", "torso", "base"):
    wbc.task(name).enabled = True
qd, err = wbc.step()

res = wbc.last_result
res.levels                # 每层的 LevelInfo：priority、tasks、status、residual（即 |r_k^*|）
res.slack["torso_box"]    # 第 1 层求得的 s*
```

实际控制时还要加上 §8.3 补充中的三条（`WholeBodyController(solver="hqp")` 已默认每层带正则项）。完整的交互演示（含模式切换、轨迹插值）见 `demo_hqp_interactive.py`。

---

## 9. WQP 与 HQP 的区别

### 9.1 一个例子：两个冲突的任务

1 个自由度，2 个任务，要求互相冲突：

$$
\text{任务 1：} v = a, \qquad \text{任务 2：} v = b, \qquad a \neq b
$$

**WQP**（忽略正则化）：

$$
\min_{v}\; \frac{w_1}{2}(v - a)^2 + \frac{w_2}{2}(v - b)^2
\;\Longrightarrow\;
v^* = \frac{w_1 a + w_2 b}{w_1 + w_2}, \qquad
|v^* - a| = \frac{w_2}{w_1 + w_2}\,|b - a| \neq 0
$$

**HQP**：

$$
\text{第 1 层：}\; v^* = a, \qquad
\text{第 2 层：}\; v \text{ 已被锁定为 } a \;\Rightarrow\; \text{任务 2 残差} = |b - a|
$$

- WQP 中，高优先级任务的残差与权重比 $w_2/w_1$ 成正比，只有 $w_1/w_2 \to \infty$ 时才趋于 HQP；
- 但权重比越大，$H$ 的条件数越差，所以**不能靠把权重拉得极大来模拟严格优先级**。

### 9.2 数学性质对比

| | WQP | HQP |
|---|---|---|
| 优化问题 | $\min \sum_i w_i f_i$ | $\operatorname{lex\,min}\,(f_1, f_2, \ldots)$ |
| 最优性含义 | 加权和最优：Pareto 前沿上由权重决定的一点 | 字典序最优 |
| 冲突时高优先级残差 | $\neq 0$，约正比于 $w_{\text{低}}/w_{\text{高}}$ | 等于该任务单独求解时能达到的最小值 |
| 两者关系 | 权重比 $\to \infty$ 时趋于 HQP | — |
| 无不等式约束时 | 退化为加权最小二乘（§1） | 退化为零空间投影（[WBC_PRINCIPLES.md 第二章](WBC_PRINCIPLES.md#第二章-零空间投影的数学推导)） |
| 数值条件 | 权重比越大越病态 | 每层独立，条件数只取决于本层 |
| 求解规模 | 1 个 QP | $K$ 个 QP，约束逐层增加 |
| 计算量 | 低 | 约为 WQP 的 $K$ 倍 |
| 任务切换 | 权重连续变化，可平滑过渡 | 层级离散，换层时可能跳变 |
| 调参对象 | 任务权重 | 层级顺序（及层内权重） |

### 9.3 怎么选

以“躯干运动、双手末端保持不动”为例：

| | WQP | HQP |
|---|---|---|
| 设置 | $w_{\text{末端}} \gg w_{\text{躯干}} \gg w_{\text{姿态}}$ | 按 §8.4 的分层表 |
| 末端误差 | 很小但不为 0，与权重比有关 | 只要手臂够得着就为 0 |
| 躯干跟踪 | 与末端折中 | 只在不影响末端的前提下跟踪 |
| 手臂到达极限时 | 末端和躯干一起“妥协” | 躯干让步，末端优先保持 |

**建议**：
1. 先实现 WQP。它在现有加权最小二乘的基础上只是多了约束，改动最小；
2. 如果高优先级任务的残差达不到要求，或者需要“某任务必须保证”的明确语义，再升级为 HQP；
3. 实际中常用**混合形式**：层数不多（2~3 层），层内用权重组合多个任务；
4. 两者共用同一套任务和约束接口（§11）。

---

## 10. 求解器：qpOASES

### 10.1 选类

| 类 | 适用 |
|---|---|
| `QProblemB` | 只有 Box 约束 |
| `QProblem` | 有线性约束，热启动之间 $H$、$A$ 不变 |
| `SQProblem` | 有线性约束，**$H$、$A$ 每次都可以变** |

WBC 中 $J$ 随姿态变化，**用 `SQProblem`**。本仓库的 Python 实现见 `wbc_function/qpoases_interface.py` 与 `qpoases_bridge.cpp`（§11）：通过 ctypes 调用 qpOASES 的 C++ 库。也可以改用 `proxsuite`、`osqp` 等有现成 Python 接口的求解器，思路相同。

### 10.2 关键细节

| 要点 | 说明 |
|---|---|
| 维度固定 | 变量数和约束数在构造时确定。启用 / 禁用**任务**不改变维度（只累加进 $H$、$g$）；启用 / 禁用**约束**会改变约束行数和松弛变量数。推荐预留固定行数：未启用的行设为 $A = 0$、$lb_A = -\infty$、$ub_A = +\infty$，对应松弛的边界设为 $[0, 0]$，这样可以继续热启动 |
| 行优先存储 | $H$、$A$ 要求行优先；Eigen 默认列优先，可用 `RowMajor` 矩阵类型 |
| `nWSR` 传引用 | 输入为迭代上限，返回实际迭代次数，每次调用前要重新赋值 |
| 冷 / 热启动 | 第一次 `init`，之后 `hotstart`；热启动失败退回 `init` |
| 选项 | `setToMPC()` 速度优先，`setToReliable()` 稳健优先；先用 Reliable 调通 |
| 返回值 | `getSimpleStatus(ret)`：0 成功、1 迭代超限、-1 内部错误、-2 不可行、-3 无界 |
| 对偶解 | 长度为“变量数 + 约束数”；正值表示下界 active，负值表示上界 active。可用于查看是什么约束在“卡”着 |

### 10.3 HQP 补充

| 要点 | 说明 |
|---|---|
| 每层一个求解器对象 | 各层约束行数不同，每层各建一个 `SQProblem`，各自热启动 |
| 计算时间 | 约为 WQP 的层数倍；层数一般控制在 3~5 层 |
| 任一层失败 | 最高层失败：本周期整体按失败处理（$\mathbf{v} = 0$）。低层失败：输出上一层的解并标记降级（`result.degraded`），它满足所有硬约束和更高层的任务，只是更低层没有优化 |
| 锁定行退化 | 上层锁定的行先做 SVD，只保留线性无关的单位行，锁定值取上一层的解（§8.6） |
| 剩余自由度为 0 | 上层锁定已确定全部自由变量时，跳过下面的层（结果不会再变） |

### 10.4 求解失败的处理

```
求解失败：
  1. 本周期 v = 0，即保持上一周期指令 q_cmd，不要下发未定义的值
  2. 记录日志：状态码、nWSR、当前 q、各约束边界
  3. 下个周期用 init 冷启动
```

采用 §5.2 的 clamp 写法和 §6 的软约束后，“不可行”在结构上已被消除，失败基本只剩迭代超限，适当增大 nWSR 即可。

---

## 11. 代码结构（`wbc_function/`）

本节内容都已在 [`wbc_function/`](wbc_function/__init__.py) 包中实现，测试见 `test_wbc_function.py`。包含 WLS（§1）、WQP（§7）和 HQP（§8），三者共用同一套任务和约束接口。

### 11.1 模块与原理的对应

| 原理 | `wbc_function/` 中的实现 | 说明 |
|---|---|---|
| 任务 $(J_i, \dot{x}_{d,i}, w_i, W_i)$（§4.1） | `problem.Task(name, J, xdot_des, weight, w_diag, priority)` | `weight` 即 $w_i$，`w_diag` 即 $W_i$ 的对角线；代价 $\frac{w}{2}\lVert J\mathbf{v} - \dot{x}_d\rVert^2_{\text{diag}(w\_diag)}$；`priority` 为 HQP 的层（数字越小越优先），WLS / WQP 忽略 |
| 末端位姿任务、前馈（§4.2） | `tasks.pose_task`、`tasks.position_task`、`kinematics.site_pose_task` | $\dot{x}_d = \dot{x}_{\text{ff}} + K_p e$，姿态误差为旋转向量 |
| 关节姿态任务（§4.4） | `tasks.joint_posture_task` | 选择矩阵 $S$ |
| 标量指标任务（臂角等） | `tasks.scalar_task` | $1 \times n$ |
| 全局阻尼 $\lambda$、§1 的姿态项 $\frac12\lVert\mathbf{v} - \mathbf{v}_{\text{posture}}\rVert^2_{\mathbf{W}_p}$ | `problem.WBCProblem(n, lambda_reg, wp_diag, v_posture)` | `wp_diag` 即 $\mathbf{W}_p$ 的对角；`v_posture` 不给（为 0）时即分组正则化 $R$ |
| 关节限位 Box（§5.2） | `tasks.joint_velocity_bounds(q, q_min, q_max, qd_max, dt, beta, frozen)` + `prob.set_bounds` | clamp 写法，永不出现 $lb > ub$ |
| 关节冻结（§4.5） | `prob.freeze(mask)` 或 `joint_velocity_bounds(..., frozen=mask)` | $lb = ub = 0$ |
| 线性化约束（§5.3） | `tasks.linearized_constraint(name, Jc, c, c_min, c_max, dt, beta, soft)` | 自动除以 $dt$ |
| 软约束（§6） | `problem.LinearConstraint(..., soft=True, w_lin, w_quad, priority)` | 线性 + 二次罚函数；HQP 中作为第 `priority` 层（§8.4） |
| 目标函数 $H$、$g$ | `WBCProblem.objective()` = `task_objective()` + `regularization()` | WLS 与 WQP 用全部；HQP 按层取任务部分 |
| 加权最小二乘（§1） | `wls_solver.WLSSolver(saturation="clip" / "scale" / None)` | 闭式解；Box 只能事后处理 |
| 加权 QP（§7） | `wqp_solver.WQPSolver(preset, max_iter, max_cputime)` | qpOASES `SQProblem`，自动热启动 |
| 分层 QP（§8） | `hqp_solver.HQPSolver(preset, regularization="last" / "all", eps, lock_tol, rank_tol)` | 每层一个 qpOASES 对象（按层的 priority 缓存热启动）；`result.levels` 为每层的 `LevelInfo`（状态、残差 $\lvert r_k^*\rvert$、锁定行数），`result.degraded` 表示低层失败 |
| QP 组装（§6.3、§8.2） | `qp_builder.build_qp`、`read_back`、`QPCache` | WQP 和 HQP 共用：$z = [\mathbf{v}; s]$ 的组装、active 标志与松弛量回读、qpOASES 对象缓存 |
| qpOASES 封装（§10） | `qpoases_interface.QpOasesSQP`、`qpoases_bridge.cpp` | `qpoases_bridge.cpp` 把 C++ 类包装成 C 函数，ctypes 调用；首次使用时自动编译 |
| 全身控制器（应用层） | `wbc_controller.WholeBodyController`：`add_frame_task` / `add_task` / `add_constraint` / `task(name)` / `set_frozen` / `step` / `compute` | 常驻任务对象（目标、前馈、开关、权重、`priority`）+ 常驻约束 + 冻结；`solver="wls"` 即 §1 的原实现，`"wqp"` 加入关节限位硬约束，`"hqp"` 再按 `priority` 分层（§11.4） |
| 运动学接口 | `kinematics.MujocoKinematics(model, data, joint_names)` | 固定自由度顺序，给出 $q$、限位、site / body 雅可比与位姿 |
| 输出积分 | `tasks.integrate(q, v, dt, q_min, q_max)` | $q \leftarrow q + \mathbf{v}\,dt$ |

**原则**：
- 任务和约束只描述自己，不关心怎么求解；WLS、WQP、HQP 共用同一个 `WBCProblem`，换求解器只需换一行；
- 受控自由度的顺序只在 `MujocoKinematics(joint_names)` 中定义一次，所有 $J$、$A$ 的列都按这个顺序；
- 所有边界和约束都是**速度量纲**（§5.4），`joint_velocity_bounds` 和 `linearized_constraint` 已经替你除以 $dt$。

### 11.2 每周期流程

```python
import wbc_function as wf

kin = wf.MujocoKinematics(model, data, joint_names)
solver = wf.WQPSolver()                                # 或 wf.WLSSolver(saturation="scale")
q_cmd = kin.q()

while running:
    kin.set_q(q_cmd)                                   # 在指令值上算 FK 与雅可比

    prob = wf.WBCProblem(kin.n, lambda_reg=1e-4, wp_diag=wp)
    prob.add_task(wf.site_pose_task(kin, "right_ee", target_pos, target_quat,
                                    kp_pos=12.0, kp_rot=12.0, xdot_ff=target_vel))
    prob.add_task(wf.joint_posture_task(q_cmd, q_ref, kp=1.0, weight=1e-3))
    prob.set_bounds(*wf.joint_velocity_bounds(q_cmd, kin.q_min, kin.q_max,
                                              qd_max, dt, beta=0.5,
                                              frozen=frozen_mask))
    prob.add_constraint(wf.linearized_constraint(
        "torso_pitch", J_pitch, pitch, pitch_min, pitch_max, dt, soft=True))

    res = solver.solve(prob)
    if res.ok:
        q_cmd = wf.integrate(q_cmd, res.v, dt)        # 失败时 res.v = 0，保持上一指令
    send(q_cmd)
    log(res)                                           # 见 11.5
```

> **为什么在 $q_{\text{cmd}}$ 上算**：运动学 WBC 一般用指令值积分，不受编码器噪声和底层跟踪滞后影响。§1 的实现在运动学模式下也是这么做的（`integrate()`）。同时监控 $\|q_{\text{cmd}} - q_{\text{meas}}\|$，偏差持续变大时（碰撞、堵转）重置为测量值或停止。

### 11.3 求解结果与使用注意

`solve()` 返回 `WBCResult`：

| 字段 | 含义 |
|---|---|
| `v` | 最优速度；失败时为 0 |
| `ok`、`status` | 是否成功；`solved` / `max_iterations` / `infeasible` / `unbounded` / `error` |
| `solve_time`、`n_iter`、`cold_start` | 求解耗时（s）、qpOASES 迭代次数、本次是否冷启动 |
| `task_residuals` | 每个任务的残差 $\lVert J_i\mathbf{v} - \dot{x}_{d,i}\rVert$ |
| `slack` | 每个软约束的松弛量 $s$（WQP） |
| `active_bounds`、`active_constraints` | 每个自由度 / 约束行：-1 下界 active，+1 上界 active，0 inactive（WQP 取自对偶解） |

**注意**：
- **保持问题形状固定**：`WQPSolver` 只在变量数、约束行数或软约束行数变化时重建 qpOASES 对象并冷启动（§10.2）。需要临时关掉某个约束时，把它的边界设为 $\pm\infty$，而不是删掉它；任务的增删不影响形状。
- **qpOASES 路径**：第一次使用 `WQPSolver` 时，会用 g++ 把 `qpoases_bridge.cpp` 编译到 `wbc_function/_build/`。qpOASES 不在 conda 默认位置时，设置 `QPOASES_ROOT`。

### 11.4 全身控制器 `WholeBodyController`

`wbc_controller.WholeBodyController` 是用上面这些组件拼出来的应用层。**任务是常驻对象**：注册一次，运行时随时修改目标、前馈速度、开关、权重；每个控制周期调用一次 `step()`。控制器只如实求解当前给定的目标，**不做任何平滑**——目标轨迹由上层规划保证连续。

**注册任务**：

```python
from robot_model import R1_FRAMES, TORSO_JOINTS   # torso: torso_link4 全 6 分量；base: base_link 只跟踪 x, y, rz
wbc = wf.WholeBodyController(model, data, joints, ["left_ee", "right_ee"],
                             solver="wqp", frames=R1_FRAMES)
wbc.add_frame_task("head", body="zed_link", components=("rx", "ry"))   # 也可以之后再加
wbc.add_task("arm_angle", lambda c: wf.TaskOutput(J=Jc, xdot=[psi_dot_cmd],
                                                  error=psi_des - psi))  # 自定义任务（error 可选）
```

- 末端（`ee_sites`）自动注册为同名的 6 分量坐标系任务；`frames` 和 `add_frame_task` 注册其他 site / body；
- `components` 选择跟踪的分量（`x y z rx ry rz`，rx/ry/rz 为绕**世界坐标系** x/y/z 轴的转动：底盘只绕 z 转，rz 就是 yaw；躯干的 ry 只在底盘 yaw ≈ 0 时才等于俯仰角）；
- 坐标系任务注册后**默认关闭**，自定义任务默认开启。

**每个周期**：

```python
hand, torso = wbc.task("right_ee"), wbc.task("torso")
hand.set_target(p, q, vel=v_ff)     # 目标 + 前馈速度 [vx,vy,vz,wx,wy,wz]（运动目标必须给，否则滞后 v/Kp）
hand.enabled = True
torso.enabled = False               # 切换任务：改开关即可，QP 维度不变，热启动不受影响
torso.weight = 2.0                  # 权重也可以随时改
wbc.set_frozen(TORSO_JOINTS)        # 冻结一组关节；set_frozen([]) 全部解冻

qd, err = wbc.step()                # err["right_ee"] = (位置误差, 姿态误差)；自定义任务为它返回的 error
q = wbc.integrate(qd, dt)
```

**快捷写法** `compute(targets)`：设置这些坐标系任务的目标并只开启它们（其余坐标系任务关闭，自定义任务不受影响），然后 `step()`。`targets` 为 `{名字: (位置, 四元数)}` 或 `{名字: (位置, 四元数, 前馈速度)}`：

```python
qd, err = wbc.compute({"right_ee": (p_hand, q_hand),
                       "base": ([x, y, 0.0], wf.rpy_to_quat([0, 0, yaw]))})   # 底盘不跟踪 z，给任意值
```

**分层（`solver="hqp"`）**：任务的 `priority` 决定所在层，数字越小越优先，同层内仍按权重加权；可在注册时给（`add_frame_task(..., priority=1)`、`frames` 里的 `"priority"`），也可随时改 `task.priority`：

```python
wbc = wf.WholeBodyController(model, data, joints, ["left_ee", "right_ee"], solver="hqp",
                             frames={"torso": {"body": "torso_link4", "priority": 1}}, dt=0.01)
wbc.max_pos_error, wbc.max_rot_error = 0.02, 0.1      # 误差限幅（§8.3 补充）
```

双手在第 0 层，躯干在第 1 层：躯干目标够不到时，躯干让步，双手不受影响。

**常驻约束** `add_constraint(name, fn)`：每个周期调用 `fn(controller)`，返回本周期的 `LinearConstraint`（或 `None`）。例如躯干位置安全盒，作为 HQP 的第 -1 层（软约束：先最小化违反量，之后各层都满足）：

```python
def torso_box(c):
    p, _ = c.get_frame_pose("torso")
    J = c.frame_jacobian("torso")[[0, 2]]                 # x、z 两行
    return wf.linearized_constraint("torso_box", J, p[[0, 2]], [x_min, z_min], [x_max, z_max],
                                    c.dt, beta=0.5, soft=True, priority=-1)
wbc.add_constraint("torso_box", torso_box)
```

**冻结**：WQP / HQP 中令被冻结自由度 $lb = ub = 0$；WLS 中求解器把上下界相等的变量固定下来，只对其余变量求解（消元），结果与 WQP 一致。都不改变问题维度，可以随时切换（§4.5）。

**能不能控制**：每个跟踪分量至少要有一个**未冻结的**受控自由度能影响它（雅可比该行不全为 0）。全部分量都控制不了——例如底盘不参与（`base_free=False`）或被冻结时还给了底盘目标——会**报错**；部分分量控制不了——例如底盘不动时躯干的 y（侧向平移）——会**警告一次**，其余分量照常跟踪。只关心部分分量时用 `components` 写明（如躯干只要高度和俯仰：`("z", "ry")`）。

**其他**：
- `solver="wls"` 时与项目最初的实现逐步一致（回归测试最大差 1e-12）；`solver="wqp"` 时再加上关节位置、速度限位的硬约束；`solver="hqp"` 时约束相同，任务按 `priority` 分层，用 `HQPSolver(regularization="all")`；
- `dt`：控制周期，WQP / HQP 由关节位置限位换算速度上下界时使用；默认取模型仿真步长，控制周期更长时要显式给出；
- 姿态项 $\frac12\lVert\mathbf{v} - \mathbf{v}_{\text{posture}}\rVert^2_{\mathbf{W}_p}$ 每个周期都通过 `WBCProblem(wp_diag=..., v_posture=...)` 加入，不算任务，不出现在任务残差中；
- 需要额外的约束时，用 `add_constraint` 注册常驻约束，或先调用 `build_problem()` 拿到 `WBCProblem`，加完再交给 `wbc.solver` 求解；
- 求解结果 `wbc.last_result` 是 `WBCResult`，各任务残差按任务名报告。

### 11.5 建议记录的数据

这些既用于调试，也是评估 WBC 能力边界的量化指标：

| 数据 | 用途 |
|---|---|
| 每个任务的残差 $\lVert J_i\mathbf{v} - \dot{x}_{d,i}\rVert$、误差 $\lVert e_i\rVert$ | 跟踪精度 |
| HQP 每层残差 $r_k^*$ | 从哪一层开始“资源不够” |
| 软约束松弛量 $s$ | 是否到达能力边界 |
| 各任务雅可比的最小奇异值 $\sigma_{\min}$ | 离奇异还有多远 |
| 关节到限位的最小余量 | 限位裕度 |
| $\lVert\mathbf{v}\rVert_\infty$、相邻周期 $\mathbf{v}$ 的差 | 平滑性 / 跳变 |
| 求解状态、nWSR、耗时 | 实时性 |
| active 约束集合 | 是什么在“卡”着 |
| $\lVert q_{\text{cmd}} - q_{\text{meas}}\rVert$ | 底层跟踪是否跟得上 |

> **残差和误差的区别**：残差是“这一步想走的没走到的部分”，反映 WBC 本身的取舍；误差是“离目标还有多远”，还包含收敛过程和跟踪滞后。评估能力边界时，残差更能说明问题。

---

## 12. 调参与诊断

### 12.1 调参顺序

每一步只改一个量：

1. 单个末端任务，只加关节限位，调 $K_p$ 和 $\lambda$，确认平稳收敛
2. 调姿态权重 $\rho$，平衡位置与姿态
3. 加第二个末端任务、躯干任务，调任务权重（HQP：定层级）
4. 调分组正则化，控制底盘 / 躯干的参与程度
5. 加躯干范围等约束；WQP 调 $w_1$、$w_2$
6. 跟踪运动目标时加前馈，确认滞后误差消失

### 12.2 参数权衡

| 想要 | 调整 | 代价 |
|---|---|---|
| 收敛更快 | 增大 $K_p$、速度上限 | 易过冲，关节速度大 |
| 奇异附近更平稳 | 增大 $\lambda$ | 跟踪误差增大 |
| 跟踪更精确 | 减小 $\lambda$、增大任务权重 | 奇异附近易跳变 |
| 底盘 / 躯干更多参与 | 减小对应的分组正则化 | 底盘 / 躯干运动增多 |
| 软约束更“硬” | 增大 $w_1$ | 基本无副作用 |

**HQP 的不同之处**：主要调的是层级顺序。按 §8.3 严格写法，高层只有 $\epsilon$，平滑性只靠最低层的 $\lambda$、$R$，高层任务接近奇异时会逐周期跳变；实际控制时用 `regularization="all"`、任务误差限幅和自适应阻尼（§8.3 补充）。

### 12.3 常见问题

| 症状 | 可能原因 | 排查 / 修复 |
|---|---|---|
| 误差不减小 | 权重太小、$\lambda$ 太大、关节贴限位、约束 active | 看任务残差、限位余量、对偶解 |
| 跟踪运动目标时误差恒定偏大 | 没加前馈，误差是滞后 $v/K_p$ | 加前馈（§4.2） |
| 关节振荡 | $K_p$ 过大、active set 来回切换 | 减小 $K_p$，增大 $\lambda$，约束边界留余量 |
| 关节跳变 | 接近奇异且 $\lambda$ 过小；失败输出未处理；约束量不连续（如角度跨越 $\pm\pi$）；冻结 / 任务切换瞬间 | 监控 $\sigma_{\min}$；失败时保持上一指令；切换时平滑过渡 |
| 位置到了姿态不对 | $\rho$ 太小，或姿态误差与雅可比坐标系不一致 | 调 $\rho$；核对坐标系（MuJoCo 的 `mj_jacSite` 为世界坐标系） |
| 约束几乎让机器人不动 | 边界漏除 $dt$ | 按 §5.4 检查量纲 |
| 软约束在可行时也被轻微违反 | 只用了二次惩罚 | 加线性惩罚 $w_1$ |
| 反复运动后构型漂移 | 冗余自由度没被约束 | 加低权重 / 最低层姿态任务 |
| HQP 高层残差不为 0 | 上层任务本身做不到（受限位等约束），或锁定约束退化 | 看该层单独求解的残差；处理线性相关的行 |

---

## 附录 A 公式速查

### A.1 任务

$$
\dot{x}_d = \dot{x}_{\text{ff}} + K_p\, e, \qquad
e_{\text{ori}} = \log\!\big(R_d R^T\big)^\vee, \qquad
\tau = \frac{1}{K_p}, \qquad
e_{\text{lag}} = \frac{v_{\text{target}}}{K_p}
$$

$$
f_i(\mathbf{v}) = \frac{1}{2}\big\|J_i\mathbf{v} - \dot{x}_{d,i}\big\|^2_{W_i}, \qquad
W_i = \text{diag}(1, 1, 1,\; \rho, \rho, \rho), \qquad
\text{任务代价} = w_i\, f_i
$$

### A.2 约束

$$
lb_i = \text{clamp}\!\left(\beta\,\frac{q_{\min,i} - q_i}{dt},\; \pm\dot{q}_{\max,i}\right), \qquad
ub_i = \text{clamp}\!\left(\beta\,\frac{q_{\max,i} - q_i}{dt},\; \pm\dot{q}_{\max,i}\right)
$$

$$
\frac{c_{\min} - c(q)}{dt} \;\leq\; J_c\,\mathbf{v} \;\leq\; \frac{c_{\max} - c(q)}{dt}, \qquad
J_{\text{pitch}} = \cos\psi\, J_{\omega,y} - \sin\psi\, J_{\omega,x}
$$

$$
\text{软约束：}\; lb_S - s \leq S\mathbf{v} \leq ub_S + s, \quad s \geq 0, \qquad
\text{代价} \mathrel{+}= w_1 \sum s + \frac{w_2}{2}\|s\|^2
$$

### A.3 WQP

$$
\min_{\mathbf{v}} \; \sum_i w_i f_i(\mathbf{v}) + \frac{\lambda}{2}\|\mathbf{v}\|^2 + \frac{1}{2}\mathbf{v}^T R\,\mathbf{v}
\quad \text{s.t.} \quad
lb \le \mathbf{v} \le ub,\;\; lb_A \le A\mathbf{v} \le ub_A
$$

$$
H = \sum_i w_i J_i^T W_i J_i + \lambda I + R, \qquad
g = -\sum_i w_i J_i^T W_i\, \dot{x}_{d,i}
$$

加入软约束时：$z = [\mathbf{v};\, s]$，$H_z = \text{blkdiag}(H,\, w_2 I)$，$g_z = [g;\; w_1\mathbf{1}]$（§7.2.2）。

### A.4 HQP 第 $k$ 层

$$
\min_{\mathbf{v}} \; \frac{1}{2}\big\|J_k\mathbf{v} - \dot{x}_{d,k}\big\|_{W_k}^2 + \begin{cases} \frac{\epsilon}{2}\|\mathbf{v}\|^2, & k < K \\ \frac{\lambda}{2}\|\mathbf{v}\|^2 + \frac12\mathbf{v}^T R\,\mathbf{v}, & k = K \end{cases}
\quad \text{s.t.} \quad
J_j\mathbf{v} - \dot{x}_{d,j} = r_j^* \;(j < k),\;\;
\mathbf{v} \in \mathcal{C}
$$

### A.5 无约束时的闭式特例

$$
\text{WQP} \to \text{加权最小二乘：}\quad
\mathbf{v}^* = \Big(\sum_i w_i J_i^T W_i J_i + \lambda I + R\Big)^{-1} \sum_i w_i J_i^T W_i\, \dot{x}_{d,i}
$$

$$
\text{HQP} \to \text{零空间投影：}\quad
\mathbf{v}_k = \mathbf{v}_{k-1} + \big(J_k N_{k-1}\big)^\dagger \big(\dot{x}_{d,k} - J_k\mathbf{v}_{k-1}\big)
$$

### A.6 输出

$$
q_{\text{cmd}} \leftarrow q_{\text{cmd}} + \mathbf{v}^*\, dt
$$

---

## 附录 B 符号与单位

| 符号 | 单位 | 含义 |
|---|---|---|
| $q$ | rad / m | 受控自由度位置（底盘平移为 m） |
| $\mathbf{v} = \dot{q}$ | rad/s / m/s | 决策变量 |
| $dt$ | s | 控制周期 |
| $\dot{q}_{\max}$ | rad/s / m/s | 速度上限 |
| $J$ | — | 任务雅可比（行的单位随任务而定） |
| $e$ | m / rad | 任务误差 |
| $K_p$ | 1/s | 反馈增益，$\tau = 1/K_p$ |
| $\dot{x}_{\text{ff}}$ | m/s / rad/s | 前馈：目标速度 |
| $w_i$、$W_i$、$\rho$ | — | 任务权重（标量）、方向权重 $\text{diag}(1,1,1,\rho,\rho,\rho)$（不含 $w_i$）、姿态相对位置权重 |
| $\lambda$、$R$、$\epsilon$ | — | 全局阻尼、分组正则化（对角阵，§1 中即 $\mathbf{W}_p$）、HQP 高层的极小阻尼（对照表见 §4.4） |
| $\beta$ | — | 限位缓冲系数，$(0, 1]$ |
| $A$、$lb_A$、$ub_A$ | — / 速度量纲 | 硬线性约束 |
| $S$、$lb_S$、$ub_S$ | — / 速度量纲 | 软约束（带松弛变量） |
| $s$；$w_1$、$w_2$ | 同约束；— | 松弛变量；线性、二次惩罚权重 |
| $r_k^*$ | 同任务 | HQP 第 $k$ 层的最优残差 |

---

## 附录 C 加权最小二乘的闭式解推导

§1.4 引用块中 $\mathbf{W}_p$ 正则方式的 $\mathbf{A}\mathbf{v} = \mathbf{b}$ 逐项展开推导。通用形式的推导见 §7.1。

### C.1 为什么是 $\mathbf{A}^{-1}\mathbf{b}$？

目标函数（$\mathbf{b}_1 = \dot{\mathbf{x}}_{\text{des}} = K\mathbf{e}_{\text{pose}}$，$\mathbf{v}_p = \mathbf{v}_{\text{posture}}$）：

$$
\mathcal{L}(\mathbf{v}) = \frac12(\mathbf{J}\mathbf{v} - \mathbf{b}_1)^T\mathbf{W}(\mathbf{J}\mathbf{v} - \mathbf{b}_1) + \frac{\lambda}{2}\mathbf{v}^T\mathbf{v} + \frac12(\mathbf{v} - \mathbf{v}_p)^T\mathbf{W}_p(\mathbf{v} - \mathbf{v}_p)
$$

展开：

$$
\begin{aligned}
\mathcal{L} &= \tfrac12\mathbf{v}^T\mathbf{J}^T\mathbf{W}\mathbf{J}\mathbf{v} - \mathbf{b}_1^T\mathbf{W}\mathbf{J}\mathbf{v} + \tfrac12\mathbf{b}_1^T\mathbf{W}\mathbf{b}_1 \\
&\quad + \tfrac{\lambda}{2}\mathbf{v}^T\mathbf{v} \\
&\quad + \tfrac12\mathbf{v}^T\mathbf{W}_p\mathbf{v} - \mathbf{v}_p^T\mathbf{W}_p\mathbf{v} + \tfrac12\mathbf{v}_p^T\mathbf{W}_p\mathbf{v}_p
\end{aligned}
$$

求导并令其为零：

$$
\frac{\partial\mathcal{L}}{\partial\mathbf{v}} = (\mathbf{J}^T\mathbf{W}\mathbf{J} + \lambda\mathbf{I} + \mathbf{W}_p)\mathbf{v} - \mathbf{J}^T\mathbf{W}\mathbf{b}_1 - \mathbf{W}_p\mathbf{v}_p = \mathbf{0}
$$

即 $\mathbf{A}\mathbf{v} = \mathbf{b}$：

- $\mathbf{A} = \mathbf{J}^T\mathbf{W}\mathbf{J} + \lambda\mathbf{I} + \mathbf{W}_p$（$n \times n$ 对称正定，单臂 $n = 14$）
- $\mathbf{b} = \mathbf{J}^T\mathbf{W}\mathbf{b}_1 + \mathbf{W}_p\mathbf{v}_p$

解为 $\mathbf{v} = \mathbf{A}^{-1}\mathbf{b}$。在 QP 的标准形式 $\frac{1}{2}\mathbf{v}^T\mathbf{H}\mathbf{v} + \mathbf{g}^T\mathbf{v}$ 下，对应 $\mathbf{H} = \mathbf{A}$、$\mathbf{g} = -\mathbf{b}$。加入不等式约束后，就是 [§7](#7-加权-qpwqp) 的加权 QP。

### C.2 为什么 $\lambda\mathbf{I}$ 保证 $\mathbf{A}$ 可逆？

- $\mathbf{J}^T\mathbf{W}\mathbf{J}$ 是 $n \times n$ 矩阵，秩最多为任务维度 $m$。本项目 $n = 14 > m = 6$，所以它**在任何姿态下都是奇异的**，不只在奇异位形下才奇异；奇异位形下秩进一步下降。
- $\mathbf{J}^T\mathbf{W}\mathbf{J}$ 和 $\mathbf{W}_p$ 都是半正定的，加上 $\lambda\mathbf{I}$（$\lambda > 0$）后，所有特征值至少为 $\lambda$。
- 因此 $\mathbf{A}$ 必定正定、可逆。

这就是 **Tikhonov 正则化**（阻尼最小二乘）。严格来说，本项目中 $\mathbf{W}_p$ 本身已经是正定对角阵，单靠它也能保证可逆；$\lambda\mathbf{I}$ 的主要作用是对所有自由度提供统一的阻尼。
