# HQP 约束与调参说明

本文说明 `HQPSolver` / `WholeBodyController(solver="hqp")` 里有哪些约束（第 1 节）、参数怎么传（第 2 节）、每个参数是什么（第 3 节）、遇到问题调哪个（第 4、5 节）。推导见 [QP_WBC.md §8](QP_WBC.md#8-分层-qphqp)，完整例子见 [§8.7](QP_WBC.md#87-例子双手躯干底盘分层求解)。

## 1. 约束

### 1.1 总览

| 约束 | 形式 | 出现在哪些层 | 满足不了时 | 来源 |
|---|---|---|---|---|
| 关节位置限位 | 上下界 | 每一层 | 由 clamp 保证有解，超限时以最大速度退回 | 控制器自动加 |
| 关节 / 底盘速度限位 | 上下界 | 每一层 | 同上 | 控制器自动加 |
| 冻结 | 上下界 $lb = ub = 0$ | 每一层 | 不会发生 | `set_frozen` |
| 硬线性约束 | $lb_A \le A\mathbf{v} \le ub_A$ | 每一层 | 第 1 层不可行，**本周期整体失败** | `add_constraint`，`soft=False` |
| 软线性约束 | $lb - s \le A\mathbf{v} \le ub + s$ | 所在层及以下 | 违反量最小，不会导致失败 | `add_constraint`，`soft=True` |
| 上层锁定 | $J_j\mathbf{v} = J_j\mathbf{v}^*$ | 上层之下的每一层 | 不会发生（上层的解本身满足） | 求解器自动加 |

硬约束和软约束的区别只在这一点：硬约束满足不了，整个 QP 不可行；软约束满足不了，就按最小违反量继续往下解。

### 1.2 关节位置与速度限位

第 $i$ 个自由度的上下界由 `joint_velocity_bounds` 算出：

$$
lb_i = \text{clamp}\!\left(\beta\,\frac{q_{\min,i} - q_i}{dt},\ -\dot{q}_{\max,i},\ \dot{q}_{\max,i}\right),\qquad
ub_i = \text{clamp}\!\left(\beta\,\frac{q_{\max,i} - q_i}{dt},\ -\dot{q}_{\max,i},\ \dot{q}_{\max,i}\right)
$$

| 符号 | 参数 | 默认值 | 含义 |
|---|---|---|---|
| $\beta$ | `limit_beta` | 0.5 | 每个周期最多走完到限位距离的多少。越小越早减速 |
| $\dot{q}_{\max}$ | `max_joint_vel` | 2.0 rad/s | 关节速度上限 |
| $\dot{q}_{\max}$（底盘平移） | `max_base_vel` | 0.6 m/s | 底盘 $x$、$y$ 各自限制为 $0.6/\sqrt{2}$（内接正方形近似圆形限制） |
| $dt$ | 构造参数 `dt` | 模型仿真步长 | **必须等于实际控制周期**。填小了，限位附近允许的速度会偏大 |

没有限位的关节（如 `base_yaw`）默认范围是 $\pm\pi$，所以 WQP / HQP 下底盘转不过 $\pm\pi$。

### 1.3 冻结

`set_frozen(joint_names)` 让这些关节 $lb = ub = 0$，每一层都生效，运行中可以随时切换，QP 维度不变。传 `[]` 全部解冻。

全部分量都受冻结影响而控制不了的任务会报错；只有部分分量控制不了时警告一次。

### 1.4 线性约束：用硬约束还是软约束

用 `add_constraint(name, fn)` 注册，`fn(controller)` 每个周期返回一个 `LinearConstraint`。非线性的量（如躯干位置）用 `linearized_constraint` 线性化：

$$
\beta\,\frac{c_{\min} - c}{dt} \;\le\; J_c\,\mathbf{v} \;\le\; \beta\,\frac{c_{\max} - c}{dt}
$$

**建议默认用软约束，并放在最高层**（`priority` 比所有任务都小）：

- 跟踪误差、扰动可能让约束量已经越界。硬约束此时可能不可行，所有任务一起失败；
- 软约束放在最高层：能满足就严格满足，满足不了就违反最少，并且下层不能让违反变大。

只有上下界宽松、当前状态不可能越界的约束，才适合用硬约束。

### 1.5 软约束的分层

软约束的 `priority` 决定它在哪一层：

- 比它**高**的层看不到这个约束；
- **本层**引入松弛量 $s \ge 0$，目标函数加 $w_{\text{lin}}\sum s + \frac{w_{\text{quad}}}{2}\|s\|^2$（默认 $10^3$、$10$），求得 $s^*$；
- 比它**低**的层把 $lb - s^* \le A\mathbf{v} \le ub + s^*$ 当硬约束，没有松弛量。

这一层只有软约束时，$w_{\text{lin}}$、$w_{\text{quad}}$ 的大小不影响结果，不用调。只有和任务放在同一层时，它们才决定约束和任务之间怎么折中。`LinearConstraint` 的 `priority` 对硬约束没有作用。

### 1.6 上层锁定

每层解完后，本层任务的 $J\mathbf{v}$ 在下层被锁定，由求解器自动处理：

- 只锁权重大于 0 的行。权重为 0 的行不算任务的一部分，不占用下层的自由度；
- 锁定的行先做 SVD，只保留线性无关的方向，所以同一方向被多层约束也不会退化；
- 上层已经用完所有自由度时，下面的层直接跳过（`LevelInfo.status == "skipped"`）。

### 1.7 不是约束的东西

下面这些容易被当成约束，其实不是：

- **限位斥力**（`limit_w`）：只是姿态项里的一个偏好，任务需要时可以压过它。不越限靠的是 1.2 节的上下界；
- **任务误差限幅**（`max_pos_error`、`max_rot_error`）：限制的是期望速度，不是位姿；
- **自碰撞**：目前没有作为约束加入控制器，需要时按 1.4 节线性化后加入。

### 1.8 限位斥力和回中：什么时候开

两者都不是约束，都属于姿态速度 $\mathbf{v}_{\text{posture}}$，只在冗余方向上起作用，优先级低于任务（公式见 [QP_WBC.md §1.3.2](QP_WBC.md#132-正则项与阻尼)）。

| | 限位斥力 `limit_w` | 回中 `posture_k` |
|---|---|---|
| 作用 | 关节进入离限位 10% 行程的区域后往回推 | 把关节一直拉向 `q_nominal` |
| 什么时候生效 | 只在限位区内 | 只要没到 `q_nominal` 就生效 |
| 目标不动时 | 推出限位区后就停 | **不会停**，一直走到 `q_nominal` |
| 解决的问题 | 关节顶到限位后丢掉冗余；手臂伸直时肘关节卡在奇异位形 | 长时间运行后构型慢慢走偏 |
| 不解决的问题 | 不保证不越限，那是上下界的事（WLS 没有上下界，靠求解后裁剪） | 不决定“谁来动”，那是 $\mathbf{W}_p$ 的事 |

**限位斥力：默认开着。** 它只在限位附近起作用，平时不影响运动。下面的情况要注意：

- 需要在限位区内长时间保持姿态时，它会把关节往外推，表现为停下后飘移。这时要么让初始姿态和常用姿态避开限位区，要么减小 `limit_w`；
- 手臂要伸到接近极限时不要关：它能把肘关节推离伸直位，和阻尼一起防止抖动（[QP_WBC.md §8.3 补充](QP_WBC.md#83-正则项放在哪一层)）；
- 用 `solver="wls"` 时不要关：WLS 求解时不考虑限位，斥力是求解前唯一提前避开限位的手段。

**回中：默认关掉，确实需要时才开。** 判断标准是：冗余自由度有没有人管。

| 场景 | 回中 | 原因 |
|---|---|---|
| 交互操作、上层规划目标、到位后要静止 | 关（`posture_k = 0`） | 开了就停不下来，表现为飘移 |
| 冗余自由度已被任务占满（如躯干、底盘都有任务） | 关 | 开了也基本没有作用 |
| 长时间连续运行、重复轨迹，构型越走越偏 | 开，取小值 | 这正是回中要解决的问题 |
| 只在某段时间需要回到舒适姿态 | 关，改用临时任务 | 比一直开着更可控 |

开回中时注意：

- **把 `q_nominal` 改成有意义的姿态。** 默认是行程中位，而中位不一定好：手臂的行程中位接近伸直奇异。通常应设为初始姿态（`wbc.q_nominal[:] = q_home`）；
- **$k$ 取小值。** 时间常数是 $1/k$，$k$ 太大时冗余方向的运动会一直很明显；
- **目标不动时也会动。** 这不是 bug，是回中的定义。

回中还可以写成**最低层的任务**（`joint_posture_task` 或自定义任务，放在最大的 `priority`）。这样它有明确的增益和开关，需要时打开，到位后关掉，比一直开着的 `posture_k` 更可控。

| 组合 | 适用场景 |
|---|---|
| 斥力开、回中关 | 默认：交互、规划目标、到位静止 |
| 斥力开、回中开 | 长时间连续运行、重复轨迹 |
| 斥力关、回中关 | 验证纯 $R$ 形式（$\mathbf{v}_{\text{posture}} = 0$），或要求绝对不动 |
| 斥力关、回中开 | 一般不用 |

代码默认值注意：`WholeBodyController` 的 `posture_k` 默认是 $2 \times 10^{-3}$（回中是开着的），交互程序 `demo_hqp_interactive.py` 默认是 0。

### 1.9 交互仿真里的约束

`demo_hqp_interactive.py` 在上面的基础上有：

| 约束 | 设置 | 默认值 |
|---|---|---|
| 躯干安全盒（$x$、$z$） | `RobotConfig.torso_box`；不给时以初始躯干位置为中心，由 `torso_box_margin` 给出 | 向后 0.15、向前 0.25、向下 0.40、向上 0.10 m |
| 安全盒的层 | 第 -1 层，软约束，$\beta = 0.5$ | 在双手之上 |
| 拖动目标的范围 | 目标夹到安全盒内，留余量 `TORSO_MARGIN` | 5 mm |
| 指令领先量 | 参考构型最多领先实测 `CMD_LEAD_MAX` | 0.05 rad |

## 2. 怎么传参数

所有参数都有默认值，定义在 [wbc_function/config.py](wbc_function/config.py)（控制器和求解器）和 [demo_hqp_interactive.py](demo_hqp_interactive.py) 的 `Options`（交互程序）。不给就用默认值。

**控制器**：构造时传 `config`。

```python
cfg = wf.WBCConfig(kp_pos=20.0, posture_k=0.0)
cfg.solver.max_iter = 200
wbc = wf.WholeBodyController(model, data, joints, sites, solver="hqp", config=cfg)
```

增益、权重类的参数构造后也可以直接改，下一周期生效，例如 `wbc.posture_k = 0.0`。求解器的参数只在构造时读取。

**任务**：`priority`、`weight`、`components`、`kp_pos`、`kp_rot` 属于单个任务，注册时给（`frames` 或 `add_frame_task`），之后也可以改，例如 `wbc.task("torso").priority = 2`。

**交互程序**：命令行选项由参数定义自动生成，选项名就是参数名（下划线换成减号）。

| 用法 | 命令 |
|---|---|
| 查看全部选项和默认值 | `python3 demo_hqp_interactive.py --help` |
| 查看当前生效的值 | 在命令后加 `--print-params` |
| 逐个设置 | `--kp-pos 20 --torso-cost 1.0 --max-iter 200` |
| 用文件设置 | `--config params.json`，键为参数名，可以平铺，也可以按 `wbc`、`solver` 嵌套 |

优先级：命令行 > 配置文件 > 默认值。写错参数名会报错。

## 3. 参数表

“选项”一列是交互程序的命令行选项。交互程序的默认值和控制器不同时，写在“默认值”里。

### 3.1 任务跟踪（`WBCConfig`）

| 参数 | 选项 | 默认值 | 含义 |
|---|---|---|---|
| `kp_pos` | `--kp-pos` | 12 | 位置比例增益 (1/s)，时间常数 $1/K_p$ |
| `kp_rot` | `--kp-rot` | 12 | 姿态比例增益 (1/s)。`--kp` 同时设置两者 |
| `pos_weight` | `--pos-weight` | 1 | 方向权重：位置 |
| `rot_weight` | `--rot-weight` | 0.4 | 方向权重：姿态 |
| `max_pos_error` | `--max-pos-error` | 不限；交互程序 0.02 | 位置误差限幅 (m)，0 表示不限 |
| `max_rot_error` | `--max-rot-error` | 不限；交互程序 0.1 | 姿态误差限幅 (rad)，0 表示不限 |

### 3.2 正则与姿态（`WBCConfig`）

| 参数 | 选项 | 默认值 | 含义 |
|---|---|---|---|
| `lambda_reg` | `--lambda` | $10^{-4}$ | 全局阻尼 $\lambda$ |
| `posture_w` | `--posture-w` | $2 \times 10^{-3}$ | 关节的 $\mathbf{W}_p$（动起来的代价） |
| `base_reg_w` | `--base-reg-w` | $10^{-4}$ | 底盘的 $\mathbf{W}_p$ |
| `posture_k` | `--posture-k` | $2 \times 10^{-3}$；交互程序 0 | 回中增益 (1/s)，0 表示不回中（1.8 节） |
| `limit_w` | `--limit-w` | 0.5 | 限位斥力强度，0 表示不要斥力（1.8 节） |
| `limit_zone` | `--limit-zone` | 0.1 | 斥力区：离限位多少比例的行程内生效 |

另有 `wbc.set_joint_cost(joint_names, w)`：单独设置一组关节的 $\mathbf{W}_p$，覆盖上面的 `posture_w`、`base_reg_w`。

### 3.3 限位（`WBCConfig`）

| 参数 | 选项 | 默认值 | 含义 |
|---|---|---|---|
| `max_joint_vel` | `--dq-max` | 2.0 | 关节速度上限 (rad/s) |
| `max_base_vel` | `--max-base-vel` | 0.6 | 底盘平移速度上限 (m/s) |
| `limit_beta` | `--limit-beta` | 0.5 | 位置限位的缓冲系数 $\beta$，越小越早减速 |

控制周期 `dt` 是控制器的构造参数，不在 `WBCConfig` 里，必须等于实际控制周期。

### 3.4 求解器（`SolverConfig`）

一般不需要动。

| 参数 | 选项 | 默认值 | 含义 |
|---|---|---|---|
| `regularization` | `--regularization` | `all` | 正则项放哪里：`all` 每层都放，`last` 只放最后一层 |
| `eps` | `--eps` | $10^{-6}$ | 高层阻尼的相对大小，只为保证 $H$ 正定 |
| `lock_tol` | `--lock-tol` | $10^{-9}$ | 上层锁定的放宽量 |
| `rank_tol` | `--rank-tol` | $10^{-6}$ | 压缩锁定行时的相对奇异值阈值 |
| `max_iter` | `--max-iter` | 100 | 每层最大迭代次数 |
| `max_cputime` | `--max-cputime` | 不限 | 每层最长求解时间 (s) |
| `preset` | `--preset` | `reliable` | qpOASES 选项：`reliable`、`mpc`、`default` |
| `active_tol` | `--active-tol` | $10^{-9}$ | 判断约束 active 的乘子阈值 |

直接构造 `HQPSolver()` 时 `regularization` 默认是 `last`，只有通过 `SolverConfig` 才是 `all`。

### 3.5 单个任务（注册时给）

| 参数 | 默认值 | 含义 |
|---|---|---|
| `priority` | 0 | 所在层，数字越小越优先 |
| `weight` | 1.0 | 任务权重，只在同一层内比较 |
| `components` | 全部 6 个 | 跟踪哪些分量 |
| `kp_pos`、`kp_rot` | 用控制器的值 | 这个任务单独的增益 |

### 3.6 交互程序（`Options`）

**分层、手臂和躯干**

| 参数 | 选项 | 默认值 | 含义 |
|---|---|---|---|
| `hand_priority` | `--hand-priority` | 0 | 双手任务所在层 |
| `torso_priority` | `--torso-priority` | 1 | 躯干任务所在层 |
| `box_priority` | `--box-priority` | -1 | 安全盒（软约束）所在层 |
| `hand_weight` | `--hand-weight` | 1 | 手的任务权重 |
| `torso_weight` | `--torso-weight` | 1 | 躯干的任务权重 |
| `arm_cost` | `--arm-cost` | $2 \times 10^{-3}$ | 手臂关节的 $\mathbf{W}_p$，所有模式生效 |
| `torso_cost` | `--torso-cost` | 0.2 | 躯干关节的 $\mathbf{W}_p$，只在模式 1 生效 |
| `torso_return_gain` | `--torso-return-gain` | 0 | 模式 1 下躯干回初始位置的增益（每周期纠正比例） |

**奇异处理**

| 参数 | 选项 | 默认值 | 含义 |
|---|---|---|---|
| `adaptive_lambda` | `--adaptive-lambda` | 0.05 | 自适应阻尼上限 $\lambda_{\max}$，0 表示不开 |
| `adaptive_lambda_sigma` | `--adaptive-lambda-sigma` | 0.1 | 手臂 $\sigma_{\min}$ 小于它 (m/rad) 时开始加阻尼 |

**躯干安全盒**

| 参数 | 选项 | 默认值 | 含义 |
|---|---|---|---|
| `box_soft` | `--box-soft` / `--no-box-soft` | 软约束 | 用软约束还是硬约束 |
| `box_beta` | `--box-beta` | 0.5 | 缓冲系数 $\beta$ |
| `box_w_lin`、`box_w_quad` | `--box-w-lin`、`--box-w-quad` | $10^3$、10 | 软约束的罚权重 |
| `torso_box` | `--torso-box X_MIN X_MAX Z_MIN Z_MAX` | 用机器人配置 | 世界坐标的安全盒 |
| `torso_box_margin` | `--torso-box-margin 后 前 下 上` | 用机器人配置 | 以初始躯干位置为中心的安全盒 |
| `torso_margin` | `--torso-margin` | 0.005 | 拖动目标离盒子边界的余量 (m) |
| `torso_leash` | `--torso-leash` | 0 | 躯干目标最多领先实测多少 (m)，0 表示不限 |

**运行**

| 参数 | 选项 | 默认值 | 含义 |
|---|---|---|---|
| `solver` | `--solver` | `hqp` | `hqp`、`wqp`、`wls` |
| `control_dt` | `--control-dt` | 0.01 | 控制周期 (s)，必须是物理步长的整数倍 |
| `command_base` | `--command-base` | `reference` | WBC 的计算构型：`reference`、`cmd`、`measured` |
| `cmd_lead_max` | `--cmd-lead-max` | 0.05 | 指令最多领先实测多少 (rad) |
| `kinematic` | `--kinematic` | 关 | 关节直接等于指令，不经过物理仿真 |

**目标插值与显示**

| 参数 | 选项 | 默认值 | 含义 |
|---|---|---|---|
| `hand_vmax`、`hand_wmax` | `--hand-vmax`、`--hand-wmax` | 0.15、0.5 | 手目标的最大线速度 (m/s)、角速度 (rad/s) |
| `torso_vmax` | `--torso-vmax` | 0.08 | 躯干目标的最大线速度 (m/s) |
| `t_min` | `--t-min` | 1.0 | 一段插值的最短时长 (s) |
| `movej_vmax` | `--movej-vmax` | 0.5 | R 键 moveJ 的峰值关节速度 (rad/s) |
| `place_dx` | `--place-dx` | 0.35 | P 键放置点的前向距离 (m) |
| `stress_hand_dx` | `--stress-hand-dx` | 用机器人配置 | B 键压力点的前向距离 (m) |
| `unreachable_error` | `--unreachable-error` | 0.02 | 误差超过它持续 1 s 判为不可达 (m) |
| `print_track`、`print_period` | `--print-track`、`--print-period` | 关、0.5 | 周期打印跟踪情况 |

## 4. 怎么调

### 4.1 调参顺序

1. **定 `dt`**：等于实际控制周期。
2. **定分层**：安全类的不等式在最上面，主任务在中间，次要任务在下面。层数一般 3~5 层。
3. **定 `components`**：只写真正关心的分量，多写的分量会占用下层的自由度。
4. **调 $K_p$**：从小往大加，出现振荡就退回。要满足 $K_p\,dt \ll 1$。运动目标要给前馈，否则滞后 $v/K_p$。
5. **调谁来动**：关节代价，见 4.3 节。
6. **调目标够不到时的表现**：误差限幅和自适应阻尼，见 4.4 节。
7. **最后才动求解器参数**，一般不需要。

### 4.2 谁优先

优先级只由 `priority` 决定，权重改变不了层与层之间的优先级。

| 想要的效果 | 调什么 |
|---|---|
| 手一定优先，躯干让步 | 默认就是，不用调 |
| 躯干优先，手让步 | 让 `torso_priority` 小于 `hand_priority` |
| 手和躯干按比例折中 | 放到同一层（`priority` 相同）或用 `--solver wqp`，再调 `hand_weight`、`torso_weight` 的比值 |

- 任务权重只在同一层内有比较的意义，手和躯干不在同一层时，调它基本没有作用；
- 两个目标不冲突时，权重不改变结果，只有目标互相冲突才看得出差别。

### 4.3 谁来动

冗余自由度给谁用，由关节代价 $\mathbf{W}_p$ 决定，看的是 $\lambda + W_{p,i}$ 的相对大小，再乘上关节对任务的影响程度。靠近机身的关节力臂长，代价相同时会抢着动。

| 想要的效果 | 调什么 |
|---|---|
| 模式 1 下躯干少动、手臂多动 | 调大 `torso_cost`，或调小 `arm_cost` |
| 模式 1 下躯干多动、手臂少动 | 调小 `torso_cost`，或调大 `arm_cost` |
| 底盘少动 | 调大 `base_reg_w` |
| 某组关节少动 | `set_joint_cost` |
| 某组关节严格后动 | 放到更低的层，或冻结。代价只能按比例分担，不是开关 |

建议只调 `torso_cost`，从默认值起每次改 5 倍。比值相同时结果接近。

$\mathbf{W}_p$ 太大时高层任务的残差也会变大（`regularization="all"` 时每层都带这一项），所以只想减少某组关节的运动时，优先用 `set_joint_cost`，不要整体加大 `posture_w`。

### 4.4 目标够不到、接近奇异

| 参数 | 调的方向 | 代价 |
|---|---|---|
| `max_pos_error`、`max_rot_error` | 调小，期望速度上限为 $K_p \times$ 限幅 | 大误差时收敛变慢 |
| `adaptive_lambda` | 调大，接近奇异时阻尼更大 | 奇异附近跟踪变差，正常区不受影响 |
| `adaptive_lambda_sigma` | 调大，更早开始加阻尼 | 影响范围变大 |
| `lambda_reg` | 调大 | 所有时候跟踪都变差，优先用自适应阻尼 |

三者要配合使用，原因见 [QP_WBC.md §8.3 补充](QP_WBC.md#83-正则项放在哪一层)。

## 5. 常见问题

先看 HUD 或 `wbc.last_result`：`levels` 给出每层的状态和残差，`slack` 给出软约束的违反量，`active_bounds` 给出哪些关节顶在上下界上。

| 现象 | 可能原因 | 怎么调 |
|---|---|---|
| 模式 1 下躯干动得太多 | 躯干力臂长，代价相同时抢着动 | 调大 `torso_cost` |
| 调任务权重没反应 | 手和躯干不在同一层，或两个任务不冲突 | 见 4.2 节 |
| 手够不到时全身抖动 | 高层接近奇异，期望速度远超限位 | 见 4.4 节 |
| 停下后关节还在慢慢飘 | 回中一直在走，或关节在限位区内被斥力推 | 见 1.8 节 |
| 低优先级任务不动 | 上层用完了自由度，`LevelInfo.status` 为 `skipped` 或 `n_locked` 接近自由度数 | 减少上层的 `components` |
| 低优先级任务残差大 | 自由度够，但被限位或安全盒卡住 | 看 `active_bounds`，放宽约束或换姿态 |
| 高优先级任务残差不为 0 | 目标够不到或被卡住；残差很小时可能是正则项的影响 | 看该层 `residual` 和 `active_bounds`；减小 `posture_w` |
| 跟踪运动目标有固定滞后 | 没给前馈 | `set_target(..., vel=...)` |
| 躯干运动时手有瞬态误差 | 伺服跟踪滞后，停下后归零 | 调小 `torso_vmax` |
| 低频振荡 | $K_p$ 相对伺服带宽太高 | 调小 `kp_pos`、`kp_rot`，用 `--command-base reference` |
| 靠近限位时过冲 | `dt` 比实际控制周期小 | 传入正确的 `dt`，调小 `limit_beta` |
| 本周期失败（`ok=False`） | 硬线性约束不可行 | 改成软约束放到最高层 |
| `status` 为 `degraded` | 某个低层求解失败，输出的是上一层的解 | 看 `levels` 哪层失败；`max_iterations` 时调大 `max_iter` |
| 底盘转不过 $\pm\pi$ | `base_yaw` 默认范围是 $\pm\pi$ | 目前的已知限制 |
