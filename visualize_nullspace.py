#!/usr/bin/env python3
"""
零空间优化可视化示例

展示：
1. 3自由度平面机械臂
2. 末端固定在目标点（主任务）
3. 肘部在零空间内上下摆动（次任务）
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches


class Planar3DOFArm:
    """3自由度平面机械臂"""

    def __init__(self, L1=1.0, L2=0.8, L3=0.6):
        self.L = np.array([L1, L2, L3])  # 连杆长度

    def forward_kinematics(self, q):
        """正运动学：关节角 → 末端位置

        Args:
            q: [q1, q2, q3] 关节角

        Returns:
            positions: [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]
                      基座, 关节1, 关节2, 末端
        """
        q1, q2, q3 = q

        # 基座
        p0 = np.array([0, 0])

        # 关节1
        p1 = p0 + self.L[0] * np.array([np.cos(q1), np.sin(q1)])

        # 关节2（肘部）
        a2 = q1 + q2
        p2 = p1 + self.L[1] * np.array([np.cos(a2), np.sin(a2)])

        # 末端
        a3 = a2 + q3
        p3 = p2 + self.L[2] * np.array([np.cos(a3), np.sin(a3)])

        return np.array([p0, p1, p2, p3])

    def jacobian(self, q):
        """雅可比矩阵：关节速度 → 末端速度

        Args:
            q: [q1, q2, q3]

        Returns:
            J: (2, 3) 矩阵
        """
        q1, q2, q3 = q
        a2 = q1 + q2
        a3 = a2 + q3

        # 末端位置对各关节角的偏导
        J = np.array([
            # x方向
            [-self.L[0]*np.sin(q1) - self.L[1]*np.sin(a2) - self.L[2]*np.sin(a3),
             -self.L[1]*np.sin(a2) - self.L[2]*np.sin(a3),
             -self.L[2]*np.sin(a3)],
            # y方向
            [self.L[0]*np.cos(q1) + self.L[1]*np.cos(a2) + self.L[2]*np.cos(a3),
             self.L[1]*np.cos(a2) + self.L[2]*np.cos(a3),
             self.L[2]*np.cos(a3)]
        ])

        return J

    def inverse_kinematics_2d(self, target_xy):
        """逆运动学：末端位置 → 关节角（数值解）"""
        from scipy.optimize import fmin

        def cost(q):
            pos = self.forward_kinematics(q)
            return np.linalg.norm(pos[-1] - target_xy)**2

        # 初始猜测
        q0 = np.array([0.5, 0.5, 0.5])
        q_sol = fmin(cost, q0, disp=False)

        return q_sol


def nullspace_projection_demo():
    """零空间投影演示"""

    print("=" * 70)
    print("零空间优化可视化演示")
    print("=" * 70)
    print()

    # 创建机械臂
    arm = Planar3DOFArm(L1=1.0, L2=0.8, L3=0.6)

    # 目标末端位置
    target_xy = np.array([1.5, 1.2])
    print(f"目标末端位置: {target_xy}")

    # 求一个满足末端约束的初始姿态
    q_init = arm.inverse_kinematics_2d(target_xy)
    print(f"初始关节角: {np.rad2deg(q_init)} deg")

    # 验证末端到达目标
    pos_init = arm.forward_kinematics(q_init)
    print(f"实际末端位置: {pos_init[-1]}")
    print(f"末端误差: {np.linalg.norm(pos_init[-1] - target_xy):.2e} m")
    print()

    # 计算雅可比和零空间
    J = arm.jacobian(q_init)
    print(f"雅可比矩阵 J (2×3):")
    print(J)
    print()

    # 零空间投影矩阵
    J_pinv = J.T @ np.linalg.inv(J @ J.T)  # 伪逆
    N = np.eye(3) - J_pinv @ J  # 投影矩阵

    print(f"零空间投影矩阵 N (3×3):")
    print(N)
    print()

    # 验证零空间性质
    v_test = np.array([1, 2, 3])
    v_null = N @ v_test
    end_vel = J @ v_null
    print(f"测试向量: {v_test}")
    print(f"投影到零空间: {v_null}")
    print(f"末端速度 J*v_null: {end_vel}")
    print(f"末端速度范数: {np.linalg.norm(end_vel):.2e} ≈ 0 ✅")
    print()

    # 生成零空间内的运动序列（肘部上下摆动）
    print("生成零空间运动序列...")
    n_frames = 100
    q_sequence = []

    for i in range(n_frames):
        # 次任务：肘部摆动（正弦波）
        t = i / n_frames * 2 * np.pi
        v_posture = np.array([0, 3 * np.sin(t), 0])  # 只动关节2（肘部）

        # 投影到零空间
        v_null = N @ v_posture

        # 更新关节角（小步长积分）
        dt = 0.01
        q_new = q_init + v_null * dt

        q_sequence.append(q_new)

        # 验证末端是否固定
        pos_new = arm.forward_kinematics(q_new)
        error = np.linalg.norm(pos_new[-1] - target_xy)

        if i % 20 == 0:
            print(f"  帧{i:3d}: 末端误差 {error:.2e} m")

    print()
    print(f"✅ 生成 {n_frames} 帧，末端始终固定在目标")
    print()

    # 可视化
    print("生成可视化动画...")
    visualize_nullspace_motion(arm, q_sequence, target_xy)


def visualize_nullspace_motion(arm, q_sequence, target_xy):
    """可视化零空间运动"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：机械臂运动
    ax1.set_xlim(-0.5, 3)
    ax1.set_ylim(-0.5, 2.5)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('零空间优化：末端固定，肘部摆动')

    # 目标点
    ax1.plot(target_xy[0], target_xy[1], 'r*', markersize=20, label='目标')

    # 机械臂
    line_arm, = ax1.plot([], [], 'o-', linewidth=3, markersize=8, color='blue', label='机械臂')
    line_elbow_trace, = ax1.plot([], [], 'g-', alpha=0.5, linewidth=1, label='肘部轨迹')

    # 右图：关节角变化
    ax2.set_xlim(0, len(q_sequence))
    ax2.set_ylim(-np.pi, np.pi)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('帧数')
    ax2.set_ylabel('关节角 (rad)')
    ax2.set_title('关节角变化（主任务固定）')

    line_q1, = ax2.plot([], [], label='关节1 (肩)')
    line_q2, = ax2.plot([], [], label='关节2 (肘)', linewidth=2)
    line_q3, = ax2.plot([], [], label='关节3 (腕)')
    ax2.legend()

    # 记录肘部轨迹
    elbow_trace = []

    # 记录关节角
    q1_history = []
    q2_history = []
    q3_history = []

    def init():
        line_arm.set_data([], [])
        line_elbow_trace.set_data([], [])
        line_q1.set_data([], [])
        line_q2.set_data([], [])
        line_q3.set_data([], [])
        return line_arm, line_elbow_trace, line_q1, line_q2, line_q3

    def update(frame):
        q = q_sequence[frame]
        positions = arm.forward_kinematics(q)

        # 更新机械臂
        x = positions[:, 0]
        y = positions[:, 1]
        line_arm.set_data(x, y)

        # 记录肘部轨迹
        elbow_trace.append(positions[2])
        if len(elbow_trace) > 1:
            elbow_xy = np.array(elbow_trace)
            line_elbow_trace.set_data(elbow_xy[:, 0], elbow_xy[:, 1])

        # 更新关节角历史
        q1_history.append(q[0])
        q2_history.append(q[1])
        q3_history.append(q[2])

        frames = range(len(q1_history))
        line_q1.set_data(frames, q1_history)
        line_q2.set_data(frames, q2_history)
        line_q3.set_data(frames, q3_history)

        # 更新标题显示当前状态
        end_pos = positions[-1]
        error = np.linalg.norm(end_pos - target_xy)
        ax1.set_title(f'零空间优化：末端固定 (误差={error:.2e}m)，肘部摆动')

        return line_arm, line_elbow_trace, line_q1, line_q2, line_q3

    # 创建动画
    anim = FuncAnimation(fig, update, init_func=init,
                        frames=len(q_sequence), interval=50, blit=True)

    # 添加图例
    ax1.legend(loc='upper left')

    plt.tight_layout()

    # 保存动画
    output_file = '/home/molufan/WBC/nullspace_demo.gif'
    try:
        anim.save(output_file, writer='pillow', fps=20)
        print(f"✅ 动画已保存: {output_file}")
    except Exception as e:
        print(f"⚠️  保存动画失败: {e}")
        print("   显示静态图...")
        plt.savefig('/home/molufan/WBC/nullspace_demo_static.png', dpi=150)
        print("✅ 静态图已保存: nullspace_demo_static.png")

    plt.show()


if __name__ == '__main__':
    # 检查依赖
    try:
        import scipy
    except ImportError:
        print("⚠️  需要安装 scipy:")
        print("   pip install scipy")
        exit(1)

    nullspace_projection_demo()
