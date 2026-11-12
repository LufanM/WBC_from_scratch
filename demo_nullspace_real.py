#!/usr/bin/env python3
"""
真实机器人零空间任务演示 - Galaxea R1 Pro

演示：末端固定在目标位置，同时肘部和躯干在零空间内自由运动
"""

import numpy as np
import mujoco
import mujoco.viewer
from robot_model import load_model, HOME_POSE
from wbc_function import WholeBodyController


def demo_nullspace_with_real_robot():
    """零空间任务：末端固定，肘部和躯干在零空间内运动"""

    print("=" * 70)
    print("Galaxea R1 Pro - 零空间优化演示")
    print("=" * 70)
    print()

    # 加载模型
    print("加载机器人模型...")
    m, d, _ = load_model(0)  # 运动学模式

    # 初始化到 HOME_POSE
    for name, value in HOME_POSE.items():
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            d.qpos[m.jnt_qposadr[jid]] = value

    mujoco.mj_forward(m, d)

    # 获取右手末端初始位置
    ee_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_ee")
    target_pos = d.site_xpos[ee_site].copy()

    # 从旋转矩阵转换为四元数
    target_mat = d.site_xmat[ee_site].reshape(3, 3)
    target_quat = np.empty(4)
    mujoco.mju_mat2Quat(target_quat, target_mat.flatten())

    print(f"目标末端位置: {np.round(target_pos, 3)}")
    print(f"目标末端姿态: {np.round(target_quat, 3)}")
    print()

    # 控制的关节（躯干+右臂）
    joint_names = (
        [f"torso_joint{i}" for i in range(1, 5)] +
        [f"right_joint{i}" for i in range(1, 8)]
    )

    # 创建控制器
    wbc = WholeBodyController(m, d, joint_names, ee_sites=["right_ee"])

    print(f"控制 {len(joint_names)} 个关节")
    print()

    # 关节索引
    elbow_idx = joint_names.index("right_joint2")
    torso_idx = joint_names.index("torso_joint2")

    print(f"肘部关节: right_joint2 (索引 {elbow_idx})")
    print(f"躯干关节: torso_joint2 (索引 {torso_idx})")
    print()
    print("主任务: 右手末端保持在目标位置")
    print("次任务: 肘部上下摆动 + 躯干前后倾斜（零空间投影）")
    print()

    # 打开查看器
    print("启动 MuJoCo 查看器...")
    print("观察：末端固定，肘部和躯干摆动")
    print("按 Ctrl+C 退出")
    print()

    viewer = mujoco.viewer.launch_passive(m, d)

    # 设置相机视角，确保能看到全身
    viewer.cam.lookat[:] = [0.0, 0.0, 0.7]  # 机器人中心
    viewer.cam.distance = 4.5  # 距离
    viewer.cam.azimuth = 90    # 从右侧看
    viewer.cam.elevation = -15  # 稍微俯视

    dt = m.opt.timestep
    t = 0
    step_count = 0

    max_pos_error = 0
    max_ang_error = 0

    try:
        while viewer.is_running():
            # 1. WBC计算主任务速度
            qd_main, _ = wbc.compute({"right_ee": (target_pos, target_quat)})

            # 2. 计算雅可比矩阵
            J = wbc.site_jacobian(0)  # 6x11

            # 3. 计算零空间投影矩阵
            J_pinv = J.T @ np.linalg.inv(J @ J.T + 0.01 * np.eye(6))  # 阻尼伪逆
            N = np.eye(len(joint_names)) - J_pinv @ J  # 零空间投影

            # 4. 定义次任务期望速度（肘部和躯干摆动）
            v_secondary = np.zeros(len(joint_names))
            
            # 肘部正弦摆动
            elbow_amp = np.deg2rad(20)
            v_secondary[elbow_idx] = elbow_amp * np.cos(2 * np.pi * 0.5 * t)
            
            # 躯干正弦摆动
            torso_amp = np.deg2rad(10)
            v_secondary[torso_idx] = torso_amp * np.sin(2 * np.pi * 0.3 * t)

            # 5. 投影到零空间
            qd_null = N @ v_secondary

            # 6. 合成总速度
            qd_total = qd_main + qd_null

            # 7. 积分更新关节角
            for i, jname in enumerate(joint_names):
                jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
                addr = m.jnt_qposadr[jid]
                d.qpos[addr] += qd_total[i] * dt

            mujoco.mj_forward(m, d)
            viewer.sync()

            # 计算误差
            current_pos = d.site_xpos[ee_site]
            current_mat = d.site_xmat[ee_site].reshape(3, 3)
            current_quat = np.empty(4)
            mujoco.mju_mat2Quat(current_quat, current_mat.flatten())

            pos_error = np.linalg.norm(current_pos - target_pos) * 1000
            quat_error = current_quat - target_quat
            if np.dot(current_quat, target_quat) < 0:
                quat_error = current_quat + target_quat
            ang_error = np.rad2deg(2 * np.arcsin(np.clip(np.linalg.norm(quat_error[1:]), 0, 1)))

            max_pos_error = max(max_pos_error, pos_error)
            max_ang_error = max(max_ang_error, ang_error)

            # 每秒打印
            if step_count % int(1.0 / dt) == 0:
                elbow_q = d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "right_joint2")]]
                torso_q = d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "torso_joint2")]]
                # 验证零空间性质
                null_check = np.linalg.norm(J @ qd_null)
                print(f"t={t:5.1f}s  pos_err={pos_error:6.2f}mm  ang_err={ang_error:5.2f}°  "
                      f"elbow={np.rad2deg(elbow_q):6.1f}°  torso={np.rad2deg(torso_q):6.1f}°  "
                      f"||J·v_null||={null_check:.2e}")

            t += dt
            step_count += 1

            if t > 30:
                break

    except KeyboardInterrupt:
        print("\n用户中断")

    finally:
        viewer.close()

    # 结果
    print()
    print("=" * 70)
    print("零空间任务完成")
    print("=" * 70)
    print(f"运行时间: {t:.1f} s")
    print(f"最大位置误差: {max_pos_error:.2f} mm")
    print(f"最大姿态误差: {max_ang_error:.2f} deg")

    if max_pos_error < 20 and max_ang_error < 5:
        print()
        print("✅ PASS - 末端成功保持固定，零空间运动正常")
        print("   说明：肘部和躯干的摆动完全在零空间内，不影响末端位置")
    else:
        print("⚠️  WARNING - 末端偏离较大")

    print()


if __name__ == "__main__":
    demo_nullspace_with_real_robot()
