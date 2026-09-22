#!/usr/bin/env python3
"""
真实机器人零空间任务演示 - 两种场景

场景1：双手末端固定，仅动肘部（大幅摆动）
场景2：双手末端固定，肘部+底盘协同运动

python3 demo_nullspace_scenarios.py 1    # 双手固定+仅肘部大幅摆动
python3 demo_nullspace_scenarios.py 2    # 双手固定+肘部+底盘协同
"""

import numpy as np
import mujoco
import mujoco.viewer
from robot_model import load_model, HOME_POSE
from wbc_controller import WholeBodyController
import sys


def scenario_1_elbow_only():
    """场景1：双手末端固定，仅肘部大幅摆动（底盘固定基座）"""

    print("=" * 70)
    print("场景1：双手末端固定，肘部大幅慢速摆动")
    print("=" * 70)
    print()

    # 加载模型（底盘作为固定基座）
    m, d, _ = load_model(0)

    # 初始化到小臂水平姿态
    # 躯干归零
    for i in range(1, 5):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"torso_joint{i}")
        if jid >= 0:
            d.qpos[m.jnt_qposadr[jid]] = 0.0

    # 设置双臂到小臂水平姿态
    # joint4控制小臂旋转，设为±90度使小臂水平
    arm_horizontal_pose = {
        "left_joint1": np.deg2rad(0),      # 肩部前后
        "left_joint2": np.deg2rad(0),      # 肩部左右
        "left_joint3": np.deg2rad(0),      # 肘部弯曲（保持伸直）
        "left_joint4": np.deg2rad(-90),    # 小臂水平（左臂-90）
        "left_joint5": np.deg2rad(0),      # 腕部
        "left_joint6": np.deg2rad(0),
        "left_joint7": np.deg2rad(0),

        "right_joint1": np.deg2rad(0),
        "right_joint2": np.deg2rad(0),
        "right_joint3": np.deg2rad(0),
        "right_joint4": np.deg2rad(-90),    # 小臂水平（右臂-90）
        "right_joint5": np.deg2rad(0),
        "right_joint6": np.deg2rad(0),
        "right_joint7": np.deg2rad(0),
    }

    for name, value in arm_horizontal_pose.items():
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            d.qpos[m.jnt_qposadr[jid]] = value

    mujoco.mj_forward(m, d)

    # 获取双手初始位置
    left_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "left_ee")
    right_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_ee")

    left_pos = d.site_xpos[left_site].copy()
    right_pos = d.site_xpos[right_site].copy()

    left_mat = d.site_xmat[left_site].reshape(3, 3)
    right_mat = d.site_xmat[right_site].reshape(3, 3)

    left_quat = np.empty(4)
    right_quat = np.empty(4)
    mujoco.mju_mat2Quat(left_quat, left_mat.flatten())
    mujoco.mju_mat2Quat(right_quat, right_mat.flatten())

    print(f"左手目标: {np.round(left_pos, 3)}")
    print(f"右手目标: {np.round(right_pos, 3)}")
    print()

    mujoco.mj_forward(m, d)

    # 获取小臂水平姿态下的双手位置
    left_pos = d.site_xpos[left_site].copy()
    right_pos = d.site_xpos[right_site].copy()

    left_mat = d.site_xmat[left_site].reshape(3, 3)
    right_mat = d.site_xmat[right_site].reshape(3, 3)
    mujoco.mju_mat2Quat(left_quat, left_mat.flatten())
    mujoco.mju_mat2Quat(right_quat, right_mat.flatten())

    print(f"小臂水平姿态 - 左手目标: {np.round(left_pos, 3)}")
    print(f"小臂水平姿态 - 右手目标: {np.round(right_pos, 3)}")
    print()

    # 控制关节（只有双臂14个关节，躯干和底盘都固定）
    joint_names = (
        [f"left_joint{i}" for i in range(1, 8)] +
        [f"right_joint{i}" for i in range(1, 8)]
    )

    wbc = WholeBodyController(m, d, joint_names, ee_sites=["left_ee", "right_ee"])

    print(f"控制 {len(joint_names)} 个关节（仅双臂）")
    print("底盘：固定基座（不在控制列表）")
    print("躯干：固定为0（不在控制列表）")
    print()

    # 关节索引
    left_elbow_idx = joint_names.index("left_joint3")
    right_elbow_idx = joint_names.index("right_joint3")

    print(f"左肘关节: left_joint3 (索引 {left_elbow_idx})")
    print(f"右肘关节: right_joint3 (索引 {right_elbow_idx})")
    print()
    print("初始姿态: left_joint4=-90度, right_joint4=-90度（小臂水平）")
    print("主任务: 双手末端保持固定")
    print("次任务: 双肘对称抬起（90°±70°，范围20-160°，1.0Hz）")
    print("零空间增益: 0.35")
    print("说明: 双肘镜像对称运动，向外弯曲抬起，避免碰撞身体")
    print()

    viewer = mujoco.viewer.launch_passive(m, d)

    # 设置相机视角，确保能看到全身
    viewer.cam.lookat[:] = [0.0, 0.0, 0.7]
    viewer.cam.distance = 4.5
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -15

    dt = m.opt.timestep
    t = 0
    step_count = 0

    max_left_err = 0
    max_right_err = 0

    try:
        while viewer.is_running():
            # 1. WBC计算双手主任务
            qd_main, err_list, _ = wbc.compute(
                [left_pos, right_pos],
                [left_quat, right_quat]
            )

            # 2. 计算双手雅可比（12x14，因为只有14个关节）
            J_left = wbc.site_jacobian(0)  # 6x14
            J_right = wbc.site_jacobian(1)  # 6x14
            J = np.vstack([J_left, J_right])  # 12x14

            # 3. 零空间投影
            J_pinv = J.T @ np.linalg.inv(J @ J.T + 0.01 * np.eye(12))
            N = np.eye(len(joint_names)) - J_pinv @ J

            # 4. 次任务：肘部对称抬起（使用PD控制）
            v_secondary = np.zeros(len(joint_names))

            # 抬肘参数：向外弯曲（0-180度）
            elbow_center = np.deg2rad(90)    # 中心位置90度
            elbow_amplitude = np.deg2rad(70) # ±70度幅度（20-160度范围，避免完全伸直和过度弯曲）
            elbow_freq = 1.0  # 1.0Hz（1秒一个周期）

            omega = 2 * np.pi * elbow_freq

            # 期望轨迹
            elbow_desired = elbow_center + elbow_amplitude * np.sin(omega * t)
            elbow_desired_dot = elbow_amplitude * omega * np.cos(omega * t)

            # 当前关节角度
            left_elbow_jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "left_joint3")
            right_elbow_jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "right_joint3")
            left_elbow_q = d.qpos[m.jnt_qposadr[left_elbow_jid]]
            right_elbow_q = d.qpos[m.jnt_qposadr[right_elbow_jid]]

            # PD控制（左右臂joint3同号输入即产生镜像对称的向外摆动，见下方实测说明）
            kp = 8.0
            kd = 1.5
            v_secondary[left_elbow_idx] = kp * (-elbow_desired - left_elbow_q) + kd * (-elbow_desired_dot)
            v_secondary[right_elbow_idx] = kp * (elbow_desired - right_elbow_q) + kd * elbow_desired_dot

            # 5. 投影并合成（添加零空间增益）
            qd_null = N @ v_secondary
            null_gain = 0.35  # 折中增益
            qd_total = qd_main + null_gain * qd_null

            # 6. 积分
            for i, jname in enumerate(joint_names):
                jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
                addr = m.jnt_qposadr[jid]
                d.qpos[addr] += qd_total[i] * dt

            mujoco.mj_forward(m, d)

            # 渲染主任务目标点（红色球体）
            viewer.user_scn.ngeom = 0  # 清除之前的几何体
            identity_mat = np.eye(3).flatten()

            # 左手目标点（红色）
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[viewer.user_scn.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.02, 0, 0]), left_pos, identity_mat,
                np.array([1, 0, 0, 0.8])  # 红色，半透明
            )
            viewer.user_scn.ngeom += 1

            # 右手目标点（红色）
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[viewer.user_scn.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.02, 0, 0]), right_pos, identity_mat,
                np.array([1, 0, 0, 0.8])  # 红色，半透明
            )
            viewer.user_scn.ngeom += 1

            viewer.sync()

            # 误差统计
            left_err = np.linalg.norm(d.site_xpos[left_site] - left_pos) * 1000
            right_err = np.linalg.norm(d.site_xpos[right_site] - right_pos) * 1000

            max_left_err = max(max_left_err, left_err)
            max_right_err = max(max_right_err, right_err)

            # 计算臂角（arm angle）- 几何方法
            # 左臂：肩部(left_link2) - 肘部(left_link4) - 腕部(left_ee)
            left_shoulder_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_link2")
            left_elbow_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_link4")

            if left_shoulder_body >= 0 and left_elbow_body >= 0:
                p_shoulder_l = d.xpos[left_shoulder_body]
                p_elbow_l = d.xpos[left_elbow_body]
                p_wrist_l = d.site_xpos[left_site]

                w_l = p_wrist_l - p_shoulder_l  # 肩到腕
                e_l = p_elbow_l - p_shoulder_l  # 肩到肘
                w_norm_l = w_l / (np.linalg.norm(w_l) + 1e-9)
                d_l = w_norm_l * np.dot(w_norm_l, e_l)  # 投影到肩-腕线
                p_l = e_l - d_l  # 垂直分量
                arm_angle_l = np.rad2deg(np.arctan2(np.linalg.norm(p_l), np.linalg.norm(d_l)))
            else:
                arm_angle_l = 0.0

            # 右臂：肩部(right_link2) - 肘部(right_link4) - 腕部(right_ee)
            right_shoulder_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_link2")
            right_elbow_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_link4")

            if right_shoulder_body >= 0 and right_elbow_body >= 0:
                p_shoulder_r = d.xpos[right_shoulder_body]
                p_elbow_r = d.xpos[right_elbow_body]
                p_wrist_r = d.site_xpos[right_site]

                w_r = p_wrist_r - p_shoulder_r
                e_r = p_elbow_r - p_shoulder_r
                w_norm_r = w_r / (np.linalg.norm(w_r) + 1e-9)
                d_r = w_norm_r * np.dot(w_norm_r, e_r)
                p_r = e_r - d_r
                arm_angle_r = np.rad2deg(np.arctan2(np.linalg.norm(p_r), np.linalg.norm(d_r)))
            else:
                arm_angle_r = 0.0

            if step_count % int(1.0 / dt) == 0:
                left_elbow_q = d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "left_joint3")]]
                right_elbow_q = d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "right_joint3")]]
                null_check = np.linalg.norm(J @ qd_null)

                print(f"t={t:5.1f}s  L_err={left_err:5.2f}mm  R_err={right_err:5.2f}mm  "
                      f"L_elbow_q={np.rad2deg(left_elbow_q):6.1f}°  R_elbow_q={np.rad2deg(right_elbow_q):6.1f}°  "
                      f"L_arm_angle={arm_angle_l:5.1f}°  R_arm_angle={arm_angle_r:5.1f}°  "
                      f"||J·v||={null_check:.2e}")

            t += dt
            step_count += 1

            if t > 30:
                break

    except KeyboardInterrupt:
        print("\n用户中断")

    finally:
        viewer.close()

    print()
    print("=" * 70)
    print("场景1完成")
    print("=" * 70)
    print(f"最大左手误差: {max_left_err:.2f} mm")
    print(f"最大右手误差: {max_right_err:.2f} mm")

    if max_left_err < 5 and max_right_err < 5:
        print()
        print("✅ PASS - 双手末端固定，肘部大幅摆动成功")
    print()


def scenario_2_elbow_and_base():
    """场景2：双手末端固定，肘部+底盘协同运动"""

    print("=" * 70)
    print("场景2：双手末端固定，肘部+底盘协同运动")
    print("=" * 70)
    print()

    # 加载模型
    m, d, _ = load_model(0)

    # 初始化
    for name, value in HOME_POSE.items():
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            d.qpos[m.jnt_qposadr[jid]] = value

    mujoco.mj_forward(m, d)

    # 获取双手初始位置
    left_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "left_ee")
    right_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_ee")

    left_pos = d.site_xpos[left_site].copy()
    right_pos = d.site_xpos[right_site].copy()

    left_mat = d.site_xmat[left_site].reshape(3, 3)
    right_mat = d.site_xmat[right_site].reshape(3, 3)

    left_quat = np.empty(4)
    right_quat = np.empty(4)
    mujoco.mju_mat2Quat(left_quat, left_mat.flatten())
    mujoco.mju_mat2Quat(right_quat, right_mat.flatten())

    print(f"左手目标: {np.round(left_pos, 3)}")
    print(f"右手目标: {np.round(right_pos, 3)}")
    print()

    # 控制关节（包含底盘+躯干+双臂）
    joint_names = (
        ["base_x", "base_y", "base_yaw"] +
        [f"torso_joint{i}" for i in range(1, 5)] +
        [f"left_joint{i}" for i in range(1, 8)] +
        [f"right_joint{i}" for i in range(1, 8)]
    )

    wbc = WholeBodyController(m, d, joint_names, ee_sites=["left_ee", "right_ee"])

    print(f"控制 {len(joint_names)} 个关节（底盘+躯干+双臂）")
    print()

    # 关节索引
    base_x_idx = 0
    base_y_idx = 1
    left_elbow_idx = joint_names.index("left_joint3")
    right_elbow_idx = joint_names.index("right_joint3")

    print(f"底盘 X: 索引 {base_x_idx}")
    print(f"底盘 Y: 索引 {base_y_idx}")
    print(f"左肘: left_joint3 (索引 {left_elbow_idx})")
    print(f"右肘: right_joint3 (索引 {right_elbow_idx})")
    print()
    print("主任务: 双手末端保持固定")
    print("次任务: 肘部大幅摆动（±80度） + 底盘大幅度慢速移动")
    print()

    viewer = mujoco.viewer.launch_passive(m, d)

    # 设置相机视角，确保能看到全身
    viewer.cam.lookat[:] = [0.0, 0.0, 0.7]
    viewer.cam.distance = 4.5
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -15

    dt = m.opt.timestep
    t = 0
    step_count = 0

    max_left_err = 0
    max_right_err = 0

    try:
        while viewer.is_running():
            # 1. WBC计算双手主任务
            qd_main, err_list, _ = wbc.compute(
                [left_pos, right_pos],
                [left_quat, right_quat]
            )

            # 2. 计算双手雅可比（12x21）
            J_left = wbc.site_jacobian(0)
            J_right = wbc.site_jacobian(1)
            J = np.vstack([J_left, J_right])

            # 3. 零空间投影
            J_pinv = J.T @ np.linalg.inv(J @ J.T + 0.01 * np.eye(12))
            N = np.eye(len(joint_names)) - J_pinv @ J

            # 4. 次任务：肘部大幅摆动 + 底盘大幅度慢速移动
            v_secondary = np.zeros(len(joint_names))

            # 底盘前后移动（0.5m幅度，0.1Hz 慢速）
            base_amp = 0.5
            base_freq = 0.1
            v_secondary[base_x_idx] = base_amp * np.sin(2 * np.pi * base_freq * t)

            # 底盘左右移动（0.4m幅度，0.15Hz，相位差90度）
            v_secondary[base_y_idx] = 0.4 * np.cos(2 * np.pi * 0.15 * t)

            # 肘部大幅摆动（±80度，0.6Hz，左右joint3同号才是镜像对称）
            elbow_amp = np.deg2rad(80)
            elbow_freq = 0.6
            elbow_phase = elbow_amp * np.sin(2 * np.pi * elbow_freq * t)
            v_secondary[left_elbow_idx] = elbow_phase
            v_secondary[right_elbow_idx] = elbow_phase

            # 5. 投影并合成
            qd_null = N @ v_secondary
            qd_total = qd_main + qd_null

            # 6. 积分
            for i, jname in enumerate(joint_names):
                jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
                addr = m.jnt_qposadr[jid]
                d.qpos[addr] += qd_total[i] * dt

            mujoco.mj_forward(m, d)

            # 渲染主任务目标点（红色球体）
            viewer.user_scn.ngeom = 0  # 清除之前的几何体
            identity_mat = np.eye(3).flatten()

            # 左手目标点（红色）
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[viewer.user_scn.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.02, 0, 0]), left_pos, identity_mat,
                np.array([1, 0, 0, 0.8])  # 红色，半透明
            )
            viewer.user_scn.ngeom += 1

            # 右手目标点（红色）
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[viewer.user_scn.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.02, 0, 0]), right_pos, identity_mat,
                np.array([1, 0, 0, 0.8])  # 红色，半透明
            )
            viewer.user_scn.ngeom += 1

            viewer.sync()

            # 误差统计
            left_err = np.linalg.norm(d.site_xpos[left_site] - left_pos) * 1000
            right_err = np.linalg.norm(d.site_xpos[right_site] - right_pos) * 1000

            max_left_err = max(max_left_err, left_err)
            max_right_err = max(max_right_err, right_err)

            if step_count % int(1.0 / dt) == 0:
                base_x = d.qpos[0]
                base_y = d.qpos[1]
                left_elbow_q = d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "left_joint3")]]
                right_elbow_q = d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "right_joint3")]]
                null_check = np.linalg.norm(J @ qd_null)

                print(f"t={t:5.1f}s  L_err={left_err:5.2f}mm  R_err={right_err:5.2f}mm  "
                      f"base_xy=({base_x:5.2f},{base_y:5.2f})  "
                      f"elbow_LR=({np.rad2deg(left_elbow_q):5.1f}°,{np.rad2deg(right_elbow_q):5.1f}°)  "
                      f"||J·v||={null_check:.2e}")

            t += dt
            step_count += 1

            if t > 30:
                break

    except KeyboardInterrupt:
        print("\n用户中断")

    finally:
        viewer.close()

    print()
    print("=" * 70)
    print("场景2完成")
    print("=" * 70)
    print(f"最大左手误差: {max_left_err:.2f} mm")
    print(f"最大右手误差: {max_right_err:.2f} mm")

    if max_left_err < 10 and max_right_err < 10:
        print()
        print("✅ PASS - 双手末端固定，肘部+底盘协同运动成功")
    print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        scenario = int(sys.argv[1])
    else:
        print("请选择场景：")
        print("  1 - 双手末端固定，仅肘部大幅摆动")
        print("  2 - 双手末端固定，肘部+底盘协同运动")
        print()
        scenario = int(input("场景编号: "))

    if scenario == 1:
        scenario_1_elbow_only()
    elif scenario == 2:
        scenario_2_elbow_and_base()
    else:
        print("无效的场景编号")
