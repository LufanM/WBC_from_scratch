#!/usr/bin/env python3
"""
标准臂角(swivel angle psi)零空间控制演示

主任务：双手末端固定
次任务：直接驱动臂角 psi（肘部绕肩-腕轴的自转相位），而不是关节角本身

psi 定义（肩S、肘E、腕W；参考方向V取世界坐标-Z，重力方向）：
    e = E - S,  w = W - S
    w_hat = w / ||w||
    l = (w x V) x w,  l_hat = l / ||l||
    p = e - w_hat (w_hat . e)          # e 垂直于 w 的分量
    psi = atan2( w_hat . (V_hat x p),  V_hat . p )

雅可比 J_c（psi = J_c @ q_dot）按公式：
    J_c = (w_hat x p_hat)^T / ||p|| * J1
        + { (V^T w / ||l||) (w_hat x l_hat)^T
            - (w_hat^T e / (||w|| ||p||)) (w_hat x p_hat)^T } * J2

其中 J1 = d(elbow_pos)/dq, J2 = d(wrist_pos)/dq（世界坐标，肩点在本场景下
对受控关节固定，因此 d(e)/dq = J1, d(w)/dq = J2，不需要额外减去肩部雅可比）。
"""

import numpy as np
import mujoco
import mujoco.viewer
from robot_model import load_model
from wbc_function import WholeBodyController

V_WORLD = np.array([0.0, 0.0, -1.0])  # 参考方向：重力方向


def compute_swivel_geometry(p_shoulder, p_elbow, p_wrist, V=V_WORLD):
    """返回 psi 及计算过程中的中间几何量（供解析雅可比复用）。"""
    e = p_elbow - p_shoulder
    w = p_wrist - p_shoulder
    w_norm = np.linalg.norm(w)
    w_hat = w / w_norm

    l = np.cross(np.cross(w, V), w)
    l_norm = np.linalg.norm(l)
    l_hat = l / l_norm

    d = w_hat * np.dot(w_hat, e)
    p = e - d
    p_norm = np.linalg.norm(p)
    p_hat = p / p_norm

    V_hat = V / np.linalg.norm(V)
    psi = np.arctan2(np.dot(w_hat, np.cross(V_hat, p)), np.dot(V_hat, p))

    return dict(e=e, w=w, w_hat=w_hat, w_norm=w_norm,
                l=l, l_hat=l_hat, l_norm=l_norm,
                d=d, p=p, p_hat=p_hat, p_norm=p_norm,
                V=V, V_hat=V_hat, psi=psi)


def compute_swivel_angle(p_shoulder, p_elbow, p_wrist, V=V_WORLD):
    return compute_swivel_geometry(p_shoulder, p_elbow, p_wrist, V)["psi"]


def compute_Jc_analytical(geom, J1, J2):
    """按截图公式组装 J_c（1 x n 行向量）。J1: d(elbow)/dq, J2: d(wrist)/dq，均 3 x n。"""
    w_hat, p_hat, p_norm = geom["w_hat"], geom["p_hat"], geom["p_norm"]
    l_hat, l_norm = geom["l_hat"], geom["l_norm"]
    w_norm, e, V = geom["w_norm"], geom["e"], geom["V"]

    term1_row = np.cross(w_hat, p_hat) / p_norm            # (3,)
    coeff_l = np.dot(V, geom["w"]) / l_norm
    coeff_p = np.dot(w_hat, e) / (w_norm * p_norm)
    term2_row = coeff_l * np.cross(w_hat, l_hat) - coeff_p * np.cross(w_hat, p_hat)

    Jc = term1_row @ J1 + term2_row @ J2
    return Jc


def compute_Jc_fd(m, d, qadr, shoulder_body_pos_fn, elbow_body_id, wrist_site_id,
                   eps=1e-6):
    """有限差分校验：对每个受控关节扰动 q，重新做前向运动学，差分 psi。"""
    q0 = d.qpos[qadr].copy()

    def psi_at(q):
        d.qpos[qadr] = q
        mujoco.mj_forward(m, d)
        p_s = shoulder_body_pos_fn()
        p_e = d.xpos[elbow_body_id].copy()
        p_w = d.site_xpos[wrist_site_id].copy()
        return compute_swivel_angle(p_s, p_e, p_w)

    psi0 = psi_at(q0)
    Jc_fd = np.zeros(len(qadr))
    for i in range(len(qadr)):
        q1 = q0.copy()
        q1[i] += eps
        psi1 = psi_at(q1)
        Jc_fd[i] = (psi1 - psi0) / eps

    d.qpos[qadr] = q0
    mujoco.mj_forward(m, d)
    return Jc_fd, psi0


def main(show_psi_vectors=False):
    """show_psi_vectors：在画面中画出臂角的几何量（见下方说明），命令行用 --show-psi 打开。"""
    print("=" * 70)
    print("标准臂角(swivel angle psi)零空间控制演示")
    print("=" * 70)
    print()

    m, d, _ = load_model(0)

    for i in range(1, 5):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"torso_joint{i}")
        if jid >= 0:
            d.qpos[m.jnt_qposadr[jid]] = 0.0

    arm_pose = {
        "left_joint1": 0.0, "left_joint2": 0.0, "left_joint3": 0.0,
        "left_joint4": np.deg2rad(-90), "left_joint5": 0.0,
        "left_joint6": 0.0, "left_joint7": 0.0,
        "right_joint1": 0.0, "right_joint2": 0.0, "right_joint3": 0.0,
        "right_joint4": np.deg2rad(-90), "right_joint5": 0.0,
        "right_joint6": 0.0, "right_joint7": 0.0,
    }
    for name, value in arm_pose.items():
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            d.qpos[m.jnt_qposadr[jid]] = value
    mujoco.mj_forward(m, d)

    left_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "left_ee")
    right_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_ee")
    left_shoulder_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_link1")
    right_shoulder_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_link1")
    left_elbow_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_link4")
    right_elbow_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_link4")

    left_pos = d.site_xpos[left_site].copy()
    right_pos = d.site_xpos[right_site].copy()
    left_mat = d.site_xmat[left_site].reshape(3, 3)
    right_mat = d.site_xmat[right_site].reshape(3, 3)
    left_quat = np.empty(4)
    right_quat = np.empty(4)
    mujoco.mju_mat2Quat(left_quat, left_mat.flatten())
    mujoco.mju_mat2Quat(right_quat, right_mat.flatten())

    # 注意：该模型左右臂关节轴约定不是严格镜像的（例如 left_joint3/right_joint3
    # 都用 axis="0 0 1"，而不是像 joint2 那样对右臂取反号），因此即便左右臂
    # 关节角完全对称，psi 的绝对值也不互为相反数（实测静止姿态 psi_L≈+25°，
    # psi_R≈-10°，并非镜像关系）。为了让双臂看起来对称摆动，摆动的是"相对
    # 各自静止值的偏移量"，不是绝对角度的正负号。
    psi_l0 = compute_swivel_angle(
        d.xpos[left_shoulder_body].copy(), d.xpos[left_elbow_body].copy(),
        d.site_xpos[left_site].copy())
    psi_r0 = compute_swivel_angle(
        d.xpos[right_shoulder_body].copy(), d.xpos[right_elbow_body].copy(),
        d.site_xpos[right_site].copy())

    joint_names = (
        [f"left_joint{i}" for i in range(1, 8)] +
        [f"right_joint{i}" for i in range(1, 8)]
    )
    wbc = WholeBodyController(m, d, joint_names, ee_sites=["left_ee", "right_ee"])

    print("初始 psi 与雅可比校验（解析 vs 有限差分）：")
    print()

    # --- 左臂：Jc 解析 vs 有限差分 ---
    p_s_l = d.xpos[left_shoulder_body].copy()
    p_e_l = d.xpos[left_elbow_body].copy()
    p_w_l = d.site_xpos[left_site].copy()
    geom_l = compute_swivel_geometry(p_s_l, p_e_l, p_w_l)

    J1_l_full = wbc.body_jacobian("left_link4")     # 3 x 14 (over all controlled joints)
    J2_l_full = wbc.site_jacobian("left_ee")[:3, :]  # 位置行只取前3行
    Jc_l_analytical = compute_Jc_analytical(geom_l, J1_l_full, J2_l_full)

    Jc_l_fd, psi_l0_check = compute_Jc_fd(m, d, wbc.qadr, lambda: d.xpos[left_shoulder_body].copy(),
                                          left_elbow_body, left_site)

    err_l = np.max(np.abs(Jc_l_analytical - Jc_l_fd))
    print(f"[左臂] psi0={np.rad2deg(psi_l0_check):.2f}deg")
    print(f"  Jc解析: {np.round(Jc_l_analytical, 4)}")
    print(f"  Jc差分: {np.round(Jc_l_fd, 4)}")
    print(f"  最大误差: {err_l:.3e}")
    print()

    # --- 右臂：Jc 解析 vs 有限差分 ---
    p_s_r = d.xpos[right_shoulder_body].copy()
    p_e_r = d.xpos[right_elbow_body].copy()
    p_w_r = d.site_xpos[right_site].copy()
    geom_r = compute_swivel_geometry(p_s_r, p_e_r, p_w_r)

    J1_r_full = wbc.body_jacobian("right_link4")
    J2_r_full = wbc.site_jacobian("right_ee")[:3, :]
    Jc_r_analytical = compute_Jc_analytical(geom_r, J1_r_full, J2_r_full)

    Jc_r_fd, psi_r0_check = compute_Jc_fd(m, d, wbc.qadr, lambda: d.xpos[right_shoulder_body].copy(),
                                          right_elbow_body, right_site)

    err_r = np.max(np.abs(Jc_r_analytical - Jc_r_fd))
    print(f"[右臂] psi0={np.rad2deg(psi_r0_check):.2f}deg")
    print(f"  Jc解析: {np.round(Jc_r_analytical, 4)}")
    print(f"  Jc差分: {np.round(Jc_r_fd, 4)}")
    print(f"  最大误差: {err_r:.3e}")
    print()

    if max(err_l, err_r) < 1e-3:
        print("✅ 解析雅可比与有限差分一致，采用解析法进行零空间控制")
    else:
        print("⚠ 解析雅可比与有限差分不一致，请检查公式实现")
    print()

    print("主任务: 双手末端位置+姿态固定 (12维)")
    print("次任务: 臂角 psi 正弦轨迹摆动（零空间）")
    print()

    viewer = mujoco.viewer.launch_passive(m, d)
    viewer.cam.lookat[:] = [0.0, 0.0, 0.7]
    viewer.cam.distance = 4.5
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -15

    dt = m.opt.timestep
    t = 0.0
    step_count = 0

    max_left_err = 0.0
    max_right_err = 0.0

    # 参数说明：实际摆动幅度会比 psi_amplitude 小，因为 (1) null_gain<1 直接打折，
    # (2) PD 跟踪在给定频率下有相位滞后和幅度衰减，(3) v_post 投影到 N 后
    # Jc@(N@v_post) 不再精确等于 Jc@v_post，也会打一次折。
    # 注意：大部分 body 几何是 contype=0/conaffinity=0（纯视觉网格），MuJoCo
    # 不会对它们做碰撞检测，因此臂角摆动过大时肘部/小臂可能视觉穿透躯干或
    # 对侧手臂而不会报错——这里先调小到较保守的幅度，避免穿透，具体是否
    # 够用需要在图形界面里实际观察确认。
    null_gain = 0.5
    kp_psi = 4.0
    kd_psi = 0.3
    psi_amplitude = np.deg2rad(25)
    psi_freq = 0.5
    omega = 2 * np.pi * psi_freq

    # show_psi_vectors：是否在viewer里画出 l_hat（参考方向, 蓝色箭头）和 p（肘部
    # 径向投影）两个向量，两者之间的夹角就是 psi。为 False 时只渲染末端目标点。
    vec_scale = 0.25   # 向量可视化长度（米），几何量本身很短，放大方便看
    vec_width = 0.0025  # 箭头杆粗细（米），细一点避免遮挡机械臂

    if show_psi_vectors:
        # 机器人渲染改为半透明，这样箭头穿过肩/肘/小臂内部时也能看清楚，
        # 不会被实体网格挡住。
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True

    def draw_psi_vector(geom_slot, origin, direction, length, rgba):
        # 带方向指示的箭头（仅用于 l_hat，它没有实体端点）。注意 MuJoCo 的
        # mjGEOM_ARROW 渲染时把 size[2] 按固定比例拆成"圆柱杆身+圆锥箭头"
        # 两段（见 render_gl3.c），杆身不会精确占满整个 size[2]，视觉上会
        # 比传入的 length 短一截，这对指示方向没问题，但不适合画需要精确
        # 到达端点的实体连线（w、p 改用 draw_psi_line）。
        end = origin + direction * length
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[geom_slot],
            mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3),
            np.eye(3).flatten(), np.array(rgba))
        mujoco.mjv_connector(viewer.user_scn.geoms[geom_slot],
                              mujoco.mjtGeom.mjGEOM_ARROW, vec_width, origin, end)

    def draw_psi_line(geom_slot, start, end, rgba):
        # 精确连接两个真实端点的辅助线段（w: 肩->腕，p: d点->肘），
        # 用 mjGEOM_CAPSULE 而不是 ARROW，端点严格落在 start/end。
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[geom_slot],
            mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
            np.eye(3).flatten(), np.array(rgba))
        mujoco.mjv_connector(viewer.user_scn.geoms[geom_slot],
                              mujoco.mjtGeom.mjGEOM_CAPSULE, vec_width, start, end)

    try:
        while viewer.is_running():
            qd_main, _ = wbc.compute({"left_ee": (left_pos, left_quat),
                                      "right_ee": (right_pos, right_quat)})

            J_left = wbc.site_jacobian(0)
            J_right = wbc.site_jacobian(1)
            J = np.vstack([J_left, J_right])

            reg = 0.01
            J_pinv = J.T @ np.linalg.inv(J @ J.T + reg * np.eye(12))
            N = np.eye(len(joint_names)) - J_pinv @ J

            # 当前几何量与 Jc（每步重新计算，姿态在变化）
            p_s_l = d.xpos[left_shoulder_body].copy()
            p_e_l = d.xpos[left_elbow_body].copy()
            p_w_l = d.site_xpos[left_site].copy()
            geom_l = compute_swivel_geometry(p_s_l, p_e_l, p_w_l)
            J1_l = wbc.body_jacobian("left_link4")
            J2_l = wbc.site_jacobian("left_ee")[:3, :]
            Jc_l = compute_Jc_analytical(geom_l, J1_l, J2_l)

            p_s_r = d.xpos[right_shoulder_body].copy()
            p_e_r = d.xpos[right_elbow_body].copy()
            p_w_r = d.site_xpos[right_site].copy()
            geom_r = compute_swivel_geometry(p_s_r, p_e_r, p_w_r)
            J1_r = wbc.body_jacobian("right_link4")
            J2_r = wbc.site_jacobian("right_ee")[:3, :]
            Jc_r = compute_Jc_analytical(geom_r, J1_r, J2_r)

            # 期望 psi 轨迹：在各自的静止值 psi_l0/psi_r0 上叠加反相摆动。
            # 这个机器人 joint4/5 两级的几何偏置是同向而非镜像的（已用官方
            # GMR 参考模型核实），因此静止姿态下 psi_l0 != -psi_r0，直接令
            # psi_des = ±psi_amplitude*sin(...) 会让两臂摆动中心点不一致，
            # 看起来不对称；叠加各自 psi0 后，两臂都绕自己的静止点对称摆动。
            psi_l_des = psi_l0 + psi_amplitude * np.sin(omega * t)
            psi_l_des_dot = psi_amplitude * omega * np.cos(omega * t)
            psi_r_des = psi_r0 - psi_amplitude * np.sin(omega * t)
            psi_r_des_dot = -psi_amplitude * omega * np.cos(omega * t)

            psidot_l_cmd = kp_psi * (psi_l_des - geom_l["psi"]) + kd_psi * psi_l_des_dot
            psidot_r_cmd = kp_psi * (psi_r_des - geom_r["psi"]) + kd_psi * psi_r_des_dot

            # 最小范数解：把标量速度指令映射回关节速度方向（行向量伪逆）
            v_secondary = np.zeros(len(joint_names))
            denom_l = Jc_l @ Jc_l
            if denom_l > 1e-9:
                v_secondary += (Jc_l / denom_l) * psidot_l_cmd
            denom_r = Jc_r @ Jc_r
            if denom_r > 1e-9:
                v_secondary += (Jc_r / denom_r) * psidot_r_cmd

            qd_null = N @ v_secondary
            qd_total = qd_main + null_gain * qd_null

            for i, jname in enumerate(joint_names):
                jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
                addr = m.jnt_qposadr[jid]
                d.qpos[addr] += qd_total[i] * dt

            mujoco.mj_forward(m, d)

            # 渲染主任务目标点（红色球体）
            viewer.user_scn.ngeom = 0
            identity_mat = np.eye(3).flatten()
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[viewer.user_scn.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.02, 0, 0]), left_pos, identity_mat,
                np.array([1, 0, 0, 0.8])
            )
            viewer.user_scn.ngeom += 1
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[viewer.user_scn.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.02, 0, 0]), right_pos, identity_mat,
                np.array([1, 0, 0, 0.8])
            )
            viewer.user_scn.ngeom += 1

            if show_psi_vectors:
                # w（肩S->腕W）和 p（d点->肘E）是辅助线段，用同一种颜色
                # （橙色），精确连接真实端点，不用 ARROW（其箭头结构会让
                # 杆身视觉上短于设定长度）。d = S + w_hat*(w_hat.e) 是 e
                # 沿 w 方向的投影点，p 从这里连到肘部 E。l_hat 没有实体
                # 端点，仍用箭头指示方向。
                aux_rgba = [1.0, 0.6, 0.1, 0.9]
                d_point_l = p_s_l + geom_l["d"]
                draw_psi_line(viewer.user_scn.ngeom, p_s_l, p_w_l, aux_rgba)
                viewer.user_scn.ngeom += 1
                draw_psi_line(viewer.user_scn.ngeom, d_point_l, p_e_l, aux_rgba)
                viewer.user_scn.ngeom += 1
                draw_psi_vector(viewer.user_scn.ngeom, d_point_l, geom_l["l_hat"],
                                 vec_scale, [0.1, 0.3, 1.0, 0.9])
                viewer.user_scn.ngeom += 1

                d_point_r = p_s_r + geom_r["d"]
                draw_psi_line(viewer.user_scn.ngeom, p_s_r, p_w_r, aux_rgba)
                viewer.user_scn.ngeom += 1
                draw_psi_line(viewer.user_scn.ngeom, d_point_r, p_e_r, aux_rgba)
                viewer.user_scn.ngeom += 1
                draw_psi_vector(viewer.user_scn.ngeom, d_point_r, geom_r["l_hat"],
                                 vec_scale, [0.1, 0.3, 1.0, 0.9])
                viewer.user_scn.ngeom += 1

            viewer.sync()

            left_err = np.linalg.norm(d.site_xpos[left_site] - left_pos) * 1000
            right_err = np.linalg.norm(d.site_xpos[right_site] - right_pos) * 1000
            max_left_err = max(max_left_err, left_err)
            max_right_err = max(max_right_err, right_err)

            if step_count % int(1.0 / dt) == 0:
                null_violation = np.linalg.norm(J @ qd_null)
                print(f"t={t:5.1f}s  L_err={left_err:5.2f}mm  R_err={right_err:5.2f}mm  "
                      f"psi_L={np.rad2deg(geom_l['psi']):6.1f}deg  "
                      f"psi_R={np.rad2deg(geom_r['psi']):6.1f}deg  "
                      f"||J*v_null||={null_violation:.2e}")

            t += dt
            step_count += 1
            if t > 20:
                break

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        viewer.close()

    print()
    print("=" * 70)
    print("演示完成")
    print("=" * 70)
    print(f"最大左手误差: {max_left_err:.2f} mm")
    print(f"最大右手误差: {max_right_err:.2f} mm")
    if max_left_err < 5 and max_right_err < 5:
        print("✅ PASS - 双手末端固定成功，臂角零空间控制正确")
    else:
        print("❌ FAIL - 末端误差过大")
    print()


if __name__ == "__main__":
    import sys
    main(show_psi_vectors="--show-psi" in sys.argv)
