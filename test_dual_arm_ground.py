"""
test_dual_arm_ground.py — 双臂够地面测试 (Dual-arm reach-to-ground test)

验证双臂全身控制器：让左右两个末端执行器 (end-effector) 同时够到地面附近的
两个低目标点，共享躯干与移动底盘。这是对 WBC 大范围、低位形规划能力的压力
测试——目标高度接近地面，要求躯干大幅前倾/下蹲配合双臂下探。

运行:
    python test_dual_arm_ground.py                 # 图形界面
    python test_dual_arm_ground.py --headless      # 无窗口 (验证/CI)
    python test_dual_arm_ground.py --ground-z 0.15 # 自定义地面目标高度
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import mujoco

from robot_model import (BASE_JOINTS, TORSO_JOINTS, LEFT_ARM_JOINTS,
                         RIGHT_ARM_JOINTS, LEFT_EE_SITE, RIGHT_EE_SITE,
                         load_model, add_marker, HOME_POSE)
from wbc_controller import WholeBodyController, quat_error

# 受控自由度：底盘 3 + 躯干 4 + 左臂 7 + 右臂 7（双臂共享躯干与底盘）
CONTROLLED = BASE_JOINTS + TORSO_JOINTS + LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS
# 两个末端执行器（左、右夹爪中心），顺序决定 target 列表顺序
EE_SITES = [LEFT_EE_SITE, RIGHT_EE_SITE]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--ground-z", type=float, default=0.25,
                    help="双臂目标点高度 (m)")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--no-base", action="store_true")
    args = ap.parse_args()

    gui = not args.headless
    model, data, markers = load_model(n_markers=2)
    wbc = WholeBodyController(model, data, CONTROLLED, EE_SITES,
                             base_free=not args.no_base)

    for name, val in HOME_POSE.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            data.qpos[model.jnt_qposadr[jid]] = val
    mujoco.mj_forward(model, data)

    # 验证门：两个末端的 Jacobian 都必须与有限差分一致
    jac_err = wbc.check_jacobian()
    print(f"[test] Jacobian 有限差分校验: 最大误差 = {jac_err:.2e}"
          f"  ({'OK' if jac_err < 1e-3 else 'BAD'})")

    left0, lq0 = wbc.get_ee_pose(0)
    right0, rq0 = wbc.get_ee_pose(1)
    # 目标：机器人前方、贴近地面的左右两点（保持起始姿态朝向）
    left_target = np.array([left0[0] + 0.15, left0[1] + 0.05, args.ground_z])
    right_target = np.array([right0[0] + 0.15, right0[1] - 0.05, args.ground_z])
    targets_pos = [left_target, right_target]
    targets_quat = [lq0, rq0]

    print(f"[test] 左臂起始 {np.round(left0, 3)} -> 目标 "
          f"{np.round(left_target, 3)}")
    print(f"[test] 右臂起始 {np.round(right0, 3)} -> 目标 "
          f"{np.round(right_target, 3)}")
    add_marker(model, data, markers[0], left_target)
    add_marker(model, data, markers[1], right_target)

    dt = model.opt.timestep
    pos_tol = 0.02          # 2 cm（双臂大范围任务，比单臂宽松）
    settle, converged_at = 0, None

    viewer = None
    if gui:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(model, data)
        viewer.cam.lookat[:] = [0.0, 0.0, 0.7]
        viewer.cam.distance = 4.5
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -15

    try:
        for step in range(args.steps):
            qd, ep, ea = wbc.compute(targets_pos, targets_quat)
            wbc.integrate(qd, dt)

            if max(ep) < pos_tol:
                settle += 1
                if settle >= 50 and converged_at is None:
                    converged_at = step
            else:
                settle = 0
                converged_at = None

            if step % 400 == 0:
                print(f"[test] step {step:4d}  左|e|={ep[0]*1000:6.1f}mm  "
                      f"右|e|={ep[1]*1000:6.1f}mm")

            if viewer is not None:
                if not viewer.is_running():
                    break
                viewer.sync()
                time.sleep(dt)
            if converged_at is not None and step > converged_at + 100:
                break

        lf, lf_q = wbc.get_ee_pose(0)
        rf, rf_q = wbc.get_ee_pose(1)
        lep = float(np.linalg.norm(left_target - lf))
        rep = float(np.linalg.norm(right_target - rf))
        lea = float(np.linalg.norm(quat_error(lf_q, targets_quat[0])))
        rea = float(np.linalg.norm(quat_error(rf_q, targets_quat[1])))

        print("\n============== 双臂够地面 结果 ==============")
        print(f"左臂末端最终 {np.round(lf, 4)}  目标 {np.round(left_target, 4)}")
        print(f"右臂末端最终 {np.round(rf, 4)}  目标 {np.round(right_target, 4)}")
        print(f"左臂位置误差 {lep*1000:.2f} mm  姿态误差 {np.degrees(lea):.2f} deg")
        print(f"右臂位置误差 {rep*1000:.2f} mm  姿态误差 {np.degrees(rea):.2f} deg")
        print(f"底盘 (x, y, yaw) {np.round(wbc.get_base_pose(), 3)}")
        if converged_at is not None:
            print(f"收敛于第 {converged_at} 步 ({converged_at*dt:.2f} s)")
        ok = (lep < pos_tol and rep < pos_tol)  # 位置为硬指标；姿态仅报告
        print(f"裁决: {'PASS' if ok else 'FAIL'}  (位置容差 {pos_tol*1000:.0f} mm)")
        print("============================================")

        if viewer is not None:
            time.sleep(3.0)
        return 0 if ok else 1
    finally:
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    raise SystemExit(main())
