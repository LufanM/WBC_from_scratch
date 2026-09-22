"""
simulate.py — MuJoCo simulation that validates the whole-body controller.

Loads the Galaxea R1 Pro, spawns a visible target marker in the reachable
workspace, runs the WBC velocity loop (base + torso + right arm), and reports
how accurately the right end-effector reaches the target.

Run:
    python simulate.py                 # interactive viewer
    python simulate.py --headless      # no window (CI / verification)
    python simulate.py --target 10.9 -0.3 1.1
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import mujoco

from robot_model import (RIGHT_ARM_JOINTS, TORSO_JOINTS, BASE_JOINTS,
                         load_model, add_marker, HOME_POSE)
from wbc_controller import WholeBodyController, quat_error, rpy_to_quat

CONTROLLED = BASE_JOINTS + TORSO_JOINTS + RIGHT_ARM_JOINTS
EE_SITE = "right_ee"

# Dynamic mode: how far ahead of the measured joint position the servo setpoint
# may run (rad / m). Large enough that gravity load is held by a real standing
# error, small enough that an unreachable command cannot wind up.
MAX_LEAD = 0.15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--target", type=float, nargs=3, default=None,
                    help="target x y z in world frame")
    ap.add_argument("--target-rpy", type=float, nargs=3, default=None,
                    help="target orientation roll pitch yaw (rad); "
                         "default keeps the starting EE orientation")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--no-base", action="store_true",
                    help="disable mobile-base planning (torso+arm only)")
    ap.add_argument("--dynamic", action="store_true",
                    help="drive MuJoCo velocity actuators instead of "
                         "integrating kinematically (kinematic is default and "
                         "tracks exactly)")
    args = ap.parse_args()

    gui = not args.headless
    model, data, markers = load_model(n_markers=1)
    wbc = WholeBodyController(model, data, CONTROLLED, EE_SITE,
                             base_free=not args.no_base)

    # Start from the home posture (arms folded in front, torso upright).
    for name, val in HOME_POSE.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            data.qpos[model.jnt_qposadr[jid]] = val
    mujoco.mj_forward(model, data)

    # Verification gate: the site Jacobian must match finite differences before
    # we trust the whole-body solve built on top of it.
    jac_err = wbc.check_jacobian()
    print(f"[sim] Jacobian finite-diff check : max err = {jac_err:.2e}"
          f"  ({'OK' if jac_err < 1e-3 else 'BAD'})")

    ee0, quat0 = wbc.get_ee_pose()
    if args.target is None:
        # Default target: offset from the current EE, well within reach.
        target_pos = ee0 + np.array([0.25, -0.15, 0.15])
    else:
        target_pos = np.array(args.target, float)
    if args.target_rpy is not None:
        target_quat = rpy_to_quat(args.target_rpy)
    else:
        target_quat = quat0

    print(f"[sim] EE start   : {np.round(ee0, 3)}")
    print(f"[sim] EE target  : {np.round(target_pos, 3)}")
    add_marker(model, data, markers[0], target_pos)

    dt = model.opt.timestep
    pos_tol, ang_tol = 0.01, 0.05   # 1 cm, ~2.9 deg
    settle_steps, converged_at = 0, None

    # Dynamic mode: resolve each controlled joint to its position servo and
    # seed the setpoint at the current pose so step 0 commands no jump.
    act_id = {}
    q_cmd = wbc.get_joint_positions()
    if args.dynamic:
        for name, q0_i in zip(wbc.joint_names, q_cmd):
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                    f"{name}_p")
            if aid < 0:
                raise KeyError(f"no position actuator for joint {name!r}")
            act_id[name] = aid
            data.ctrl[aid] = q0_i

    viewer = None
    if gui:
        # `from ... import` rather than `import mujoco.viewer`: the latter is a
        # local binding for `mujoco` and would shadow the module-level import.
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(model, data)

    try:
        for step in range(args.steps):
            qd, ep, ea = wbc.compute(target_pos, target_quat)

            if args.dynamic:
                # Integrate the planned velocity into a position setpoint and
                # let the joint servos track it through the physics. Unlike the
                # kinematic path this carries real actuator lag and gravity
                # load, so the reported error includes tracking error.
                # Integrate command-on-command so a standing setpoint error
                # remains to hold the limb against gravity, but clamp how far
                # the setpoint may lead the measured position. The wheeled base
                # cannot slide as fast as the QP asks, and unclamped that
                # shortfall winds the setpoint metres ahead of the robot;
                # re-anchoring on the measurement instead would leave only one
                # step of error, too little torque to hold the arm up at all.
                q_cmd = np.clip(q_cmd + qd * dt, wbc.lower, wbc.upper)
                q_meas = wbc.get_joint_positions()
                q_cmd = np.clip(q_cmd, q_meas - MAX_LEAD, q_meas + MAX_LEAD)
                for name, target in zip(wbc.joint_names, q_cmd):
                    aid = act_id.get(name)
                    if aid is not None:
                        data.ctrl[aid] = target
                mujoco.mj_step(model, data)
            else:
                # Resolved-rate kinematic integration: validates the
                # controller math without actuator dynamics.
                wbc.integrate(qd, dt)

            if ep < pos_tol and ea < ang_tol:
                settle_steps += 1
                if settle_steps >= 50 and converged_at is None:
                    converged_at = step
            else:
                settle_steps = 0
                converged_at = None  # error grew again — invalidate

            if step % 200 == 0:
                print(f"[sim] step {step:4d}  |e_pos|={ep*1000:7.2f} mm  "
                      f"|e_ang|={np.degrees(ea):6.2f} deg")

            if viewer is not None:
                if not viewer.is_running():
                    break
                viewer.sync()
                time.sleep(dt)
            if converged_at is not None and step > converged_at + 100:
                break

        ee_final, quat_f = wbc.get_ee_pose()
        final_ep = float(np.linalg.norm(target_pos - ee_final))
        final_ea = float(np.linalg.norm(quat_error(quat_f, target_quat)))

        print("\n================ RESULT ================")
        print(f"EE final position : {np.round(ee_final, 4)}")
        print(f"Target position   : {np.round(target_pos, 4)}")
        print(f"Final pos error   : {final_ep*1000:.2f} mm")
        print(f"Final ang error   : {np.degrees(final_ea):.2f} deg")
        print(f"Base (x, y, yaw)  : {np.round(wbc.get_base_pose(), 3)}")
        if converged_at is not None:
            print(f"Converged at step : {converged_at}  "
                  f"({converged_at*dt:.2f} s sim time)")
        ok = (final_ep < pos_tol) and (final_ea < ang_tol)
        print(f"Verdict           : {'PASS' if ok else 'FAIL'}  "
              f"(tol = {pos_tol*1000:.0f} mm, {np.degrees(ang_tol):.1f} deg)")
        print("========================================")

        if viewer is not None:
            print("[sim] holding window 15 s ...")
            time.sleep(15.0)
        return 0 if ok else 1
    finally:
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    raise SystemExit(main())
