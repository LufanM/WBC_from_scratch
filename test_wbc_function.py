"""
test_wbc_function.py — checks for the wbc_function package.

1. qpOASES bridge: a small QP with a known solution, dual sign convention.
2. Soft constraints: exact penalty (s = 0 when feasible), slack when not.
3. WholeBodyController (solver="wls") equals the original closed form
   A v = b, written out independently here.
4. WQPSolver equals WLSSolver when no constraint is active.
5. Closed loop on the R1 Pro (kinematic integration): WholeBodyController
   with "wls" vs "wqp" reaching a far target, joint limits, solve time.
6. HQPSolver: strict priority, null-space equivalence, bounds, soft
   constraints as a level, dependent locks, controller with priorities.
7. Robot configs: generic URDF loader == R1 converter, JSON config,
   demo_hqp_interactive self-check on R1.
8. Parameters: WBCConfig defaults, config reaching the solver, command
   line / config file precedence.
9. Interactive modes: drag state machine stays in the eight modes, each mode
   solves and only moves what it should.

Run:  python3 test_wbc_function.py
"""
from __future__ import annotations

import sys

import numpy as np
import mujoco

import wbc_function as wf

FAILS = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILS.append(msg)


def test_qp_basic():
    print("[1] qpOASES bridge")
    # min 1/2||x - (1, 0.5)||^2  s.t.  x0 <= 0.5,  x0 + x1 <= 2
    prob = wf.WBCProblem(2, lambda_reg=0.0)
    prob.add_task(wf.Task("t", np.eye(2), [1.0, 0.5]))
    prob.set_bounds([-10, -10], [0.5, 10])
    prob.add_constraint(wf.LinearConstraint("sum", [[1.0, 1.0]], -np.inf, 2.0))
    res = wf.WQPSolver().solve(prob)
    check(res.ok, f"solved (status={res.status})")
    check(np.allclose(res.v, [0.5, 0.5], atol=1e-8), f"x = {res.v}")
    check(res.active_bounds.tolist() == [1, 0],
          f"upper bound on x0 reported active: {res.active_bounds}")
    check(res.active_constraints["sum"].tolist() == [0],
          f"sum constraint inactive: {res.active_constraints}")

    # linear constraint active on its upper side
    prob.tasks[0] = wf.Task("t", np.eye(2), [0.0, 3.0])
    res = wf.WQPSolver().solve(prob)
    check(np.allclose(res.v, [-0.5, 2.5], atol=1e-8)
          and res.active_constraints["sum"].tolist() == [1],
          f"sum active (upper): x={res.v}, {res.active_constraints}")

    # lower-side activity
    prob.set_bounds([1.5, -10], [10, 10])
    prob.tasks[0] = wf.Task("t", np.eye(2), [1.0, 0.0])
    res = wf.WQPSolver().solve(prob)
    check(np.allclose(res.v, [1.5, 0.0], atol=1e-8)
          and res.active_bounds.tolist() == [-1, 0],
          f"lower bound active: x={res.v}, flags={res.active_bounds}")

    # no linear constraints at all (nC = 0)
    p2 = wf.WBCProblem(3, lambda_reg=0.0)
    p2.add_task(wf.Task("t", np.eye(3), [1.0, -2.0, 3.0]))
    p2.set_bounds(-1.0, 1.0)
    r2 = wf.WQPSolver().solve(p2)
    check(r2.ok and np.allclose(r2.v, [1.0, -1.0, 1.0]),
          f"box-only QP (nC=0): {r2.v}")

    # hot start on the next tick
    s = wf.WQPSolver()
    s.solve(p2)
    p2.tasks[0] = wf.Task("t", np.eye(3), [0.5, -0.2, 3.0])
    r3 = s.solve(p2)
    check(r3.ok and not r3.cold_start and np.allclose(r3.v, [0.5, -0.2, 1.0]),
          f"hot start: cold={r3.cold_start}, v={r3.v}")


def test_soft():
    print("[2] soft constraints")
    # task wants v = 1, soft constraint v <= 0.2: exact penalty keeps s = 0
    prob = wf.WBCProblem(1, lambda_reg=0.0)
    prob.add_task(wf.Task("t", [[1.0]], [1.0]))
    prob.add_constraint(wf.LinearConstraint("cap", [[1.0]], -np.inf, 0.2,
                                            soft=True, w_lin=10.0, w_quad=1.0))
    res = wf.WQPSolver().solve(prob)
    check(abs(res.v[0] - 0.2) < 1e-8 and res.slack["cap"][0] < 1e-8,
          f"feasible -> held at 0.2, s=0: v={res.v}, s={res.slack['cap']}")

    # quadratic-only penalty leaks: the same push violates the limit
    prob.constraints[0] = wf.LinearConstraint("cap", [[1.0]], -np.inf, 0.2,
                                              soft=True, w_lin=0.0,
                                              w_quad=10.0)
    res = wf.WQPSolver().solve(prob)
    check(res.v[0] > 0.2 + 1e-3,
          f"quadratic-only penalty is violated as expected: v={res.v[0]:.4f}")

    # contradictory: hard v >= 1 and soft v <= 0 -> slack = 1
    prob = wf.WBCProblem(1, lambda_reg=1e-6)
    prob.add_task(wf.Task("t", [[1.0]], [0.0]))
    prob.set_bounds(1.0, 5.0)
    prob.add_constraint(wf.LinearConstraint("cap", [[1.0]], -np.inf, 0.0,
                                            soft=True))
    res = wf.WQPSolver().solve(prob)
    check(res.ok and abs(res.slack["cap"][0] - 1.0) < 1e-6,
          f"infeasible soft constraint -> s = {res.slack['cap']}")


def test_wls_saturation():
    print("[2b] WLSSolver saturation")
    prob = wf.WBCProblem(2, lambda_reg=0.0)
    prob.add_task(wf.Task("t", np.eye(2), [4.0, 2.0]))
    prob.set_bounds(-1.0, 1.0)
    v = wf.WLSSolver("scale").solve(prob).v
    check(np.allclose(v, [1.0, 0.5]), f"scale keeps the direction: {v}")
    v = wf.WLSSolver("clip").solve(prob).v
    check(np.allclose(v, [1.0, 1.0]), f"clip changes the direction: {v}")
    # 0 outside the box (a joint past its limit must move back): scaling
    # cannot help, so the result must still land inside the bounds.
    prob.tasks[0] = wf.Task("t", np.eye(2), [0.5, 0.5])
    prob.set_bounds([1.0, -1.0], [2.0, 1.0])
    v = wf.WLSSolver("scale").solve(prob).v
    check(np.all(v >= prob.lb) and np.all(v <= prob.ub),
          f"scale falls back to clip when 0 is outside the box: {v}")


def _setup_robot():
    from robot_model import (BASE_JOINTS, TORSO_JOINTS, RIGHT_ARM_JOINTS,
                             load_model, HOME_POSE)
    model, data, _ = load_model()
    for name, val in HOME_POSE.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            data.qpos[model.jnt_qposadr[jid]] = val
    mujoco.mj_forward(model, data)
    joints = BASE_JOINTS + TORSO_JOINTS + RIGHT_ARM_JOINTS
    return model, data, joints, BASE_JOINTS


def test_controller_closed_form():
    print("[3] WholeBodyController (wls) vs the original closed form A v = b")
    model, data, joints, _ = _setup_robot()
    wbc = wf.WholeBodyController(model, data, joints, "right_ee")
    wbc.max_joint_vel = np.inf      # compare the raw least-squares solution
    wbc.max_base_vel = np.inf

    # Kinematics against raw MuJoCo calls
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_ee")
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, sid)
    J_raw = np.vstack([jacp, jacr])[:, wbc.dofadr]
    check(np.allclose(wbc.site_jacobian(), J_raw), "site Jacobian matches mj_jacSite")
    check(wbc.check_jacobian() < 1e-5, "Jacobian finite-difference check")

    pos0, quat0 = wbc.get_ee_pose()
    target_pos = pos0 + np.array([0.3, -0.2, 0.1])
    target_quat = wf.rpy_to_quat([0.2, -0.1, 0.3])
    qd, _ = wbc.compute({"right_ee": (target_pos, target_quat)})

    # Original algorithm, written out by hand:
    #   A = J^T W J + lambda I + Wp,  b = J^T W xdot + Wp v_posture
    q = wbc.get_joint_positions()
    xdot = np.concatenate([wbc.Kp_pos * (target_pos - pos0),
                           wbc.Kp_rot * wf.quat_error(quat0, target_quat)])
    W = np.diag(wbc.task_w)
    Wp = np.diag(np.full(wbc.n, wbc.posture_w))
    A = J_raw.T @ W @ J_raw + wbc.lambda_reg * np.eye(wbc.n) + Wp
    b = J_raw.T @ W @ xdot + Wp @ wbc._posture_velocity(q)
    v_ref = np.linalg.solve(A, b)
    err = np.max(np.abs(qd - v_ref))
    check(err < 1e-9, f"same velocity, max diff = {err:.1e}")

    print("[4] WQPSolver == WLSSolver when nothing is active")
    prob, _ = wbc.build_problem({"right_ee": (target_pos, target_quat)})
    v_wls = wf.WLSSolver(saturation=None).solve(prob).v
    res = wf.WQPSolver().solve(prob)
    err = np.max(np.abs(res.v - v_wls))
    check(res.ok and err < 1e-7, f"max diff = {err:.1e}, "
          f"solve {res.solve_time * 1e3:.2f} ms")


def test_target_forms():
    print("[4b] WholeBodyController targets by name (dual arm)")
    from robot_model import (BASE_JOINTS, TORSO_JOINTS, LEFT_ARM_JOINTS,
                             RIGHT_ARM_JOINTS)
    model, data, _, _ = _setup_robot()
    joints = BASE_JOINTS + TORSO_JOINTS + LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS
    wbc = wf.WholeBodyController(model, data, joints, ["left_ee", "right_ee"])
    (pl, ql), (pr, qr) = wbc.get_ee_pose("left_ee"), wbc.get_ee_pose("right_ee")
    tl, tr = pl + [0.1, 0.0, 0.0], pr + [0.0, -0.1, 0.0]

    qd_a, err_a = wbc.compute({"left_ee": (tl, ql), "right_ee": (tr, qr)})
    qd_b, err_b = wbc.compute({"right_ee": (tr, qr), "left_ee": (tl, ql)})
    check(np.allclose(qd_a, qd_b), "key order does not matter")
    check(set(err_a) == {"left_ee", "right_ee"}
          and np.isclose(err_a["left_ee"][0], 0.1) and np.isclose(err_a["right_ee"][0], 0.1),
          f"errors keyed by site: {err_a}")
    check(set(wbc.last_result.task_residuals) == {"left_ee", "right_ee"},
          f"tasks named by site: {sorted(wbc.last_result.task_residuals)}")

    # only the frames given this tick are tracked
    _, err_l = wbc.compute({"left_ee": (tl, ql)})
    check(set(err_l) == {"left_ee"}
          and set(wbc.last_result.task_residuals) == {"left_ee"},
          "a frame without a target this tick is not tracked")

    for bad, desc, exc in (
            (lambda: wbc.compute([(tl, ql), (tr, qr)]), "non-dict targets", TypeError),
            (lambda: wbc.compute({}), "empty targets", ValueError),
            (lambda: wbc.compute({"left_hand": (tl, ql)}), "unknown name", KeyError)):
        try:
            bad()
            check(False, f"{desc} should raise")
        except exc:
            check(True, f"{desc} raises {exc.__name__}")


def test_frames():
    print("[4c] torso / base tracking (frames)")
    from robot_model import (BASE_JOINTS, TORSO_JOINTS, RIGHT_ARM_JOINTS,
                             R1_FRAMES)
    for kind in ("wls", "wqp", "hqp"):
        model, data, _, _ = _setup_robot()
        joints = BASE_JOINTS + TORSO_JOINTS + RIGHT_ARM_JOINTS
        wbc = wf.WholeBodyController(model, data, joints, "right_ee",
                                     solver=kind, frames=R1_FRAMES)
        dt = model.opt.timestep
        p_ee, q_ee = wbc.get_frame_pose("right_ee")
        p_t, q_t = wbc.get_frame_pose("torso")
        p_b, _ = wbc.get_frame_pose("base")
        yaw = 0.4
        # base tracks x, y, rz only; z is given but not used
        base_target = ([p_b[0] + 0.3, p_b[1] - 0.2, p_b[2]],
                       wf.rpy_to_quat([0, 0, yaw]))

        # phase 1: base only
        for _ in range(1500):
            qd, err = wbc.compute({"base": base_target})
            wbc.integrate(qd, dt)
        bx, by, byaw = wbc.get_base_pose()
        check(abs(bx - (p_b[0] + 0.3)) < 2e-3 and abs(by - (p_b[1] - 0.2)) < 2e-3
              and abs(byaw - yaw) < 2e-3,
              f"[{kind}] base reaches (x, y, yaw) target: err {err['base']}")

        # phase 2: switch to hand + torso height, base not tracked any more
        _, q_t2 = wbc.get_frame_pose("torso")
        p_t2, _ = wbc.get_frame_pose("torso")
        p_ee2, q_ee2 = wbc.get_frame_pose("right_ee")
        targets = {"right_ee": (p_ee2 + [0.1, 0.0, 0.05], q_ee2),
                   "torso": (p_t2 + [0.0, 0.0, -0.05], q_t2)}
        for _ in range(1500):
            qd, err = wbc.compute(targets)
            wbc.integrate(qd, dt)
        check(err["right_ee"][0] < 2e-3 and err["torso"][0] < 2e-3
              and err["torso"][1] < 1e-2,
              f"[{kind}] hand + torso targets reached: {err}")

    # base_free=False: tracking the base is impossible -> error;
    # torso y (sideways) is no longer reachable -> one warning, rest still tracked
    import warnings
    model, data, _, _ = _setup_robot()
    wbc_nb = wf.WholeBodyController(model, data, BASE_JOINTS + TORSO_JOINTS + RIGHT_ARM_JOINTS,
                                    "right_ee", base_free=False, frames=R1_FRAMES)
    p_b, q_b = wbc_nb.get_frame_pose("base")
    try:
        wbc_nb.compute({"base": (p_b, q_b)})
        check(False, "base target with base_free=False should raise")
    except ValueError:
        check(True, "base target with base_free=False raises")
    p_t, q_t = wbc_nb.get_frame_pose("torso")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        wbc_nb.compute({"torso": (p_t, q_t)})
        wbc_nb.compute({"torso": (p_t, q_t)})
    msgs = [str(x.message) for x in w]
    check(len(msgs) == 1 and "'y'" in msgs[0],
          f"torso y uncontrollable without base: warned once ({len(msgs)} warning)")

    # input checks
    try:
        wbc.compute({"torso_typo": (p_t, q_t)})
        check(False, "unknown frame name should raise")
    except KeyError:
        check(True, "unknown frame name raises")
    try:
        wbc.compute({"base": (None, q_t)})
        check(False, "missing position for x/y should raise")
    except ValueError:
        check(True, "missing position for x/y raises")


def test_resident_tasks():
    print("[4d] resident tasks: switching, feedforward, freezing, custom task")
    from robot_model import (BASE_JOINTS, TORSO_JOINTS, RIGHT_ARM_JOINTS,
                             R1_FRAMES)
    joints = BASE_JOINTS + TORSO_JOINTS + RIGHT_ARM_JOINTS

    def make(kind):
        model, data, _, _ = _setup_robot()
        return wf.WholeBodyController(model, data, joints, ["right_ee"],
                                      solver=kind, frames=R1_FRAMES), model.opt.timestep

    # --- switching by flags: hand only, then torso only (hand released)
    wbc, dt = make("wqp")
    hand, torso = wbc.task("right_ee"), wbc.task("torso")
    p_h, q_h = wbc.get_frame_pose("right_ee")
    p_t, q_t = wbc.get_frame_pose("torso")
    hand.set_target(p_h + [0.1, 0, 0], q_h).enabled = True
    for _ in range(1000):
        qd, err = wbc.step()
        wbc.integrate(qd, dt)
    check(set(err) == {"right_ee"} and err["right_ee"][0] < 1e-3,
          f"hand task alone reaches its target: {err['right_ee'][0]*1e3:.3f} mm")
    hand.enabled = False
    torso.set_target(p_t + [0, 0, -0.05], q_t).enabled = True
    for _ in range(1000):
        qd, err = wbc.step()
        wbc.integrate(qd, dt)
    check(set(err) == {"torso"} and err["torso"][0] < 1e-3,
          f"after switching, torso task alone reaches its target: {err['torso'][0]*1e3:.3f} mm")

    # --- feedforward removes the v/Kp lag on a moving target
    lags = {}
    for use_ff in (False, True):
        wbc, dt = make("wqp")
        hand = wbc.task("right_ee")
        p0, q0 = wbc.get_frame_pose("right_ee")
        v = np.array([0.1, 0, 0])
        for k in range(1000):
            hand.set_target(p0 + v * k * dt, q0,
                            vel=np.r_[v, 0, 0, 0] if use_ff else None).enabled = True
            qd, err = wbc.step()
            wbc.integrate(qd, dt)
        lags[use_ff] = err["right_ee"][0]
    check(abs(lags[False] - 0.1 / 12.0) < 1e-3 and lags[True] < 1e-4,
          f"moving target lag: {lags[False]*1e3:.2f} mm without feedforward "
          f"(v/Kp = {0.1/12*1e3:.2f} mm), {lags[True]*1e3:.4f} mm with")

    # --- freezing the torso: torso joints stay put, hand still reaches (wls and wqp)
    for kind in ("wls", "wqp"):
        wbc, dt = make(kind)
        wbc.set_frozen(TORSO_JOINTS)
        torso_idx = wbc.kin.mask(TORSO_JOINTS)
        q0 = wbc.get_joint_positions()
        p_h, q_h = wbc.get_frame_pose("right_ee")
        for _ in range(1500):
            qd, err = wbc.compute({"right_ee": (p_h + [0.05, -0.05, 0.05], q_h)})
            wbc.integrate(qd, dt)
        moved = np.max(np.abs(wbc.get_joint_positions()[torso_idx] - q0[torso_idx]))
        check(moved == 0.0 and err["right_ee"][0] < 1e-3,
              f"[{kind}] frozen torso does not move ({moved:.1e}), hand error "
              f"{err['right_ee'][0]*1e3:.3f} mm")
    wbc.set_frozen([])
    check(wbc.frozen_joints == [], "set_frozen([]) unfreezes everything")

    # --- WLS freezing is exact: equals WQP when no bound is active
    w1, _ = make("wls"); w2, _ = make("wqp")
    for w in (w1, w2):
        w.set_frozen(TORSO_JOINTS)
    p_h, q_h = w1.get_frame_pose("right_ee")
    tgt = {"right_ee": (p_h + [0.01, 0, 0], q_h)}
    qd1, _ = w1.compute(tgt); qd2, _ = w2.compute(tgt)
    check(np.max(np.abs(qd1 - qd2)) < 1e-7,
          f"WLS freeze (elimination) == WQP freeze (lb=ub=0): {np.max(np.abs(qd1-qd2)):.1e}")

    # --- custom task: joint-space target for torso_joint1
    wbc, dt = make("wqp")
    i1 = wbc.joint_names.index("torso_joint1")
    q_goal = wbc.get_joint_positions()[i1] + 0.2

    def torso1_task(c):
        S = np.zeros((1, c.n)); S[0, i1] = 1.0
        q_now = c.get_joint_positions()[i1]
        return wf.TaskOutput(J=S, xdot=[2.0 * (q_goal - q_now)],
                             error=abs(q_goal - q_now))

    wbc.add_task("torso1", torso1_task, weight=1.0)
    for _ in range(2000):
        qd, err = wbc.step()
        wbc.integrate(qd, dt)
    check(abs(wbc.get_joint_positions()[i1] - q_goal) < 1e-3
          and set(err) == {"torso1"} and err["torso1"] < 1e-3,
          f"custom task drives torso_joint1 to its goal, error reported: {err}")
    wbc.add_task("bad", lambda c: (np.zeros((1, c.n)), [0.0]))
    try:
        wbc.step()
        check(False, "custom task returning a tuple should raise")
    except TypeError:
        check(True, "custom task must return TaskOutput")

    # --- frozen base: tracking the base is impossible -> error
    wbc, dt = make("wqp")
    wbc.set_frozen(BASE_JOINTS)
    p_b, q_b = wbc.get_frame_pose("base")
    try:
        wbc.compute({"base": (p_b + [0.1, 0, 0], q_b)})
        check(False, "base target with frozen base should raise")
    except ValueError:
        check(True, "base target with frozen base raises")


def test_closed_loop():
    print("[5] closed-loop reach (kinematic), far target")
    results = {}
    for kind in ("wls", "wqp"):
        model, data, joints, _ = _setup_robot()
        wbc = wf.WholeBodyController(model, data, joints, "right_ee",
                                     solver=kind)
        dt = model.opt.timestep
        target_pos = np.array([1.2, -1.5, 1.0])
        _, target_quat = wbc.get_ee_pose()

        q = wbc.get_joint_positions()
        times, worst_violation, converged, settle = [], 0.0, None, 0
        for step in range(3000):
            qd, err = wbc.compute({"right_ee": (target_pos, target_quat)})
            ep = err["right_ee"][0]
            res = wbc.last_result
            times.append(res.solve_time)
            if not res.ok:
                print(f"     step {step}: solver {res.status}")
            # Measure limit violation BEFORE any clamping, then apply the
            # controller's own integration (which clamps, as before).
            q_next = q + qd * dt
            worst_violation = max(worst_violation,
                                  float(np.max(wbc.lower - q_next)),
                                  float(np.max(q_next - wbc.upper)))
            q = wbc.integrate(qd, dt)
            settle = settle + 1 if ep < 0.01 else 0
            if settle >= 50 and converged is None:
                converged = (step - 49) * dt
        results[kind] = (ep, converged, worst_violation)
        t = np.array(times) * 1e3
        print(f"     {kind}: final err {ep * 1e3:7.2f} mm, converged "
              f"{'-' if converged is None else f'{converged:.2f} s'}, "
              f"unclamped limit overshoot {max(worst_violation, 0) * 1e3:.3f} mrad/mm, "
              f"solve mean {t.mean():.3f} ms / p99 {np.percentile(t, 99):.3f} ms")

    ep, conv, _ = results["wls"]
    check(ep < 0.01 and conv is not None, "WLS reaches the far target (<1 cm)")
    ep, conv, viol = results["wqp"]
    check(ep < 0.01 and conv is not None, "WQP reaches the far target (<1 cm)")
    check(viol <= 1e-9, "WQP never commands past a joint limit")


def _nullspace_hqp(levels, n):
    """Reference: recursive null-space projection (QP_WBC.md §8.5), eps -> 0."""
    v = np.zeros(n)
    N = np.eye(n)
    J_stack = np.zeros((0, n))
    for J, x in levels:
        v = v + np.linalg.pinv(J @ N) @ (x - J @ v)
        J_stack = np.vstack([J_stack, J])
        N = np.eye(n) - np.linalg.pinv(J_stack) @ J_stack
    return v


def test_hqp():
    print("[6] HQPSolver")
    rng = np.random.default_rng(0)

    # (a) one level == WQP
    prob = wf.WBCProblem(5, lambda_reg=1e-3)
    prob.add_task(wf.Task("a", rng.normal(size=(3, 5)), rng.normal(size=3), 2.0))
    prob.add_task(wf.Task("b", rng.normal(size=(2, 5)), rng.normal(size=2), 0.5))
    prob.set_bounds(-0.3, 0.3)
    r_w, r_h = wf.WQPSolver().solve(prob), wf.HQPSolver().solve(prob)
    check(r_h.ok and np.allclose(r_w.v, r_h.v, atol=1e-8),
          f"single level == WQP (diff {np.abs(r_w.v - r_h.v).max():.1e})")

    # (b) two conflicting levels: level 0 exact, level 1 best in what is left
    prob = wf.WBCProblem(2, lambda_reg=1e-9)
    prob.add_task(wf.Task("high", [[1.0, 1.0]], [1.0], priority=0))
    prob.add_task(wf.Task("low", np.eye(2), [1.0, 1.0], weight=100.0, priority=1))
    r = wf.HQPSolver().solve(prob)
    r_w = wf.WQPSolver().solve(prob)
    check(r.ok and np.allclose(r.v, [0.5, 0.5], atol=1e-6)
          and r.task_residuals["high"] < 1e-6,
          f"strict priority: v = {np.round(r.v, 6)}, high-level residual "
          f"{r.task_residuals['high']:.1e} (WQP with weight 100 on low: "
          f"{r_w.task_residuals['high']:.2f})")

    # (c) no constraints: equals recursive null-space projection
    n = 7
    J1, x1 = rng.normal(size=(2, n)), rng.normal(size=2)
    J2, x2 = rng.normal(size=(3, n)), rng.normal(size=3)
    J3, x3 = np.eye(n), rng.normal(size=n)
    prob = wf.WBCProblem(n, lambda_reg=1e-12)
    for k, (J, x) in enumerate([(J1, x1), (J2, x2), (J3, x3)]):
        prob.add_task(wf.Task(f"t{k}", J, x, priority=k))
    r = wf.HQPSolver().solve(prob)
    v_ref = _nullspace_hqp([(J1, x1), (J2, x2), (J3, x3)], n)
    check(r.ok and np.allclose(r.v, v_ref, atol=1e-5),
          f"3 levels == null-space projection (diff {np.abs(r.v - v_ref).max():.1e})")

    # (d) bounds active: level 0 still optimal under the bounds, bounds respected
    prob = wf.WBCProblem(3, lambda_reg=1e-9)
    prob.add_task(wf.Task("high", [[1.0, 1.0, 0.0]], [1.5], priority=0))
    prob.add_task(wf.Task("low", [[1.0, -1.0, 1.0]], [3.0], priority=1))
    prob.set_bounds(-1.0, 1.0)
    r = wf.HQPSolver().solve(prob)
    check(r.ok and np.all(np.abs(r.v) <= 1 + 1e-9) and r.task_residuals["high"] < 1e-6
          and abs(r.v[2] - 1.0) < 1e-6,
          f"bounds respected, high level exact, low level uses the rest: v = {np.round(r.v, 4)}")

    # (e) soft constraint as a level: minimal violation, then kept by lower levels
    prob = wf.WBCProblem(2, lambda_reg=1e-9)
    prob.add_constraint(wf.LinearConstraint("box", [[1.0, 0.0]], -np.inf, 0.2,
                                            soft=True, priority=0))
    prob.add_task(wf.Task("pull", [[1.0, 0.0], [0.0, 1.0]], [1.0, 0.5], priority=1))
    r = wf.HQPSolver().solve(prob)
    check(r.ok and abs(r.v[0] - 0.2) < 1e-6 and abs(r.v[1] - 0.5) < 1e-6
          and r.slack["box"][0] < 1e-9,
          f"soft constraint above task: feasible -> kept exactly (v = {np.round(r.v, 6)})")
    prob.set_bounds([0.5, -1], [1.0, 1])          # now the constraint cannot hold
    r = wf.HQPSolver().solve(prob)
    check(r.ok and abs(r.v[0] - 0.5) < 1e-6 and abs(r.slack["box"][0] - 0.3) < 1e-6,
          f"infeasible soft constraint: minimal violation s* = {r.slack['box'][0]:.4f} kept")

    # (f) linearly dependent locks (same rows in two levels) and a skipped level
    prob = wf.WBCProblem(3, lambda_reg=1e-9)
    J = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    prob.add_task(wf.Task("a", J, [0.1, 0.2], priority=0))
    prob.add_task(wf.Task("a_again", J, [0.3, 0.4], priority=1))
    prob.add_task(wf.Task("z", [[0.0, 0.0, 1.0]], [0.7], priority=2))
    prob.add_task(wf.Task("too_late", np.eye(3), [9.0, 9.0, 9.0], priority=3))
    r = wf.HQPSolver().solve(prob)
    check(r.ok and np.allclose(r.v, [0.1, 0.2, 0.7], atol=1e-6)
          and r.levels[1].n_locked == 2 and r.levels[3].skipped,
          f"dependent locks handled, no freedom left -> last level skipped: "
          f"{[(l.priority, l.status) for l in r.levels]}")

    # (f2) rows with zero weight are not part of the task, so they must not be locked
    prob = wf.WBCProblem(2, lambda_reg=1e-9)
    prob.add_task(wf.Task("high", np.eye(2), [1.0, 5.0], w_diag=[1.0, 0.0], priority=0))
    prob.add_task(wf.Task("low", [[0.0, 1.0]], [3.0], priority=1))
    r = wf.HQPSolver().solve(prob)
    check(r.ok and np.allclose(r.v, [1.0, 3.0], atol=1e-6),
          f"zero-weight rows of a high level are not locked: v = {np.round(r.v, 6)}")

    # (g) controller: hands above torso. The torso target pulls the hand away;
    # HQP keeps the hand, WQP compromises.
    from robot_model import BASE_JOINTS, TORSO_JOINTS, RIGHT_ARM_JOINTS, R1_FRAMES
    import warnings
    warnings.filterwarnings("ignore", message=".*'y'.*")   # torso y without base: expected
    hand_err = {}
    for kind in ("wqp", "hqp"):
        model, data, _, _ = _setup_robot()
        frames = {k: dict(v) for k, v in R1_FRAMES.items()}
        frames["torso"]["priority"] = 1
        wbc = wf.WholeBodyController(model, data, TORSO_JOINTS + RIGHT_ARM_JOINTS,
                                     "right_ee", base_free=False, solver=kind,
                                     frames=frames)
        dt = model.opt.timestep
        p_ee, q_ee = wbc.get_frame_pose("right_ee")
        p_t, q_t = wbc.get_frame_pose("torso")
        targets = {"right_ee": (p_ee, q_ee), "torso": (p_t + [0.3, 0.0, -0.3], q_t)}
        worst = 0.0
        for _ in range(1500):
            qd, err = wbc.compute(targets)
            wbc.integrate(qd, dt)
            worst = max(worst, err["right_ee"][0])
        hand_err[kind] = worst
        if kind == "hqp":
            check(wbc.last_result.ok and not wbc.last_result.degraded,
                  f"[hqp] controller solves every cycle (levels "
                  f"{[l.tasks for l in wbc.last_result.levels]})")
    check(hand_err["hqp"] < 1e-3 < hand_err["wqp"],
          f"unreachable torso target: hand drift HQP {hand_err['hqp'] * 1000:.3f} mm, "
          f"WQP {hand_err['wqp'] * 1000:.1f} mm")

    # (h) regularization="all": every level carries lambda + posture term.
    # One level: same as "last". Two levels: priority still holds (high level
    # only pays the small regularization residual).
    prob = wf.WBCProblem(4, lambda_reg=1e-4, wp_diag=np.full(4, 2e-3),
                         v_posture=[0.1, -0.1, 0.0, 0.2])
    prob.add_task(wf.Task("a", rng.normal(size=(2, 4)), rng.normal(size=2)))
    r_last = wf.HQPSolver().solve(prob)
    r_all = wf.HQPSolver(regularization="all").solve(prob)
    check(np.allclose(r_last.v, r_all.v, atol=1e-9), "regularization='all', one level == 'last'")
    prob = wf.WBCProblem(2, lambda_reg=1e-4, wp_diag=np.full(2, 2e-3))
    prob.add_task(wf.Task("high", [[1.0, 1.0]], [1.0], priority=0))
    prob.add_task(wf.Task("low", np.eye(2), [1.0, 1.0], weight=100.0, priority=1))
    r = wf.HQPSolver(regularization="all").solve(prob)
    check(r.ok and r.task_residuals["high"] < 3e-3,
          f"regularization='all' keeps priority: high-level residual "
          f"{r.task_residuals['high']:.1e} (vs 0.98 for WQP)")

    # (i) error clamp: desired velocity bounded, reported error is the real one
    model, data, _, _ = _setup_robot()
    wbc = wf.WholeBodyController(model, data, TORSO_JOINTS + RIGHT_ARM_JOINTS, "right_ee",
                                 base_free=False, solver="hqp")
    wbc.max_pos_error, wbc.max_rot_error = 0.02, 0.1
    p_ee, q_ee = wbc.get_frame_pose("right_ee")
    prob, err = wbc.build_problem({"right_ee": (p_ee + [0.5, 0, 0], q_ee)})
    xdot = prob.tasks[0].xdot_des
    check(abs(err["right_ee"][0] - 0.5) < 1e-9
          and abs(np.linalg.norm(xdot[:3]) - wbc.Kp_pos * 0.02) < 1e-9,
          f"error clamp: reported error {err['right_ee'][0]:.3f} m, "
          f"|xdot| {np.linalg.norm(xdot[:3]):.3f} m/s = Kp * 0.02")


def test_modes():
    print("[9] interactive modes: the nine modes and the drag state machine")
    import contextlib, io
    import robots
    import demo_hqp_interactive as D

    # every (mode, drag event) stays inside the eight modes; base stays frozen unless the base is dragged
    closed = all(ev(m) in D.MODES.values() for m in D.MODES.values()
                 for ev in (D.on_hand_target_event, D.on_torso_target_event, D.on_base_target_event))
    check(closed and len(set(D.MODES.values())) == 9, "drag events never leave the nine modes")
    check(D.Options().mode == 5 and D.MODES[7] == D.ControlMode(D.HAND_TRACK, D.TORSO_FREE, D.BASE_TRACK)
          and D.mode_number(D.on_base_target_event(D.MODES[5])) == 7,
          "default mode 5; mode 7 = hands track, torso free, base track; dragging the base enters it")
    stays = all(D.on_hand_target_event(D.MODES[n]).base == D.BASE_FROZEN
                and D.on_torso_target_event(D.MODES[n]).base == D.BASE_FROZEN for n in (1, 2, 3, 4))
    check(stays, "dragging hands or torso in modes 1~4 does not bring the base in")

    def build(mode):
        robot = robots.get_robot("r1")
        # kinematic: joints equal the command, so "frozen" can be checked exactly
        # (under dynamics the servos deflect slightly from the reaction of the moving parts)
        opt = D.Options(mode=mode, kinematic=True)
        m, d, sub = D.load_sim(robot, opt.control_dt)
        D.set_initial_posture(robot, m, d)
        c = D.InteractiveController(m, d, sub, opt, robot)
        c.initialize()
        return c

    moved = {}
    for n in D.MODES:
        with contextlib.redirect_stdout(io.StringIO()):
            c = build(n)
            q0 = c.read_q()
            base0 = c.base_pose()[0]
            if D.MODES[n].hands == D.HAND_TRACK:      # stay in this mode: move the targets directly
                dx = np.array([0.5, 0.0, 0.0])
                c.commit_hands(c.home(D.LEFT) + dx, c.parts[D.LEFT].seg.ori_goal,
                               c.home(D.RIGHT) + dx, c.parts[D.RIGHT].seg.ori_goal, c.time())
            elif D.MODES[n].base == D.BASE_TRACK:
                c.commit_base(c.home(D.BASE) + [0.3, 0.0, 0.0], 0.3, c.time())
            else:
                c.commit_torso(c.home(D.TORSO) + [0.0, 0.0, -0.05], c.time())
            end = c.time() + c.remaining_motion() + 1.0
            while c.time() < end:
                c.step()
        q = c.read_q() - q0
        names = c.joints
        idx = lambda group: [names.index(j) for j in group]
        moved[n] = tuple(float(np.abs(q[idx(g)]).max()) for g in
                         (c.robot.base_joints, c.robot.torso_joints, c.robot.arm_joints()))
        ok = c.fails == 0 and c.mode == D.MODES[n]
        base, torso, arm = moved[n]
        m = D.MODES[n]
        # frozen parts do not move at all; free parts help; the part with a new target moves
        for value, state, frozen, free in ((base, m.base, D.BASE_FROZEN, D.BASE_FREE),
                                           (torso, m.torso, D.TORSO_FROZEN, D.TORSO_FREE),
                                           (arm, m.hands, D.HAND_FROZEN, None)):
            ok = ok and (value < 1e-9 if state == frozen else True)
            ok = ok and (value > 1e-3 if state == free else True)
        ok = ok and (arm > 1e-3 if m.hands == D.HAND_TRACK else True)
        ok = ok and (base > 1e-3 if n == 9 else True) and (torso > 1e-3 if n == 3 else True)
        check(ok, f"mode {n} solves, frozen parts stay and others move: "
                  f"base {base:.3f} torso {torso:.3f} arms {arm:.3f}")


def test_target_round():
    print("[10] interactive modes: setting several targets in one round")
    import contextlib, io
    import robots
    import demo_hqp_interactive as D

    robot = robots.get_robot("r1")
    opt = D.Options(mode=8)
    m, d, sub = D.load_sim(robot, opt.control_dt)
    D.set_initial_posture(robot, m, d)
    with contextlib.redirect_stdout(io.StringIO()):
        c = D.InteractiveController(m, d, sub, opt, robot)
        c.initialize()
        hand0, torso0, base0 = c.home(D.LEFT).copy(), c.home(D.TORSO).copy(), c.home(D.BASE).copy()

        order = [c.round_next()]
        c.begin_drag(D.LEFT)
        c.drag_to(D.LEFT, hand0 + [0.1, 0.0, 0.0])
        c.commit_drag(D.LEFT)
        pending = np.allclose(c.parts[D.LEFT].seg.goal, hand0) and c.parts[D.LEFT].seg.duration == 0.0
        order.append(c.round_next())
        c.skip_group()                          # torso: keep
        order.append(c.round_next())
        c.begin_drag(D.BASE)
        c.drag_to(D.BASE, base0 + [0.1, 0.05, 0.0])
        c.drag_base_yaw(0.2)
        c.commit_drag(D.BASE)
        seg = c.parts
    check(order == ["hands", "torso", "base"] and pending,
          "modes with several TRACK: targets are set one by one and nothing moves before the last")
    check(seg[D.LEFT].seg.duration == seg[D.RIGHT].seg.duration == seg[D.BASE].seg.duration > 0
          and np.allclose(seg[D.TORSO].seg.goal, torso0) and c.round_next() == "hands"
          and D.mode_number(c.mode) == 8,
          "after the last target all go together; a skipped target keeps its goal; mode unchanged")

    opt = D.Options(mode=9)
    m, d, sub = D.load_sim(robot, opt.control_dt)
    D.set_initial_posture(robot, m, d)
    with contextlib.redirect_stdout(io.StringIO()):
        c = D.InteractiveController(m, d, sub, opt, robot)
        c.initialize()
        c.begin_drag(D.BASE)
        c.drag_base_yaw(0.5)
        c.commit_drag(D.BASE)
    check(c.parts[D.BASE].seg.duration > 0 and abs(c.parts[D.BASE].goal_yaw - 0.5) < 1e-12
          and np.allclose(c.parts[D.BASE].seg.goal, c.home(D.BASE)),
          "single TRACK: release executes; base yaw target set, position unchanged")


def test_torso_lateral():
    print("[11] torso y: only when the base is not frozen")
    import contextlib, io
    import robots
    import demo_hqp_interactive as D

    def run(mode):
        robot = robots.get_robot("r1")
        opt = D.Options(mode=mode)
        m, d, sub = D.load_sim(robot, opt.control_dt)
        D.set_initial_posture(robot, m, d)
        with contextlib.redirect_stdout(io.StringIO()):
            c = D.InteractiveController(m, d, sub, opt, robot)
            c.initialize()
            t0 = c.measured(D.TORSO).copy()
            c.begin_drag(D.TORSO)
            c.drag_to(D.TORSO, t0 + [0.0, 0.2, 0.0])
            sent = c.ghost(D.TORSO).copy()
            c.commit_drag(D.TORSO)
            while c.round_next() is not None and c.round_dragged:
                c.skip_group()
            end = c.time() + c.remaining_motion() + 2.0
            peak = 0.0
            while c.time() < end:
                c.step()
                peak = max(peak, c.errors[D.LEFT])
        return c, t0, sent, peak

    c, t0, sent, _ = run(3)
    check(abs(sent[1] - t0[1]) < 1e-9, "base frozen (mode 3): torso y target is not open")
    c, t0, sent, peak = run(6)
    check(abs(sent[1] - t0[1] - 0.2) < 1e-9 and abs(c.measured(D.TORSO)[1] - t0[1] - 0.2) < 2e-3
          and peak < 5e-3 and c.fails == 0,
          f"mode 6: torso y reached by the free base "
          f"(base y {c.base_pose()[0][1]:.3f} m), hands stay within {peak * 1000:.2f} mm")
    with contextlib.redirect_stdout(io.StringIO()):
        c.request_mode(D.mode_from_number(2))     # base frozen again: nothing to chase
        for _ in range(100):
            c.step()
    check(c.errors[D.TORSO] < 2e-3 and D.mode_number(c.mode) == 2,
          "back to a frozen base: torso target re-anchored")


def test_posture():
    print("[12] posture term: set_posture (q / w / k by group), v_posture only at the lowest HQP level")
    from robot_model import TORSO_JOINTS, RIGHT_ARM_JOINTS
    from wbc_function.hqp_solver import Level, LevelInfo
    model, data, joints, base = _setup_robot()
    wbc = wf.WholeBodyController(model, data, joints, "right_ee", solver="hqp")
    q = wbc.get_joint_positions()
    arm = [joints.index(n) for n in RIGHT_ARM_JOINTS]
    torso = [joints.index(n) for n in TORSO_JOINTS]
    base_i = [joints.index(n) for n in base]

    v0 = wbc._posture_velocity(q)
    wbc.posture_kl = 0.0
    wbc.set_posture(RIGHT_ARM_JOINTS, q=q[arm] + 0.1, k=0.5)
    wbc.set_posture(TORSO_JOINTS, w=0.05, k=0.0)
    v = wbc._posture_velocity(q)
    wp = wbc._posture_weights()
    check(np.allclose(v[arm], 0.05) and np.allclose(v[torso], 0.0) and np.allclose(v[base_i], 0.0),
          "q / k by group: arm pulled to its q_ref, torso k = 0, base default k = 0")
    check(np.allclose(wp[torso], 0.05) and np.allclose(np.delete(wp, torso), wbc.posture_w),
          "w by group: torso W_p set, others keep posture_w")
    wbc.set_posture(RIGHT_ARM_JOINTS, q=None)
    check(np.allclose(wbc._posture_velocity(q)[arm], 0.5 * (wbc.q_mid[arm] - q[arm])),
          "q=None restores the default (mid of the range); k is kept")
    wbc.set_posture(joints, q=None, w=None, k=None)
    wbc.posture_kl = wbc.config.posture_kl
    check(np.allclose(wbc._posture_velocity(q), v0), "all None: back to the defaults")

    wbc.set_posture(RIGHT_ARM_JOINTS, k=0.5, q=q[arm] + 0.1)
    pos0, quat0 = wbc.get_ee_pose()
    prob, _ = wbc.build_problem({"right_ee": (pos0 + [0.05, 0.0, 0.0], quat0)})
    hqp = wf.HQPSolver(regularization="all")
    task = prob.tasks[0]
    Ht, gt = prob.task_objective([task])
    lam = prob.lambda_reg * np.eye(prob.n) + np.diag(prob.wp_diag)
    info = LevelInfo(0)
    H, g = hqp._objective(prob, Level(0, [task], []), False, info)
    check(np.allclose(g, gt) and np.allclose(H, Ht + lam + info.eps * np.eye(prob.n)),
          "upper level: weight regularization only (lambda I + W_p), no v_posture")
    H, g = hqp._objective(prob, Level(0, [task], []), True, LevelInfo(0))
    check(np.allclose(g, gt - prob.wp_diag * prob.v_posture) and np.abs(prob.v_posture).max() > 0,
          "lowest level: full posture term with v_posture")


def test_config():
    print("[8] parameters: WBCConfig / SolverConfig, command line, config file")
    import json, tempfile, os
    import demo_hqp_interactive as demo
    model, data, joints, _ = _setup_robot()

    # defaults reproduce the previous hard-coded controller
    wbc = wf.WholeBodyController(model, data, joints, "right_ee")
    check(wbc.Kp_pos == 12.0 and wbc.posture_k == 2e-3 and wbc.posture_kl == 0.5
          and np.allclose(wbc.task_w, [1, 1, 1, 0.4, 0.4, 0.4]) and wbc.max_pos_error is None,
          "default WBCConfig == previous controller defaults")

    # config reaches the controller and the solver
    cfg = wf.WBCConfig(kp_pos=20.0, rot_weight=0.2, limit_zone=0.05)
    cfg.solver.eps, cfg.solver.max_iter, cfg.solver.regularization = 1e-5, 200, "last"
    wbc = wf.WholeBodyController(model, data, joints, "right_ee", solver="hqp", config=cfg)
    check(wbc.Kp_pos == 20.0 and wbc.task_w[3] == 0.2 and wbc.limit_zone == 0.05
          and wbc.solver.eps == 1e-5 and wbc.solver.regularization == "last"
          and wbc.solver._qp.max_iter == 200,
          "WBCConfig reaches controller and HQPSolver")

    # command line > config file > defaults; unknown names are rejected
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"kp_pos": 20, "torso_cost": 1.0, "wbc": {"solver": {"max_iter": 50}}}, f)
    try:
        _, opt = demo.parse_args(["--config", f.name, "--kp-pos", "30", "--eps", "1e-5"])
    finally:
        os.remove(f.name)
    check(opt.wbc.kp_pos == 30 and opt.torso_cost == 1.0 and opt.wbc.solver.max_iter == 50
          and opt.wbc.solver.eps == 1e-5 and opt.wbc.posture_k == 0.0,
          "command line overrides config file, file overrides defaults")
    try:
        demo.parse_args(["--config", "/dev/null", "--no-such-param", "1"])
        check(False, "unknown parameter should be rejected")
    except SystemExit:
        check(True, "unknown parameter is rejected")


def test_robots():
    print("[7] robot configs: generic URDF loader, JSON config, interactive demo on R1")
    import contextlib, io, json, os, re, tempfile
    import robots
    import demo_hqp_interactive as demo

    # same URDF through the R1 converter and through the generic loader
    poses = {}
    for name in ("r1", "r1_urdf"):
        cfg = robots.get_robot(name)
        m, d = cfg.load()
        robots._set_home(m, d, cfg.home)
        poses[name] = (d.site_xpos[m.site(cfg.right_ee_site).id].copy(),
                       d.xpos[m.body(cfg.torso_body).id].copy(),
                       m.actuator_gainprm[:, 0].copy(), m.ngravcomp)
    a, b = poses["r1"], poses["r1_urdf"]
    check(np.allclose(a[0][:2], b[0][:2], atol=1e-6) and np.allclose(a[1][:2], b[1][:2], atol=1e-6)
          and np.allclose(a[2], b[2], rtol=1e-4) and a[3] > 0 and b[3] > 0,
          "generic loader == R1 converter (frames x/y, servo gains), gravity compensation compiled in")

    # JSON config for an arbitrary URDF
    import robot_model as rm
    cfg = {"name": "r1_json", "urdf": rm.URDF_PATH, "planar_base": True,
           "torso_joints": rm.TORSO_JOINTS, "left_arm_joints": rm.LEFT_ARM_JOINTS,
           "right_arm_joints": rm.RIGHT_ARM_JOINTS,
           "ee_sites": {"left_ee": "left_gripper_center", "right_ee": "right_gripper_center"},
           "left_ee_site": "left_ee", "right_ee_site": "right_ee",
           "torso_body": rm.TORSO_BODY, "base_body": rm.BASE_BODY, "home": rm.HOME_POSE}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(cfg, f)
    try:
        robot = robots.get_robot(f.name)
        m, d = robot.load()
        check(m.nu == 21 and robot.base_joints == ["base_x", "base_y", "base_yaw"],
              f"JSON config loads (nu = {m.nu}, base joints {robot.base_joints})")
    finally:
        os.remove(f.name)

    # interactive demo self-check on R1 (dynamics, reference command mode)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        ret = demo.main(["--demo", "--robot", "r1"])
    text = out.getvalue()
    b_line = next(l for l in text.splitlines() if l.startswith("B "))
    b_err = float(b_line.split("末端剩余误差 ")[1].split(" mm")[0])
    check(ret == 0 and b_err < 1.0, f"demo_hqp_interactive --demo on R1: exit {ret}, B {b_err:.4f} mm")
    print("        " + "\n        ".join(l for l in text.splitlines() if l[:2] in ("E ", "F ", "G ", "H ")))
    nums = lambda head: [float(x) for x in re.findall(r"-?\d+\.\d+", next(
        l for l in text.splitlines() if l.startswith(head)))]
    e, f, g, h = (nums(x) for x in ("E ", "F ", "G ", "H "))
    check(e[-2] < 0.5 and e[-1] < 5.0, "mode 8: base (lowest level) yields, hands stay")
    check(abs(f[-2] - 5.0) < 2e-3, "base yaw goes past +/-pi")
    check(g[-2] < 1.0 and g[-1] < 5.0, "after base yaw 90 deg, torso still tracks forward / height")
    check(h[-2] < 1.0 and h[-1] > 1.0, "free base helps the hands reach further")


if __name__ == "__main__":
    test_qp_basic()
    test_soft()
    test_wls_saturation()
    test_controller_closed_form()
    test_target_forms()
    test_frames()
    test_resident_tasks()
    test_closed_loop()
    test_hqp()
    test_robots()
    test_config()
    test_modes()
    test_target_round()
    test_torso_lateral()
    test_posture()
    print()
    if FAILS:
        print(f"{len(FAILS)} check(s) failed")
        sys.exit(1)
    print("all checks passed")
