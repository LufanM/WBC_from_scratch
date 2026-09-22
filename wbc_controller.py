"""
wbc_controller.py — QP-based Whole-Body Controller for the Galaxea R1 Pro
wheeled dual-arm humanoid, on MuJoCo.

Goal
----
Drive one or both end-effectors (EE) to target poses in the world by jointly
planning the *mobile base* (x, y, yaw), the *torso* (4 DOF), and the *arms*
(7 DOF each). This is a resolved-rate / velocity whole-body controller: each
control tick solves a small QP for the whole-body velocity that best drives the
EEs toward their targets while respecting joint limits and staying near a
comfortable posture.

Formulation
-----------
Decision variable:  v = q_dot over the controlled DOFs.

Unlike the earlier PyBullet version, the base is *not* a special case: the
MJCF built by ``convert_urdf.py`` exposes ``base_x``, ``base_y`` and
``base_yaw`` as ordinary joints, so MuJoCo's own site Jacobian already contains
the base columns. There is no hand-derived base Jacobian to get wrong.

Primary task (EE 6-D pose, one block per EE):   J v = xdot_des
    xdot_des = K * pose_error(target, current)
    J = mj_jacSite(...) restricted to the controlled DOF columns.

We solve a weighted, regularized least-squares (a strictly-convex QP with a
closed-form solution):

    min_v  || J v - xdot_des ||^2_W
           + lambda_reg || v ||^2               (damping / singularity-robust)
           + || v - v_posture ||^2_Wp           (secondary posture task)

    v_posture pulls joints toward a nominal comfortable configuration and
    pushes them away from their limits (soft joint-limit avoidance).

Joint-limit handling: the integrated position is clamped, and a repulsive
posture velocity grows near the limits so the QP naturally avoids them.

Quaternion convention throughout is MuJoCo's: [w, x, y, z].
"""
from __future__ import annotations

import numpy as np
import mujoco


# ------------------------------------------------------------------ utilities
def quat_error(q_cur, q_des):
    """Orientation error as a world-frame rotation vector (3,).

    Returns the rotation vector w such that rotating ``q_cur`` by ``w`` yields
    ``q_des``. Uses the error quaternion q_err = q_des * q_cur^-1, taking the
    shortest path. Quaternions are [w, x, y, z].
    """
    qc = np.asarray(q_cur, float)
    qd = np.asarray(q_des, float)
    qc_inv = np.empty(4)
    mujoco.mju_negQuat(qc_inv, qc)
    qe = np.empty(4)
    mujoco.mju_mulQuat(qe, qd, qc_inv)
    if qe[0] < 0.0:  # shortest path
        qe = -qe
    v = qe[1:]
    n = np.linalg.norm(v)
    if n < 1e-9:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(n, qe[0])
    return (angle / n) * v


def rpy_to_quat(rpy):
    """Roll-pitch-yaw (radians, XYZ fixed-axis) -> quaternion [w, x, y, z]."""
    r, p, y = (float(v) for v in rpy)
    cr, sr = np.cos(r / 2), np.sin(r / 2)
    cp, sp = np.cos(p / 2), np.sin(p / 2)
    cy, sy = np.cos(y / 2), np.sin(y / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


class WholeBodyController:
    """Velocity-resolved whole-body controller on a MuJoCo model.

    Parameters
    ----------
    model, data : mujoco.MjModel, mujoco.MjData
        The compiled scene and its state. The controller reads state from
        ``data`` and never steps physics itself.
    joint_names : list[str]
        Actuated joints to control, in control order. Must be single-DOF
        (hinge / slide) joints — the R1 Pro base chain and arms all are.
    ee_sites : str | list[str]
        Name(s) of the site(s) whose pose we track. One for single-arm use,
        two for dual-arm.
    base_free : bool
        If True, the base joints participate in the solve. If False they are
        held (their columns are dropped from the QP), giving a torso+arms-only
        controller.
    base_joints : list[str]
        Which of ``joint_names`` are base DOFs. Used only to apply base-
        specific weights and to implement ``base_free=False``.
    """

    def __init__(self, model, data, joint_names, ee_sites, base_free=True,
                 base_joints=("base_x", "base_y", "base_yaw")):
        self.m = model
        self.d = data
        self.base_free = base_free

        if isinstance(ee_sites, str):
            self.ee_sites = [ee_sites]
        else:
            self.ee_sites = list(ee_sites)
        self.ee_site = self.ee_sites[0]  # convenience for single-arm use
        self.site_id = [self._site_id(s) for s in self.ee_sites]

        base_set = set(base_joints)
        # Drop base DOFs entirely when the base is held: they then never
        # appear in the QP, so no weight tuning can accidentally move them.
        names = [n for n in joint_names
                 if base_free or n not in base_set]
        self.joint_names = names
        self.n = len(names)

        self.jnt_id, self.qadr, self.dofadr = [], [], []
        for name in names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise KeyError(f"joint {name!r} not in model")
            jtype = model.jnt_type[jid]
            if jtype not in (mujoco.mjtJoint.mjJNT_HINGE,
                             mujoco.mjtJoint.mjJNT_SLIDE):
                raise ValueError(
                    f"joint {name!r} is not single-DOF (type {jtype}); the "
                    "controller assumes one qpos/dof per controlled joint")
            self.jnt_id.append(jid)
            self.qadr.append(int(model.jnt_qposadr[jid]))
            self.dofadr.append(int(model.jnt_dofadr[jid]))
        self.qadr = np.array(self.qadr, dtype=int)
        self.dofadr = np.array(self.dofadr, dtype=int)
        self.base_mask = np.array([n in base_set for n in names])

        # Joint limits. Unlimited joints (base_yaw, wheels) get a wide but
        # finite range so the posture term stays well-defined.
        lower, upper = [], []
        for jid in self.jnt_id:
            lo, hi = model.jnt_range[jid]
            if not model.jnt_limited[jid] or lo >= hi:
                lo, hi = -np.pi, np.pi
            lower.append(lo)
            upper.append(hi)
        self.lower = np.array(lower)
        self.upper = np.array(upper)
        # Nominal posture = mid-range, except base DOFs, whose "comfortable"
        # value is wherever they happen to be (mid-range would drag the robot
        # back to the origin).
        self.q_nominal = 0.5 * (self.lower + self.upper)
        self.q_nominal[self.base_mask] = 0.0
        self.posture_mask = ~self.base_mask

        # gains / weights
        self.Kp_pos = 12.0       # EE position P-gain
        self.Kp_rot = 12.0       # EE orientation P-gain
        self.task_w = np.array([1, 1, 1, 0.4, 0.4, 0.4])  # xyz weighted > rpy
        self.lambda_reg = 1e-4   # damping (singularity robustness)
        self.posture_w = 2e-3    # secondary posture pull (joints only)
        self.base_reg_w = 1e-4   # base regularization, kept tiny so the base
                                 # is used freely for far targets
        self.limit_w = 0.5       # joint-limit repulsion strength
        self.max_joint_vel = 2.0
        self.max_base_vel = 0.6

    def _site_id(self, name):
        sid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid < 0:
            raise KeyError(f"site {name!r} not in model")
        return sid

    # ---------------------------------------------------------- state readers
    def get_joint_positions(self):
        return self.d.qpos[self.qadr].copy()

    def set_joint_positions(self, q):
        """Write controlled-joint positions and refresh kinematics."""
        self.d.qpos[self.qadr] = q
        mujoco.mj_forward(self.m, self.d)

    def get_base_pose(self):
        """(x, y, yaw) of the mobile base, read from the base joints."""
        vals = {}
        for name in ("base_x", "base_y", "base_yaw"):
            jid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, name)
            vals[name] = (float(self.d.qpos[self.m.jnt_qposadr[jid]])
                          if jid >= 0 else 0.0)
        return np.array([vals["base_x"], vals["base_y"], vals["base_yaw"]])

    def get_ee_pose(self, which=0):
        """World (position, quaternion) of an EE site.

        ``which`` is an index into ``ee_sites`` (or the site name itself).
        MuJoCo stores site orientation as a 3x3 matrix; we convert to a
        [w, x, y, z] quaternion for a uniform interface.
        """
        sid = self._site_id(which) if isinstance(which, str) \
            else self.site_id[which]
        pos = self.d.site_xpos[sid].copy()
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, self.d.site_xmat[sid])
        return pos, quat

    # ------------------------------------------------------------- jacobians
    def site_jacobian(self, which=0):
        """6 x n site Jacobian (world frame) over the controlled DOFs.

        ``mj_jacSite`` returns 3 x nv translational and rotational blocks over
        *all* DOFs; we select the columns of the joints we control. Because the
        base is a real joint chain in the model, its columns come from MuJoCo
        too — the whole-body Jacobian needs no analytic base term.
        """
        sid = self._site_id(which) if isinstance(which, str) \
            else self.site_id[which]
        jacp = np.zeros((3, self.m.nv))
        jacr = np.zeros((3, self.m.nv))
        mujoco.mj_jacSite(self.m, self.d, jacp, jacr, sid)
        return np.vstack([jacp[:, self.dofadr], jacr[:, self.dofadr]])

    def body_jacobian(self, body_name):
        """3 x n world-frame position Jacobian of a body origin.

        Same column-selection convention as ``site_jacobian``; used for
        picking out intermediate link points (e.g. shoulder/elbow) that have
        no site defined on them.
        """
        bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            raise KeyError(f"body {body_name!r} not in model")
        jacp = np.zeros((3, self.m.nv))
        jacr = np.zeros((3, self.m.nv))
        mujoco.mj_jacBody(self.m, self.d, jacp, jacr, bid)
        return jacp[:, self.dofadr]

    def check_jacobian(self, eps=1e-6):
        """Finite-difference validation of the site Jacobian.

        Perturbs each controlled joint by ``eps``, measures the actual site
        pose change via FK, and compares against the Jacobian column. Returns
        the max absolute discrepancy (expect ~1e-6 or below). Restores state.

        This is the verification gate: the column selection (dofadr indexing)
        and the world-frame convention are the classic silent-bug sources, and
        every result downstream is built on top of them.
        """
        q0 = self.get_joint_positions()
        max_err = 0.0
        for k in range(len(self.site_id)):
            mujoco.mj_forward(self.m, self.d)
            J = self.site_jacobian(k)
            pos0, quat0 = self.get_ee_pose(k)
            for c in range(self.n):
                q = q0.copy()
                q[c] += eps
                self.set_joint_positions(q)
                pos1, quat1 = self.get_ee_pose(k)
                measured = np.concatenate([(pos1 - pos0) / eps,
                                           quat_error(quat0, quat1) / eps])
                max_err = max(max_err,
                              float(np.max(np.abs(measured - J[:, c]))))
            self.set_joint_positions(q0)
        return max_err

    # ------------------------------------------------------------- posture
    def _posture_velocity(self, q):
        """Secondary velocity: pull to nominal + repel from joint limits.

        Base DOFs are excluded from the nominal pull (they have no preferred
        world position) but still respect their travel limits.
        """
        v = self.posture_w * (self.q_nominal - q) * self.posture_mask
        # limit repulsion: grows sharply within 10% of range near a bound
        rng = self.upper - self.lower
        margin = 0.1 * rng + 1e-6
        near_lo = q - self.lower
        near_hi = self.upper - q
        v += self.limit_w * np.where(near_lo < margin,
                                     (margin - near_lo) / margin, 0.0)
        v -= self.limit_w * np.where(near_hi < margin,
                                     (margin - near_hi) / margin, 0.0)
        return v

    # ------------------------------------------------------------- main step
    def compute(self, target_pos, target_quat):
        """One whole-body velocity step.

        Single-arm usage:
            target_pos : (3,)   target_quat : (4,) [w, x, y, z]
            returns (q_dot[n], err_pos, err_ang)                (scalars)

        Dual/multi-arm usage (targets as lists, in ``ee_sites`` order):
            returns (q_dot[n], err_pos_list, err_ang_list)

        ``q_dot`` is indexed like ``joint_names``; use ``dofadr`` / ``qadr`` to
        apply it to MuJoCo state or actuators.
        """
        multi = isinstance(target_pos, (list, tuple))
        tps = list(target_pos) if multi else [target_pos]
        tqs = list(target_quat) if multi else [target_quat]
        assert len(tps) == len(self.site_id) == len(tqs), (
            "number of targets must match number of end-effectors")

        q = self.get_joint_positions()

        J_rows, xdot_rows, W_diag = [], [], []
        err_pos, err_ang = [], []
        for k, (tp, tq) in enumerate(zip(tps, tqs)):
            pos, quat = self.get_ee_pose(k)
            e_pos = np.asarray(tp, float) - pos
            e_rot = quat_error(quat, np.asarray(tq, float))
            J_rows.append(self.site_jacobian(k))
            xdot_rows.append(np.concatenate([self.Kp_pos * e_pos,
                                             self.Kp_rot * e_rot]))
            W_diag.append(self.task_w)
            err_pos.append(float(np.linalg.norm(e_pos)))
            err_ang.append(float(np.linalg.norm(e_rot)))

        J = np.vstack(J_rows)                 # (6*m) x n
        xdot_des = np.concatenate(xdot_rows)  # (6*m,)

        v_post = self._posture_velocity(q)

        # --- weighted damped least squares (closed-form QP solution)
        W = np.diag(np.concatenate(W_diag))
        reg = np.full(self.n, self.posture_w)
        reg[self.base_mask] = self.base_reg_w
        Wp = np.diag(reg)
        A = J.T @ W @ J + self.lambda_reg * np.eye(self.n) + Wp
        b = J.T @ W @ xdot_des + Wp @ v_post
        qd = np.linalg.solve(A, b)

        # --- saturate. Base translation gets its own (lower) bound, scaled as
        # a 2-D vector so the drive direction is preserved; everything else,
        # base_yaw included, is clipped per-DOF.
        qd = np.clip(qd, -self.max_joint_vel, self.max_joint_vel)
        if self.base_free:
            xy = np.array([n in ("base_x", "base_y")
                           for n in self.joint_names])
            nb = np.linalg.norm(qd[xy])
            if nb > self.max_base_vel:
                qd[xy] *= self.max_base_vel / nb

        if multi:
            return qd, err_pos, err_ang
        return qd, err_pos[0], err_ang[0]

    # -------------------------------------------------------------- integrate
    def integrate(self, qd, dt):
        """Kinematically apply a velocity step, clamped to joint limits.

        This is the resolved-rate path: it validates the controller math
        without actuator dynamics. Returns the new joint positions.
        """
        q_new = np.clip(self.get_joint_positions() + qd * dt,
                        self.lower, self.upper)
        self.set_joint_positions(q_new)
        return q_new
