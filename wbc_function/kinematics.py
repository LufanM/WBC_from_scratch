"""
kinematics.py —— 从 MuJoCo 模型中取出受控自由度的 q、限位、雅可比和位姿。

受控自由度的顺序只在这里定义一次（joint_names 的顺序），所有任务的 J、
约束的 A、上下界都按这个顺序排列。
"""
from __future__ import annotations

import numpy as np

from .tasks import pose_task


class MujocoKinematics:
    """MuJoCo 模型的受控自由度视图。

    model, data    MuJoCo 的 MjModel、MjData
    joint_names    受控关节名（必须是单自由度的 hinge / slide 关节）
    default_range  没有限位的关节（如 base_yaw）使用的范围 [-r, r]，默认很大，相当于不限
    """

    def __init__(self, model, data, joint_names, default_range=1e3):
        import mujoco
        self._mj = mujoco
        self.m, self.d = model, data
        self.joint_names = list(joint_names)
        self.n = len(self.joint_names)

        qadr, dofadr, lower, upper = [], [], [], []
        single_dof = (int(mujoco.mjtJoint.mjJNT_HINGE),
                      int(mujoco.mjtJoint.mjJNT_SLIDE))
        for name in self.joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise KeyError(f"模型中没有关节 {name!r}")
            # 按整数比较：新版 mujoco 中 `np.int32 in (枚举, ...)` 会返回 False
            if int(model.jnt_type[jid]) not in single_dof:
                raise ValueError(f"关节 {name!r} 不是单自由度关节")
            qadr.append(int(model.jnt_qposadr[jid]))
            dofadr.append(int(model.jnt_dofadr[jid]))
            lo, hi = model.jnt_range[jid]
            if not model.jnt_limited[jid] or lo >= hi:
                lo, hi = -default_range, default_range
            lower.append(lo)
            upper.append(hi)

        self.qadr = np.array(qadr)        # 在 data.qpos 中的下标
        self.dofadr = np.array(dofadr)    # 在雅可比列（data.qvel）中的下标
        self.q_min = np.array(lower, float)
        self.q_max = np.array(upper, float)

    def mask(self, names):
        """给定关节名 → 受控自由度上的布尔 mask（用于冻结、区分底盘和关节等）。"""
        names = set(names)
        return np.array([n in names for n in self.joint_names])

    def q(self):
        """当前受控自由度的位置。"""
        return self.d.qpos[self.qadr].copy()

    def set_q(self, q):
        """写入受控自由度的位置，并刷新运动学。"""
        self.d.qpos[self.qadr] = q
        self._mj.mj_forward(self.m, self.d)

    def _site_id(self, site):
        if isinstance(site, str):
            sid = self._mj.mj_name2id(self.m, self._mj.mjtObj.mjOBJ_SITE, site)
            if sid < 0:
                raise KeyError(f"模型中没有 site {site!r}")
            return sid
        return int(site)

    def site_pose(self, site):
        """site 在世界坐标系中的 (位置, 四元数 [w,x,y,z])。"""
        sid = self._site_id(site)
        quat = np.empty(4)
        self._mj.mju_mat2Quat(quat, self.d.site_xmat[sid])
        return self.d.site_xpos[sid].copy(), quat

    def site_jacobian(self, site):
        """site 的 6 x n 世界坐标系雅可比（前 3 行线速度，后 3 行角速度）。"""
        jacp = np.zeros((3, self.m.nv))
        jacr = np.zeros((3, self.m.nv))
        self._mj.mj_jacSite(self.m, self.d, jacp, jacr, self._site_id(site))
        return np.vstack([jacp[:, self.dofadr], jacr[:, self.dofadr]])

    def _body_id(self, body):
        bid = self._mj.mj_name2id(self.m, self._mj.mjtObj.mjOBJ_BODY, body)
        if bid < 0:
            raise KeyError(f"模型中没有 body {body!r}")
        return bid

    def body_pose(self, body):
        """body 原点在世界坐标系中的 (位置, 四元数 [w,x,y,z])。"""
        bid = self._body_id(body)
        return self.d.xpos[bid].copy(), self.d.xquat[bid].copy()

    def body_jacobian(self, body):
        """body 原点的 6 x n 世界坐标系雅可比。"""
        bid = self._body_id(body)
        jacp = np.zeros((3, self.m.nv))
        jacr = np.zeros((3, self.m.nv))
        self._mj.mj_jacBody(self.m, self.d, jacp, jacr, bid)
        return np.vstack([jacp[:, self.dofadr], jacr[:, self.dofadr]])


def site_pose_task(kin: MujocoKinematics, site, target_pos, target_quat,
                   name=None, **kwargs):
    """在当前状态下，为 MuJoCo 的某个 site 构造 pose_task。

    其余参数（kp_pos、kp_rot、w_rot、xdot_ff 等）原样传给 pose_task。
    """
    pos, quat = kin.site_pose(site)
    return pose_task(name or str(site), kin.site_jacobian(site), pos, quat,
                     target_pos, target_quat, **kwargs)
