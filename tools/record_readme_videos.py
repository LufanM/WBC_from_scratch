"""
record_readme_videos.py —— 生成 README 中的两个动图（动画 WebP，保存到 README_pics/）。

    python3 tools/record_readme_videos.py            # 两个都生成
    python3 tools/record_readme_videos.py compare    # 只生成 WQP / HQP 对比
    python3 tools/record_readme_videos.py interactive
    python3 tools/record_readme_videos.py arm_angle

compare      R1 双手固定、躯干目标拉到够不着的地方，左右并排对比 WQP 和 HQP
             （运动学积分，两边的目标、增益完全相同）
interactive  demo_hqp_interactive.py 的真实界面（隐藏窗口），按脚本依次演示：
             双手前伸（全身协同）→ 拖躯干（双手不动）→ 冻结手臂只动躯干 → 压力点 → 回初始姿态
arm_angle    运行 demo_arm_angle_swivel.py 本身（替换掉它的 viewer），双手固定、臂角在零空间中摆动

需要显示器（GLFW 创建 OpenGL 上下文）。
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT_DIR = os.path.join(ROOT, "README_pics")


def _font(size):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _save_webp(frames, path, fps, quality=88):
    """保存为动画 WebP：真彩色（小的彩色目标球不会被调色板量化掉），体积比 GIF 小得多。"""
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps),
                   loop=0, quality=quality, method=6, background=(0, 0, 0, 0))
    print(f"[video] {path}: {len(frames)} frames, {os.path.getsize(path) / 1e6:.1f} MB")


def _add_sphere(scene, pos, radius, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, radius),
                        np.asarray(pos, float), np.eye(3).ravel(), np.asarray(rgba, np.float32))
    g.category = mujoco.mjtCatBit.mjCAT_DECOR
    scene.ngeom += 1


# =============================================================================
# 1. WQP / HQP 对比
# =============================================================================
def record_compare(path=os.path.join(OUT_DIR, "hqp_vs_wqp.webp")):
    import wbc_function as wf
    from robot_model import LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS, TORSO_JOINTS, load_model
    from robots import r1_interactive_home
    warnings.simplefilter("ignore")      # 底盘不动时躯干 y 不可控的提示

    W, H, FPS, DT = 640, 560, 15, 0.01
    MOVE_T, HOLD_T = 4.0, 2.0
    TORSO_DELTA = np.array([0.4, 0.0, -0.4])           # 躯干目标：前 0.4 m、下 0.4 m（够不着）

    sims = []
    for solver in ("wqp", "hqp"):
        model, data, _ = load_model()
        for n, v in r1_interactive_home().items():          # 与交互演示相同的初始姿态
            data.qpos[model.joint(n).qposadr[0]] = v
        mujoco.mj_forward(model, data)
        wbc = wf.WholeBodyController(
            model, data, TORSO_JOINTS + LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS,
            ["left_ee", "right_ee"], base_free=False, solver=solver, dt=DT,
            frames={"torso": {"body": "torso_link4", "components": ("x", "z", "ry"),
                              "priority": 1}})
        hands = {s: wbc.get_frame_pose(s) for s in ("left_ee", "right_ee")}
        torso0, torso_q = wbc.get_frame_pose("torso")
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, W)   # 离屏缓冲区够大
        model.vis.global_.offheight = max(model.vis.global_.offheight, H)
        renderer = mujoco.Renderer(model, H, W)
        sims.append(dict(solver=solver, model=model, data=data, wbc=wbc, hands=hands,
                         torso0=torso0, torso_q=torso_q, renderer=renderer, worst=0.0))

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = sims[0]["torso0"] + [0.3, 0.0, -0.2]
    cam.distance, cam.azimuth, cam.elevation = 2.7, 125.0, -12.0

    big, small = _font(30), _font(22)
    frames = []
    steps = int((MOVE_T + HOLD_T) / DT)
    every = int(round(1.0 / (FPS * DT)))
    for k in range(steps):
        s = min(1.0, (k + 1) * DT / MOVE_T)
        s = s * s * s * (10 - 15 * s + 6 * s * s)          # 最小 jerk
        panels = []
        for sim in sims:
            w = sim["wbc"]
            torso_target = sim["torso0"] + s * TORSO_DELTA
            for name, (p, q) in sim["hands"].items():
                w.task(name).set_target(p, q)
                w.task(name).enabled = True
            w.task("torso").set_target(torso_target, sim["torso_q"])
            w.task("torso").enabled = True
            qd, err = w.step()
            w.integrate(qd, DT)
            hand_err = max(err["left_ee"][0], err["right_ee"][0])
            sim["worst"] = max(sim["worst"], hand_err)
            if k % every:
                continue
            r = sim["renderer"]
            r.update_scene(sim["data"], camera=cam)
            for p, _ in sim["hands"].values():
                _add_sphere(r.scene, p, 0.03, (0.1, 0.85, 0.2, 0.9))       # 手的目标（绿）
            _add_sphere(r.scene, torso_target, 0.04, (1.0, 0.55, 0.1, 0.9))  # 躯干目标（橙）
            img = Image.fromarray(r.render())
            d = ImageDraw.Draw(img)
            title = ("WQP  (weighted)" if sim["solver"] == "wqp"
                     else "HQP  (hands > torso)")
            d.rectangle([0, 0, W, 112], fill=(20, 20, 20))
            d.text((16, 10), title, font=big, fill=(255, 255, 255))
            color = (255, 120, 90) if hand_err > 0.01 else (120, 230, 120)
            d.text((16, 50), f"hand error  {hand_err * 1000:6.1f} mm", font=small, fill=color)
            d.text((16, 78), f"torso error {err['torso'][0] * 1000:6.1f} mm", font=small,
                   fill=(230, 230, 230))
            panels.append(img)
        if panels:
            frame = Image.new("RGB", (2 * W + 6, H), (255, 255, 255))
            frame.paste(panels[0], (0, 0))
            frame.paste(panels[1], (W + 6, 0))
            frames.append(frame)
    for sim in sims:
        print(f"[compare] {sim['solver']}: worst hand error {sim['worst'] * 1000:.1f} mm")
        sim["renderer"].close()
    _save_webp(frames, path, FPS)


# =============================================================================
# 2. 交互仿真
# =============================================================================
def record_interactive(path=os.path.join(OUT_DIR, "hqp_interactive.webp")):
    import glfw
    import demo_hqp_interactive as D
    import robots

    FPS = 10
    robot = robots.get_robot("r1")
    opt = D.Options()
    model, data, substeps = D.load_sim(robot, opt.control_dt)
    D.set_initial_posture(robot, model, data)
    ctl = D.InteractiveController(model, data, substeps, opt, robot)
    ctl.initialize()
    viewer = D.Viewer(ctl)
    viewer.show_help = False
    viewer.robot_transparent = True       # 机身 40% 透明（T 键），被挡住的目标球也能看见

    def drag(part, delta, vertical, duration, t0, now):
        """模拟 Ctrl+拖动：按下、目标点在 duration 内移动 delta、松开。"""
        if viewer.dragging < 0:
            viewer.ctl.begin_drag(part)
            viewer.dragging, viewer.drag_vertical = part, vertical
            viewer._drag_start = viewer.ctl.ghost(part).copy()
        a = min(1.0, (now - t0) / duration)
        viewer.ctl.drag_to(part, viewer._drag_start + a * np.asarray(delta))
        if a >= 1.0:
            viewer.ctl.commit_drag(part)
            viewer.dragging = -1
            return True
        return False

    # 脚本：(开始时间, 动作)。动作返回 True 表示完成。
    home_t = ctl.home(D.TORSO)
    script = [
        (1.0, lambda t0, now: (ctl.place_hands_forward(), True)[1]),          # P：双手前伸
        (6.5, lambda t0, now: drag(D.TORSO, [-0.12, 0, 0], False, 1.0, t0, now)),  # 拖躯干后移
        (10.0, lambda t0, now: drag(D.TORSO, [0, 0, -0.15], True, 1.0, t0, now)),  # 拖躯干下降
        (14.0, lambda t0, now: (ctl.request_mode(D.mode_from_number(3)), True)[1]),  # 3：冻结手臂
        (15.0, lambda t0, now: drag(D.TORSO, [0, 0, 0.12], True, 1.0, t0, now)),   # 只动躯干
        (19.0, lambda t0, now: (ctl.go_to_stress_point(), True)[1]),          # B：压力点
        (27.0, lambda t0, now: (ctl.return_home(), True)[1]),                 # R：回初始姿态
    ]
    end_time = 33.0

    frames = []
    state = {"i": 0, "t0": None, "next_frame": 0.0}
    orig_render = viewer.render

    def render():
        orig_render()
        now = ctl.time()
        i = state["i"]
        if i < len(script) and now >= script[i][0]:
            if state["t0"] is None:
                state["t0"] = now
            if script[i][1](state["t0"], now):
                state["i"], state["t0"] = i + 1, None
        if now >= state["next_frame"]:
            state["next_frame"] += 1.0 / FPS
            fw, fh = glfw.get_framebuffer_size(viewer.window)
            rgb = np.zeros((fh, fw, 3), np.uint8)
            mujoco.mjr_readPixels(rgb, None, mujoco.MjrRect(0, 0, fw, fh), viewer.con)
            img = Image.fromarray(np.flipud(rgb))
            frames.append(img)                       # 保留窗口原始分辨率
        if now >= end_time:
            glfw.set_window_should_close(viewer.window, True)

    viewer.render = render
    glfw.init()
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)                 # 隐藏窗口
    viewer.run(sim_per_frame=1.0 / FPS)      # 每帧固定推进 1/FPS 秒：视频按真实速度播放、不丢帧
    _save_webp(frames, path, FPS)


# =============================================================================
# 3. 臂角零空间控制
# =============================================================================
def record_arm_angle(path=os.path.join(OUT_DIR, "arm_angle_swivel.webp"), seconds=8.0):
    """运行 demo_arm_angle_swivel.py 本身（打开臂角几何量的可视化），用一个“假 viewer”
    代替 mujoco.viewer：演示每次 viewer.sync() 时，离屏渲染当前画面并存成一帧。"""
    import mujoco.viewer
    import demo_arm_angle_swivel as demo

    W, H, FPS = 1280, 800, 15
    big, small = _font(34), _font(26)
    frames = []

    class FakeViewer:
        def __init__(self, m, d):
            self.m, self.d = m, d
            self.cam = mujoco.MjvCamera()
            self.opt = mujoco.MjvOption()
            self.user_scn = mujoco.MjvScene(m, maxgeom=50)
            m.vis.global_.offwidth = max(m.vis.global_.offwidth, W)      # 离屏缓冲区够大
            m.vis.global_.offheight = max(m.vis.global_.offheight, H)
            self.renderer = mujoco.Renderer(m, H, W)
            self.dt = m.opt.timestep
            self.n_sync = 0
            self.every = max(1, int(round(1.0 / (FPS * self.dt))))
            ids = lambda kind, name: mujoco.mj_name2id(m, kind, name)
            self.ids = {side: (ids(mujoco.mjtObj.mjOBJ_BODY, f"{side}_link1"),
                               ids(mujoco.mjtObj.mjOBJ_BODY, f"{side}_link4"),
                               ids(mujoco.mjtObj.mjOBJ_SITE, f"{side}_ee"))
                        for side in ("left", "right")}
            self.targets = {side: d.site_xpos[self.ids[side][2]].copy()
                            for side in ("left", "right")}

        def is_running(self):
            return self.n_sync * self.dt < seconds

        def sync(self):
            self.n_sync += 1
            if self.n_sync % self.every:
                return
            cam = mujoco.MjvCamera()                 # 正面略偏左，看得到双肘的摆动
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = [0.2, 0.0, 1.05]
            cam.distance, cam.azimuth, cam.elevation = 2.3, 155.0, -15.0
            r = self.renderer
            r.update_scene(self.d, camera=cam, scene_option=self.opt)   # 含演示设的半透明
            for i in range(self.user_scn.ngeom):    # 演示画的目标球、臂角几何量
                if r.scene.ngeom < r.scene.maxgeom:
                    mujoco.mjv_initGeom(r.scene.geoms[r.scene.ngeom], 0, np.zeros(3),
                                        np.zeros(3), np.eye(3).ravel(), np.zeros(4, np.float32))
                    g, src = r.scene.geoms[r.scene.ngeom], self.user_scn.geoms[i]
                    g.type, g.size[:], g.pos[:], g.mat[:], g.rgba[:] = (
                        src.type, src.size, src.pos, src.mat, src.rgba)
                    g.category = mujoco.mjtCatBit.mjCAT_DECOR
                    r.scene.ngeom += 1
            img = Image.fromarray(r.render())
            draw = ImageDraw.Draw(img)
            d = self.d
            psi = {s: np.degrees(demo.compute_swivel_angle(d.xpos[b1], d.xpos[b4], d.site_xpos[site]))
                   for s, (b1, b4, site) in self.ids.items()}
            err = {s: np.linalg.norm(d.site_xpos[self.ids[s][2]] - self.targets[s]) * 1000
                   for s in self.ids}
            draw.rectangle([0, 0, W, 124], fill=(20, 20, 20))
            draw.text((20, 12), "Null-space swivel-angle control", font=big, fill=(255, 255, 255))
            draw.text((20, 56), f"hand error  L {err['left']:5.2f} mm   R {err['right']:5.2f} mm",
                      font=small, fill=(120, 230, 120))
            draw.text((20, 88), f"swivel angle  L {psi['left']:6.1f} deg   R {psi['right']:6.1f} deg",
                      font=small, fill=(230, 230, 230))
            frames.append(img)

        def close(self):
            self.renderer.close()

    original = mujoco.viewer.launch_passive
    mujoco.viewer.launch_passive = lambda m, d, **kw: FakeViewer(m, d)
    demo.mujoco.viewer.launch_passive = mujoco.viewer.launch_passive
    try:
        demo.main(show_psi_vectors=True)       # 画出臂角几何量：w、p（橙）和参考方向 l（蓝）
    finally:
        mujoco.viewer.launch_passive = original
    _save_webp(frames, path, FPS)


if __name__ == "__main__":
    which = sys.argv[1:] or ["compare", "interactive", "arm_angle"]
    os.makedirs(OUT_DIR, exist_ok=True)
    if "compare" in which:
        record_compare()
    if "interactive" in which:
        record_interactive()
    if "arm_angle" in which:
        record_arm_angle()
