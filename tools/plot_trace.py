"""demo_hqp_interactive.py仿真记录的绘图脚本。

仿真程序加 --data-record 时只记录数据，记录写到仓库根目录的 hqp_sim_trace/<时间>/trace.csv；
绘图都由本脚本完成。

用法（在工程根目录运行）：
  python3 tools/plot_trace.py                       画最新一次记录，保存 PNG
  python3 tools/plot_trace.py <记录目录或 trace.csv>  画指定记录，保存 PNG
  ... --show    打开可缩放 / 平移的交互窗口查看（全分辨率）
  ... --live    边仿真边看：每秒读取新数据刷新，仿真结束后窗口保留
PNG（joints.png、end_effectors.png）保存在 trace.csv 所在目录。
交互窗口：工具栏放大镜框选放大、十字箭头平移；缩放过的坐标轴不再自动跟随新数据，按 a 恢复自动缩放。
--show / --live 需要显示器（DISPLAY）。
"""
import io, os, sys, time
import numpy as np
import matplotlib

LIVE = "--live" in sys.argv
SHOW = LIVE or "--show" in sys.argv
if not SHOW:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 记录目录：仓库根目录下的 hqp_sim_trace/
HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hqp_sim_trace")


def find_csv(args):
    """参数可以是 trace.csv、记录目录，或省略（取本目录下最新一次记录）。"""
    if args:
        path = args[0]
        return os.path.join(path, "trace.csv") if os.path.isdir(path) else path
    runs = sorted(d for d in os.listdir(HERE) if os.path.isfile(os.path.join(HERE, d, "trace.csv")))
    if not runs:
        sys.exit("no recordings (<time>/trace.csv) in " + HERE)
    path = os.path.join(HERE, runs[-1], "trace.csv")
    print("[trace] latest recording: " + path)
    return path


csv = find_csv([a for a in sys.argv[1:] if not a.startswith("--")])
out = os.path.dirname(os.path.abspath(csv))
RED = "tab:red"
MAX_LIVE_POINTS = 20000  # 实时刷新时每条曲线最多画这么多点（超出则抽稀），仿真结束后用全分辨率重画


class Trace:
    """trace.csv 读取；实时模式下每次只读新写入的完整行。"""

    def __init__(self, path):
        while not os.path.exists(path):
            time.sleep(0.2)
        self.f = open(path, "r")
        header = ""
        while not header.endswith("\n"):
            header += self.f.readline()
            if not header.endswith("\n"):
                time.sleep(0.2)
        self.names = header.strip().split(",")
        self.col = {n: i for i, n in enumerate(self.names)}
        self.rows = np.zeros((0, len(self.names)))
        self.partial = ""

    def poll(self):
        chunk = self.f.read()
        if not chunk:
            return False
        lines = (self.partial + chunk).split("\n")
        self.partial = lines.pop()
        lines = [line for line in lines if line]
        if not lines:
            return False
        new = np.loadtxt(io.StringIO("\n".join(lines)), delimiter=",", ndmin=2)
        self.rows = np.vstack([self.rows, new])
        return True

    def __getitem__(self, name):
        return self.rows[:, self.col[name]]


def nanmax_abs(x):
    x = np.abs(x[np.isfinite(x)])
    return float(x.max()) if x.size else float("nan")


class Plots:
    def __init__(self, tr, interactive):
        self.tr = tr
        self.lines = []      # (曲线, 取数函数)
        self.axes = []       # (坐标轴, 是否在顶行标模式)
        self.n_marks = 0     # 已画的模式起点数
        self.figs = {}       # 文件名 -> figure
        joints = [n[4:] for n in tr.names if n.startswith("cmd_")]
        if interactive:
            # 屏幕上一张图放 18 个关节太挤：按躯干 / 左臂 / 右臂分成三个窗口
            self._joints("joints_torso", joints[:4], 2, (12, 8), "Joints: torso")
            self._joints("joints_left_arm", joints[4:11], 4, (17, 8), "Joints: left arm")
            self._joints("joints_right_arm", joints[11:], 4, (17, 8), "Joints: right arm")
            self._ees((17, 10))
        else:
            self._joints("joints", joints, 3, (16, 30), "Joints")
            self._ees((18, 16))
        for fig in self.figs.values():
            fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _setup(self, ax, top):
        ax.ticklabel_format(axis="y", useOffset=False)
        ax.tick_params(labelsize=7)
        self.axes.append((ax, top))

    def _line(self, ax, fn, *args, **kw):
        (line,) = ax.plot([], [], *args, **kw)
        self.lines.append((line, fn))
        return line

    def _joints(self, key, names, ncols, figsize, title):
        nrows = (len(names) + ncols - 1) // ncols
        fig = plt.figure(key, figsize=figsize)
        gs = fig.add_gridspec(2 * nrows, ncols, height_ratios=[2, 1] * nrows)
        for k, n in enumerate(names):
            r, c = k // ncols, k % ncols
            ax = fig.add_subplot(gs[2 * r, c])
            axe = fig.add_subplot(gs[2 * r + 1, c], sharex=ax)
            self._line(ax, lambda tr, n=n: np.degrees(tr["cmd_" + n]), "--", lw=1.0, label="target (command)")
            self._line(ax, lambda tr, n=n: np.degrees(tr["meas_" + n]), "-", lw=1.0, label="state")
            self._line(axe, lambda tr, n=n: np.degrees(tr["cmd_" + n] - tr["meas_" + n]),
                       color=RED, lw=0.7, label="error = target - state")
            axe.axhline(0.0, color="0.7", lw=0.6)
            ax.set_title(n, fontsize=9)
            ax.set_ylabel("deg", fontsize=8)
            axe.set_ylabel("err deg", fontsize=8, color=RED)
            ax.tick_params(labelbottom=False)
            self._setup(ax, r == 0)
            self._setup(axe, False)
            if k + ncols >= len(names):
                axe.set_xlabel("sim time (s)")
            if k == 0:
                handles = ax.get_legend_handles_labels()[0] + axe.get_legend_handles_labels()[0]
                labels = ax.get_legend_handles_labels()[1] + axe.get_legend_handles_labels()[1]
        fig.legend(handles, labels, loc="upper right")
        fig.suptitle(title + ": target (command) vs state (upper), tracking error (lower)   (dotted lines: mode changes)")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        self.figs[key] = fig

    def _ees(self, figsize):
        parts = [("left", "left hand"), ("right", "right hand"), ("torso", "torso")]
        fig = plt.figure("end_effectors", figsize=figsize)
        gs = fig.add_gridspec(6, 4, height_ratios=[2, 1] * 3)
        for r, (p, title) in enumerate(parts):
            for c, a in enumerate("xyz"):
                ax = fig.add_subplot(gs[2 * r, c])
                axe = fig.add_subplot(gs[2 * r + 1, c], sharex=ax)
                self._line(ax, lambda tr, p=p, a=a: tr["%s_target_%s" % (p, a)], "--", lw=1.0, label="target")
                self._line(ax, lambda tr, p=p, a=a: tr["%s_fk_%s" % (p, a)], "-", lw=1.0, label="FK state")
                self._line(axe, lambda tr, p=p, a=a: (tr["%s_fk_%s" % (p, a)] - tr["%s_target_%s" % (p, a)]) * 1000.0,
                           color=RED, lw=0.7, label="error = FK - target")
                axe.axhline(0.0, color="0.7", lw=0.6)
                ax.set_title("%s  %s" % (title, a), fontsize=9)
                ax.set_ylabel("m", fontsize=8)
                axe.set_ylabel("err mm", fontsize=8, color=RED)
                ax.tick_params(labelbottom=False)
                self._setup(ax, r == 0)
                self._setup(axe, False)
                if r == 2:
                    axe.set_xlabel("sim time (s)")
                if r == 0 and c == 0:
                    handles = ax.get_legend_handles_labels()[0] + axe.get_legend_handles_labels()[0]
                    labels = ax.get_legend_handles_labels()[1] + axe.get_legend_handles_labels()[1]
            ax = fig.add_subplot(gs[2 * r:2 * r + 2, 3])
            self._line(ax, lambda tr, p=p: self.ee_norm(tr, p), color=RED, lw=0.8)
            ax.set_title("%s  |FK - target|" % title, fontsize=9)
            ax.set_ylabel("mm", fontsize=8)
            self._setup(ax, r == 0)
            if r == 2:
                ax.set_xlabel("sim time (s)")
        fig.legend(handles, labels, loc="upper right")
        fig.suptitle("End effectors: target vs FK state (upper), tracking error (lower)   "
                     "(no target line = no task for it in that mode)")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        self.figs["end_effectors"] = fig

    @staticmethod
    def ee_norm(tr, p):
        e = [(tr["%s_fk_%s" % (p, a)] - tr["%s_target_%s" % (p, a)]) * 1000.0 for a in "xyz"]
        return np.sqrt(np.sum(np.square(e), axis=0))

    def update(self, full):
        tr = self.tr
        t = tr["t"]
        if t.size == 0:
            return
        step = 1 if full or t.size <= MAX_LIVE_POINTS else int(np.ceil(t.size / MAX_LIVE_POINTS))
        for line, fn in self.lines:
            line.set_data(t[::step], fn(tr)[::step])
        for ax, _ in self.axes:
            if ax.get_autoscale_on():  # 用户缩放过的坐标轴保持当前视图
                ax.relim()
                ax.autoscale_view()
        mode = tr["mode"]
        starts = [0] + list(np.nonzero(np.diff(mode))[0] + 1)
        for i in starts[self.n_marks:]:
            for ax, top in self.axes:
                if i > 0:
                    ax.axvline(t[i], color="0.6", ls=":", lw=0.8)
                if top:
                    ax.text(t[i], 0.98, " M%d" % mode[i], transform=ax.get_xaxis_transform(),
                            fontsize=7, color="0.35", va="top")
        self.n_marks = len(starts)
        for fig in self.figs.values():
            fig.canvas.draw_idle()

    def _on_key(self, event):
        if event.key == "a":  # 恢复自动缩放，跟随最新数据
            for ax, _ in self.axes:
                if ax.figure is event.canvas.figure:
                    ax.set_autoscale_on(True)
            self.update(full=True)

    def summary(self):
        tr = self.tr
        for n in [c[4:] for c in tr.names if c.startswith("cmd_")]:
            print("[trace] max |error| %-26s %.3f" % ("joint %s (deg)" % n,
                                                      nanmax_abs(np.degrees(tr["cmd_" + n] - tr["meas_" + n]))))
        for p, title in [("left", "left hand"), ("right", "right hand"), ("torso", "torso")]:
            print("[trace] max |error| %-26s %.3f" % ("%s position (mm)" % title, nanmax_abs(self.ee_norm(tr, p))))


tr = Trace(csv)
tr.poll()
plots = Plots(tr, interactive=SHOW)
if not LIVE:
    plots.update(full=True)
    if SHOW:
        plt.show()
    else:
        for key, fig in plots.figs.items():
            fig.savefig(os.path.join(out, key + ".png"), dpi=100)
        print("[trace] figures: %s  %s" % (os.path.join(out, "joints.png"), os.path.join(out, "end_effectors.png")))
        plots.summary()
else:
    plt.ion()
    plots.update(full=False)
    plt.show(block=False)
    idle, finalized = 0, False
    while plt.get_fignums():
        if tr.poll():
            plots.update(full=False)
            idle, finalized = 0, False
        else:
            idle += 1
            if idle >= 4 and not finalized:  # 约 2 s 没有新数据（仿真结束或暂停）：用全分辨率重画一次
                plots.update(full=True)
                finalized = True
        plt.pause(0.5)
