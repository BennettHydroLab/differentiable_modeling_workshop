"""Generate the workshop's schematic diagrams as SVG files.

Every diagram is drawn from code with matplotlib, so the HBV-lite base figure
and its variants stay in sync, and a change to the palette changes every
figure at once. Output goes to ``notebooks/figures/``.

    python tools/make_diagrams.py                 # all diagrams
    python tools/make_diagrams.py hbv_lite        # just one (or several)
    python tools/make_diagrams.py --png           # also write PNG previews
    python tools/make_diagrams.py --markdown      # print the notebook embeds

Notebooks reference the SVGs by absolute raw.githubusercontent.com URL, not
by relative path, because Colab opens only the .ipynb and cannot resolve
``figures/...`` next to it. ``--markdown`` prints the line to paste.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from workshop_utils.plotting import COLORS  # noqa: E402

OUT = ROOT / "notebooks" / "figures"
RAW_URL = ("https://raw.githubusercontent.com/BennettHydroLab/"
           "differentiable_modeling_workshop/main/notebooks/figures/{name}.svg")

# Text is converted to paths so the figures look identical in every viewer,
# and the hash salt plus a blank date keep the SVGs byte-stable across runs.
plt.rcParams.update({
    "svg.fonttype": "path",
    "svg.hashsalt": "differentiable-modeling-workshop",
    "font.size": 10,
    "mathtext.fontset": "dejavusans",
})

SCALE = 0.75  # inches per drawing unit; keeps text the same size in every figure

INK = "#333333"
MUTED = "#777777"
BWD = COLORS["ml"]          # backward pass / gradients
KIND = {
    "physics": COLORS["physics"],
    "nn": COLORS["ml"],
    "hybrid": COLORS["hybrid"],
    "loss": COLORS["extra"],
    "data": "#7a7a7a",
    "plain": INK,
}
DASHED = (0, (4, 3))


# --------------------------------------------------------------------------
# Drawing primitives
# --------------------------------------------------------------------------

def _tint(color, a=0.16):
    r, g, b = mcolors.to_rgb(color)
    return (1 - a + a * r, 1 - a + a * g, 1 - a + a * b)


def _visible_len(line):
    """Rough rendered length of a line, ignoring mathtext markup."""
    return len(re.sub(r"\\[a-zA-Z]+|[${}_^\\]", "", line))


@dataclass
class Box:
    x: float
    y: float
    w: float
    h: float

    @property
    def l(self):
        return (self.x - self.w / 2, self.y)

    @property
    def r(self):
        return (self.x + self.w / 2, self.y)

    @property
    def t(self):
        return (self.x, self.y + self.h / 2)

    @property
    def b(self):
        return (self.x, self.y - self.h / 2)

    def at(self, side, frac):
        """Point on a side; ``frac`` runs 0-1 left-to-right or bottom-to-top."""
        x0, y0 = self.x - self.w / 2, self.y - self.h / 2
        return {
            "t": (x0 + frac * self.w, y0 + self.h),
            "b": (x0 + frac * self.w, y0),
            "l": (x0, y0 + frac * self.h),
            "r": (x0 + self.w, y0 + frac * self.h),
        }[side]


def canvas(x0, x1, y0, y1):
    fig = plt.figure(figsize=((x1 - x0) * SCALE, (y1 - y0) * SCALE))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def box(ax, x, y, text, kind="plain", w=None, h=None, fs=10, sub=None, color=None):
    lines = text.split("\n")
    if w is None:
        w = max(_visible_len(s) for s in lines) * 0.105 * fs / 10 + 0.5
    if h is None:
        h = 0.36 + 0.32 * len(lines) * fs / 10 + (0.3 if sub else 0)
    ec = color or KIND[kind]
    fc = "white" if kind == "plain" and color is None else _tint(ec)
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0,rounding_size=0.12",
        fc=fc, ec=ec, lw=1.4, zorder=3))
    if sub:
        ax.text(x, y + 0.14, text, ha="center", va="center", fontsize=fs,
                zorder=4, linespacing=1.3)
        ax.text(x, y - h / 2 + 0.2, sub, ha="center", va="center", fontsize=7.5,
                color=MUTED, family="monospace", zorder=4)
    else:
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=4,
                linespacing=1.3)
    return Box(x, y, w, h)


def arrow(ax, p, q, color=INK, lw=1.4, ls="-", rad=0.0, label=None, lpos=0.5,
          loff=0.2, fs=9, lcolor=None, head=True, zorder=2, **text_kw):
    """Straight (rad=0) or arc3-curved arrow from p to q, optionally labelled.

    For curved arrows the label sits on the bulge side; for straight ones it
    sits above (or left of a vertical arrow). Negative ``loff`` flips it.
    """
    ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle="-|>" if head else "-", mutation_scale=12,
        color=color, lw=lw, linestyle=ls, shrinkA=0, shrinkB=0,
        connectionstyle=f"arc3,rad={rad}", zorder=zorder))
    if label:
        (x1, y1), (x2, y2) = p, q
        dx, dy = x2 - x1, y2 - y1
        n = np.hypot(dx, dy)
        cx, cy = (x1 + x2) / 2 + rad * dy, (y1 + y2) / 2 - rad * dx
        t = lpos
        bx = (1 - t) ** 2 * x1 + 2 * (1 - t) * t * cx + t ** 2 * x2
        by = (1 - t) ** 2 * y1 + 2 * (1 - t) * t * cy + t ** 2 * y2
        if rad:
            nx, ny = np.sign(rad) * dy / n, -np.sign(rad) * dx / n
        else:
            nx, ny = -dy / n, dx / n
            if ny < 0 or (ny == 0 and nx > 0):
                nx, ny = -nx, -ny
        kw = dict(ha="center", va="center", fontsize=fs, color=lcolor or color, zorder=5)
        kw.update(text_kw)
        ax.text(bx + nx * loff, by + ny * loff, label, **kw)


def path_arrow(ax, pts, color=INK, lw=1.4, ls="-", zorder=2):
    """Polyline with an arrowhead on the final segment (for elbow routes)."""
    xs, ys = zip(*pts)
    ax.add_line(Line2D(xs[:-1], ys[:-1], color=color, lw=lw, linestyle=ls,
                       solid_capstyle="butt", zorder=zorder))
    arrow(ax, pts[-2], pts[-1], color=color, lw=lw, ls=ls, zorder=zorder)


def text(ax, x, y, s, fs=10, color=INK, **kw):
    kw.setdefault("ha", "center")
    kw.setdefault("va", "center")
    ax.text(x, y, s, fontsize=fs, color=color, zorder=5, linespacing=1.3, **kw)


def tag(ax, x, y, s="NN$_\\varphi$", w=0.85, h=0.45, fs=9):
    """Small inline network component sitting on top of an arrow."""
    return box(ax, x, y, s, kind="nn", w=w, h=h, fs=fs)


def legend(ax, x, y, items, fs=9, dy=0.38):
    """items: (label, color, linestyle). Drawn as short arrows with text."""
    for i, (label, color, ls) in enumerate(items):
        yy = y - i * dy
        arrow(ax, (x, yy), (x + 0.7, yy), color=color, ls=ls)
        text(ax, x + 0.85, yy, label, fs=fs, ha="left")


# --------------------------------------------------------------------------
# Theory: notebook 01
# --------------------------------------------------------------------------

def computational_graph():
    fig, ax = canvas(-0.6, 12.8, -0.9, 5.0)
    W, H = 2.0, 1.0
    x = box(ax, 0.8, 2.0, "$x = 1.3$", w=W, h=H, sub="leaf, requires_grad")
    u = box(ax, 3.5, 2.0, "$u = x^2$", w=W, h=H, sub="PowBackward0")
    a = box(ax, 6.4, 3.4, "$a = \\sin u$", w=W, h=H, sub="SinBackward0")
    b = box(ax, 6.4, 0.6, "$b = \\cos u$", w=W, h=H, sub="CosBackward0")
    y = box(ax, 9.3, 2.0, "$y = a + b$", w=W, h=H, sub="AddBackward0")

    # forward: values flow left to right
    arrow(ax, x.r, u.l, label="1.3")
    arrow(ax, u.at("r", 0.75), a.at("l", 0.4), label="1.69", loff=0.22)
    arrow(ax, u.at("r", 0.25), b.at("l", 0.6), label="1.69", loff=-0.22)
    arrow(ax, a.at("r", 0.4), y.at("t", 0.3), label="0.993", loff=0.25)
    arrow(ax, b.at("r", 0.6), y.at("b", 0.3), label="−0.119", loff=-0.3)
    arrow(ax, y.r, (11.4, 2.0))
    text(ax, 11.5, 2.0, "$y = 0.874$", ha="left")

    # backward: chain-rule products flow right to left
    kw = dict(color=BWD, ls=DASHED, fs=8.5)
    arrow(ax, y.at("t", 0.7), a.r, rad=0.35, label="$\\bar{y} = 1$", loff=0.15, **kw)
    arrow(ax, y.at("b", 0.7), b.r, rad=-0.35, label="$\\bar{y} = 1$", loff=0.15, **kw)
    arrow(ax, a.l, u.t, rad=0.35, label="$\\bar{a}\\cos u = -0.119$", loff=0.35, **kw)
    arrow(ax, b.l, u.b, rad=-0.35, label="$-\\bar{b}\\sin u = -0.993$", loff=0.35, **kw)
    arrow(ax, u.at("b", 0.3), x.at("b", 0.7), rad=-0.5,
          label="$\\bar{u} \\cdot 2x = -2.891 = dy/dx$", loff=0.3, **kw)

    legend(ax, 0.0, 4.6, [("forward pass: values", INK, "-"),
                          ("backward pass: gradients (chain rule)", BWD, DASHED)])
    return fig


def forward_reverse_mode():
    fig, ax = canvas(-0.6, 12.4, -0.5, 4.3)
    m, n, c = 3, 8, 0.55

    def grid(x0, y0, hl_col=None, hl_row=None, color=INK):
        for i in range(m):
            for j in range(n):
                on = (j == hl_col) or (i == hl_row)
                ax.add_patch(Rectangle(
                    (x0 + j * c, y0 + (m - 1 - i) * c), c, c,
                    fc=_tint(color, 0.45) if on else "white",
                    ec=color if on else "#bbbbbb", lw=1.2 if on else 0.8, zorder=2))
        text(ax, x0 + n * c / 2, y0 - 0.35, "inputs / parameters ($n$)", fs=9, color=MUTED)
        text(ax, x0 - 0.3, y0 + m * c / 2, "outputs ($m$)", fs=9, color=MUTED, rotation=90)
        text(ax, x0 + n * c / 2, y0 + m * c / 2 + 1.25, "$J = \\partial(\\mathrm{outputs})"
             "/\\partial(\\mathrm{inputs})$", fs=9, color=MUTED)

    grid(0.6, 1.0, hl_col=2, color=COLORS["physics"])
    text(ax, 0.6 + 8 * c / 2, 3.9, "Forward mode: one column per pass", fs=10.5,
         fontweight="bold")
    text(ax, 0.6 + 8 * c / 2, 3.45, "$Jv$: every output w.r.t. one input", fs=9)

    grid(7.0, 1.0, hl_row=0, color=BWD)
    text(ax, 7.0 + 8 * c / 2, 3.9, "Reverse mode: one row per pass", fs=10.5,
         fontweight="bold")
    text(ax, 7.0 + 8 * c / 2, 3.45, "$v^{\\top}J$: one output w.r.t. every input", fs=9)

    text(ax, 5.9, -0.05, "Calibration has one scalar loss ($m = 1$) and many parameters "
         "($n$ large): reverse mode needs one pass, forward mode needs $n$.", fs=9.5)
    return fig


def unrolled_model():
    fig, ax = canvas(-0.3, 13.2, -1.4, 5.6)
    fx = [1.9, 4.5, 8.0]
    names = ["1", "2", "T"]
    states = [box(ax, 0.6, 2.0, "$u_0$", w=0.9, h=0.7),
              box(ax, 3.2, 2.0, "$u_1$", w=0.9, h=0.7),
              box(ax, 5.8, 2.0, "$u_2$", w=0.9, h=0.7),
              box(ax, 9.3, 2.0, "$u_T$", w=0.9, h=0.7)]
    steps = [box(ax, xx, 2.0, "$f$", kind="physics", w=0.9, h=0.7) for xx in fx]
    theta = box(ax, 5.0, 4.9, "$\\theta$  (shared by every step)", kind="physics", w=3.4, h=0.7)

    # forward chain
    arrow(ax, states[0].r, steps[0].l)
    arrow(ax, steps[0].r, states[1].l)
    arrow(ax, states[1].r, steps[1].l)
    arrow(ax, steps[1].r, states[2].l)
    arrow(ax, states[2].r, (6.55, 2.0), head=False)
    text(ax, 6.9, 2.0, "⋯", fs=14)
    arrow(ax, (7.25, 2.0), steps[2].l)
    arrow(ax, steps[2].r, states[3].l)

    loss = box(ax, 4.95, -0.8, "$\\mathcal{L} = \\sum_t \\, (Q_t - Q^{obs}_t)^2$",
               kind="loss", w=7.4, h=0.75)
    for s, nm in zip(steps, names):
        xb = box(ax, s.x - 0.55, 3.45, f"$x_{nm}$", kind="data", w=0.75, h=0.55, fs=9)
        arrow(ax, xb.b, s.at("t", 0.2), lw=1.1)
        arrow(ax, theta.at("b", 0.3 + 0.2 * names.index(nm)), s.at("t", 0.75),
              color=COLORS["physics"], lw=1.1)
        arrow(ax, s.b, (s.x, loss.t[1]), lw=1.1,
              label=f"$Q_{nm}$", lpos=0.62, loff=-0.3, fs=9)

    # backward: adjoints flow back along the state chain
    kw = dict(color=BWD, ls=DASHED, rad=-0.45)
    arrow(ax, states[3].at("b", 0.3), states[2].at("b", 0.7), **kw)
    arrow(ax, states[2].at("b", 0.3), states[1].at("b", 0.7), **kw)
    arrow(ax, states[1].at("b", 0.3), states[0].at("b", 0.7), **kw)
    text(ax, 10.0, 4.9, "backward pass:\n$\\partial\\mathcal{L}/\\partial u_t$ chained "
         "back to $u_0$;\n$\\partial\\mathcal{L}/\\partial\\theta$ collects a term\n"
         "from every step", fs=9, color=BWD, ha="left", va="top")
    return fig


# --------------------------------------------------------------------------
# Theory: notebook 02
# --------------------------------------------------------------------------

def dto_vs_adjoint():
    fig, ax = canvas(-0.4, 13.6, -0.6, 6.6)
    xs = np.linspace(1.0, 8.6, 6)
    labels = ["$u_0$", "$u_1$", "$u_2$", "$u_3$", "$u_4$", "$u_T$"]

    def chain(y, faded):
        for i, xx in enumerate(xs):
            keep = (not faded) or i in (0, len(xs) - 1)
            ax.add_patch(Circle((xx, y), 0.2, fc=COLORS["physics"] if keep else "white",
                                ec=COLORS["physics"], lw=1.4,
                                ls="-" if keep else (0, (2, 2)), zorder=3))
            text(ax, xx, y + 0.45, labels[i], fs=9)
        for x1, x2 in zip(xs[:-1], xs[1:]):
            arrow(ax, (x1 + 0.22, y), (x2 - 0.22, y), color=COLORS["physics"])

    # top: discretize-then-optimize
    y = 4.6
    text(ax, -0.2, 6.15, "Discretize then optimize (backprop through the solver)",
         fs=10.5, ha="left", fontweight="bold")
    chain(y, faded=False)
    for xx in xs[:-1]:
        ax.add_patch(Rectangle((xx - 0.22, y - 1.05), 0.44, 0.36, fc="#eeeeee",
                               ec=MUTED, lw=0.9, zorder=3))
        arrow(ax, (xx, y - 0.22), (xx, y - 0.69), color=MUTED, lw=0.9, ls=(0, (2, 2)))
    text(ax, xs[0] - 0.45, y - 0.87, "tape", fs=8.5, color=MUTED, ha="right")
    for x1, x2 in zip(xs[:-1], xs[1:]):
        arrow(ax, (x2 - 0.05, y - 1.15), (x1 + 0.05, y - 1.15), color=BWD, ls=DASHED,
              rad=-0.3)
    text(ax, 9.4, y, "memory grows with the\nnumber of steps;\nexact gradient of the\n"
         "computed trajectory", fs=9, ha="left")

    # bottom: optimize-then-discretize
    y = 1.2
    text(ax, -0.2, 2.6, "Optimize then discretize (continuous adjoint)",
         fs=10.5, ha="left", fontweight="bold")
    chain(y, faded=True)
    text(ax, xs[-1], y - 0.5, "keep only $u(T)$", fs=8.5, color=MUTED)
    for x1, x2 in zip(xs[:-1], xs[1:]):
        arrow(ax, (x2 - 0.05, y - 0.85), (x1 + 0.05, y - 0.85), color=BWD, ls=DASHED,
              rad=-0.3)
    text(ax, (xs[0] + xs[-1]) / 2, y - 1.4,
         "solve $u(t)$ and $\\lambda(t)$ backward from $T$", fs=9, color=BWD)
    text(ax, 9.4, y, "memory roughly constant;\nthe reverse solve of $u$\ncan be "
         "inaccurate or\nunstable on stiff systems", fs=9, ha="left")
    return fig


def ude_spectrum():
    fig, ax = canvas(0.0, 15.2, -1.2, 4.6)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "spec", [COLORS["physics"], "#9a9a9a", COLORS["ml"]])
    ax.imshow(np.linspace(0, 1, 256)[None, :], cmap=cmap, aspect="auto",
              extent=(0.4, 14.8, 3.3, 3.55), zorder=1)
    text(ax, 0.4, 4.05, "more physics: constraints, interpretability", fs=9.5,
         ha="left", color=COLORS["physics"])
    text(ax, 14.8, 4.05, "more learned: flexibility", fs=9.5, ha="right",
         color=COLORS["ml"])

    cards = [
        ("$\\dot{u} = f(u, x, \\theta)$", "physics only;\ncalibrate $\\theta$",
         "NB 00, 01"),
        ("$\\dot{u} = f(u, x, \\mathrm{NN}_\\varphi(x, a, u))$",
         "learned parameters:\n$\\theta$ varies with $x, a, u$", "NB 04;\n06 section 4"),
        ("$\\dot{u} = f(u, x, \\theta, \\mathrm{NN}_\\varphi(u, x))$",
         "learned process\ninside known physics", "NB 02 section 5, 03, 05;\n07 section 1"),
        ("$\\dot{u} = f(u, x, \\theta) + \\mathrm{NN}_\\varphi(u, x)$",
         "physics plus a\nlearned residual", "(not built here)"),
        ("$\\dot{u} = \\mathrm{NN}_\\varphi(u, x)$", "neural ODE:\nlearned dynamics",
         "NB 02 section 3"),
    ]
    for i, (eq, desc, where) in enumerate(cards):
        cx = 1.85 + i * 2.875
        col = cmap(i / 4)
        ax.plot([cx, cx], [3.3, 2.75], color=col, lw=1.4, zorder=1)
        ax.add_patch(FancyBboxPatch((cx - 1.35, -0.85), 2.7, 3.6,
                                    boxstyle="round,pad=0,rounding_size=0.12",
                                    fc=_tint(col, 0.12), ec=col, lw=1.4, zorder=2))
        text(ax, cx, 2.4, f"Form {i}", fs=10.5, fontweight="bold", color=col)
        text(ax, cx, 1.75, eq, fs=8.5)
        text(ax, cx, 0.8, desc, fs=9)
        text(ax, cx, -0.4, where, fs=8.5, color=MUTED)
    return fig


def neural_ode_vs_ude():
    fig, ax = canvas(0.0, 14.4, -0.6, 4.6)
    ax.plot([6.6, 6.6], [-0.4, 4.4], color="#dddddd", lw=1.2)

    # left: neural ODE
    text(ax, 0.2, 4.2, "Neural ODE (section 3)", fs=10.5, ha="left", fontweight="bold")
    u = box(ax, 0.9, 3.0, "$u_t$", w=0.9, h=0.7)
    xb = box(ax, 0.9, 1.6, "$x_t$", kind="data", w=0.9, h=0.7)
    nn = box(ax, 2.8, 2.3, "NN$_\\varphi$", kind="nn", w=1.3, h=0.8)
    sv = box(ax, 4.7, 2.3, "solver\nstep", w=1.3, h=0.9)
    arrow(ax, u.r, nn.at("l", 0.7))
    arrow(ax, xb.r, nn.at("l", 0.3))
    arrow(ax, nn.r, sv.l, label="$du/dt$", loff=0.25)
    arrow(ax, sv.r, (6.1, 2.3))
    text(ax, 6.2, 2.75, "$u_{t+1}$", ha="right", fs=9.5)
    text(ax, 3.2, 0.35, "the network is all of $du/dt$:\nno mass balance or positivity "
         "\nunless it learns them", fs=9, color=MUTED)

    # right: UDE
    text(ax, 6.9, 4.2, "UDE: HybridBucket (section 5)", fs=10.5, ha="left", fontweight="bold")
    s = box(ax, 7.8, 3.2, "$S_t$", w=0.9, h=0.7)
    nn = box(ax, 9.65, 3.2, "NN$_\\varphi(S/S_{ref})$", kind="nn", w=2.0, h=0.75)
    sp = box(ax, 11.6, 3.2, "softplus", kind="nn", w=1.3, h=0.75)
    pe = box(ax, 7.5, 1.4, "$P_t, E_t$", kind="data", w=1.0, h=0.7)
    mb = box(ax, 10.4, 1.4, "$S_{t+1} = S_t + P_t - ET(S_t, E_t) - q_t$",
             kind="physics", w=3.9, h=0.8)
    arrow(ax, s.r, nn.l)
    arrow(ax, nn.r, sp.l)
    arrow(ax, sp.r, (13.3, 3.2), label="$Q_t = q_t$", loff=0.25)
    arrow(ax, sp.b, mb.at("t", 0.84), label="$q_t \\geq 0$", loff=-0.45)
    arrow(ax, s.b, mb.at("t", 0.08))
    arrow(ax, pe.r, mb.l)
    arrow(ax, mb.r, (13.3, 1.4), label="$S_{t+1}$", loff=0.25)
    text(ax, 10.4, 0.15, "the physics keeps mass balance; the network fills one slot",
         fs=9, color=MUTED)
    return fig


# --------------------------------------------------------------------------
# HBV-lite and its variants (notebooks 00, 03, 04, 05)
# --------------------------------------------------------------------------

def _tank(ax, cx, cy, w, h, title, level=0.55):
    x0, x1, y0, y1 = cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2
    ax.add_patch(Rectangle((x0, y0), w, h * level, fc=_tint(COLORS["precip"], 0.35),
                           ec="none", zorder=1))
    ax.plot([x0, x0, x1, x1], [y1, y0, y0, y1], color=INK, lw=1.8,
            solid_joinstyle="miter", zorder=2)
    text(ax, cx, cy, title, fs=9.5)
    return Box(cx, cy, w, h)


def _hbv(ax, dx=0.0, slots=(), rain_in=False):
    """Draw HBV-lite with its origin shifted by ``dx``.

    ``slots`` names fluxes replaced by a learned closure ('soil', 'et',
    'route'); each is drawn green with an inline network tag. With
    ``rain_in`` the precipitation arrow is left for the caller (snow variant).
    """
    G = COLORS["hybrid"]
    _tank(ax, 2.0 + dx, 6.0, 2.6, 1.6, "Soil store\n$S_{soil}$")
    _tank(ax, 5.4 + dx, 3.6, 2.2, 1.4, "Fast reservoir\n$S_{fast}$")
    _tank(ax, 5.4 + dx, 0.9, 2.2, 1.4, "Slow reservoir\n$S_{slow}$")
    qx, qy = 8.6 + dx, 2.25
    ax.add_patch(Circle((qx, qy), 0.3, fc="white", ec=INK, lw=1.4, zorder=3))
    text(ax, qx, qy, "+", fs=13)

    if not rain_in:
        arrow(ax, (1.4 + dx, 8.3), (1.4 + dx, 6.85))
        text(ax, 1.4 + dx, 8.55, "$P$")

    # evapotranspiration
    on = "et" in slots
    arrow(ax, (2.6 + dx, 6.55), (3.2 + dx, 8.3), color=G if on else INK, lw=2.2 if on else 1.4)
    text(ax, 3.35 + dx, 8.45, "$ET = PET\\cdot\\sigma(\\mathrm{logit}\\,\\frac{S_{soil}}"
         "{LP\\cdot FC} + \\mathrm{NN}_\\varphi)$" if on else
         "$ET = PET\\cdot\\min(S_{soil}/(LP\\cdot FC),\\,1)$",
         fs=9, ha="left", color=G if on else INK)
    if on:
        tag(ax, 2.9 + dx, 7.42)

    # recharge
    on = "soil" in slots
    c, lw = (G, 2.2) if on else (INK, 1.4)
    path_arrow(ax, [(3.3 + dx, 5.6), (4.9 + dx, 5.6), (4.9 + dx, 4.35)], color=c, lw=lw)
    text(ax, 3.45 + dx, 5.95, "$R = P\\cdot\\sigma(\\mathrm{logit}\\,(S_{soil}/FC)^{\\beta}"
         " + \\mathrm{NN}_\\varphi)$" if on else
         ("$R = P_{in}\\,(S_{soil}/FC)^{\\beta}$" if rain_in else "$R = P\\,(S_{soil}/FC)^{\\beta}$"),
         fs=9, ha="left", va="bottom", color=c)
    if on:
        tag(ax, 4.1 + dx, 5.6)

    # percolation
    arrow(ax, (5.4 + dx, 2.9), (5.4 + dx, 1.65))
    text(ax, 5.25 + dx, 2.27, "$perc = \\min(PERC,\\,S_{fast})$", fs=9, ha="right")

    # fast release
    on = "route" in slots
    c, lw = (G, 2.2) if on else (INK, 1.4)
    arrow(ax, (6.5 + dx, 3.3), (qx - 0.27, qy + 0.14), color=c, lw=lw)
    if on:
        tag(ax, 7.42 + dx, 2.86)
        text(ax, 6.7 + dx, 3.6, "$q_{fast} = k\\,S_{fast}$\n$k = \\sigma(\\mathrm{logit}\\,"
             "K_{fast} + \\mathrm{NN}_\\varphi(S_{fast}/S_{ref}))$",
             fs=9, ha="left", va="bottom", color=G)
    else:
        text(ax, 7.0 + dx, 3.55, "$q_{fast} = K_{fast}\\,S_{fast}$", fs=9, ha="left")

    # slow release
    arrow(ax, (6.5 + dx, 0.7), (qx - 0.24, qy - 0.18))
    text(ax, 7.3 + dx, 0.75, "$q_{slow} = K_{slow}\\,S_{slow}$", fs=9, ha="left")

    arrow(ax, (qx + 0.3, qy), (qx + 1.3, qy))
    text(ax, qx + 1.45, qy, "$Q$", ha="left")


def hbv_lite():
    fig, ax = canvas(0.2, 10.6, -0.1, 9.0)
    _hbv(ax)
    return fig


def hbv_route_closure():
    fig, ax = canvas(0.2, 11.0, -0.1, 9.0)
    _hbv(ax, slots=("route",))
    return fig


def hbv_closure_slots():
    fig, ax = canvas(0.2, 11.0, -0.1, 9.2)
    _hbv(ax, slots=("soil", "et", "route"))
    text(ax, 0.4, 3.6, "Candidate slots\n(trained one at a\ntime, then all three):\n"
         "  soil:  runoff generation\n  et:     evaporative stress\n"
         "  route: storage-discharge", fs=9, ha="left", va="top", color=COLORS["hybrid"])
    return fig


def snow_bucket():
    dx = 3.6
    fig, ax = canvas(-0.6, 14.3, -0.1, 10.2)
    _hbv(ax, dx=dx, rain_in=True)

    split = box(ax, 1.4, 8.4, "rain/snow split\n$f_{snow} = \\sigma(-(T - T_{snow})/0.7)$",
                kind="physics", w=3.2, h=1.0, fs=9)
    arrow(ax, (1.4, 9.6), split.t)
    text(ax, 1.4, 9.8, "$P,\\ T$")
    _tank(ax, 1.4, 5.6, 2.0, 1.4, "Snowpack\nSWE", level=0.5)
    arrow(ax, split.b, (1.4, 6.35), label="snow", loff=0.35)
    path_arrow(ax, [split.r, (1.4 + dx, 8.4), (1.4 + dx, 6.85)])
    text(ax, 4.0, 8.6, "rain", fs=9)
    arrow(ax, (2.4, 5.5), (0.7 + dx, 5.5), label="melt", loff=0.2)
    text(ax, 1.4, 4.55, "melt = $DDF\\cdot\\mathrm{softplus}(T)$\n(capped at available SWE)\n\n"
         "$P_{in}$ = rain + melt",
         fs=8.5, va="top")
    return fig


def calibration_loop():
    fig, ax = canvas(-0.2, 14.3, -1.3, 4.6)

    def loop(x0, title, nodes, note, back_color):
        text(ax, x0 + 1.75, 4.2, title, fs=10.5, fontweight="bold")
        pos = [(x0, 3.0), (x0 + 3.5, 3.0), (x0 + 3.5, 0.9), (x0, 0.9)]
        bs = [box(ax, px, py, s, kind=k, w=2.8, h=0.95, fs=9)
              for (px, py), (s, k) in zip(pos, nodes)]
        arrow(ax, bs[0].r, bs[1].l)
        arrow(ax, bs[1].b, bs[2].t)
        arrow(ax, bs[2].l, bs[3].r, color=back_color,
              ls=DASHED if back_color == BWD else "-")
        arrow(ax, bs[3].t, bs[0].b)
        text(ax, x0 + 1.75, -0.6, note, fs=9, color=MUTED)

    loop(1.5, "Derivative-free (random search)",
         [("sample $\\theta^{(i)}$ within bounds", "physics"),
          ("simulate $Q(\\theta^{(i)})$", "physics"),
          ("score NSE", "loss"),
          ("keep the best so far", "plain")],
         "cost grows with the number of samples;\nno gradient needed", INK)
    loop(8.9, "Gradient-based",
         [("$\\theta$ = ParamMap(raw)", "physics"),
          ("simulate $Q(\\theta)$\n(autograd records the graph)", "physics"),
          ("loss $\\mathcal{L}(Q, Q^{obs})$", "loss"),
          ("backward: $\\partial\\mathcal{L}/\\partial$raw\nAdam update", "nn")],
         "one backward pass per step,\nhowever many parameters", BWD)
    return fig


def split_timeline():
    fig, ax = canvas(-1.8, 12.6, -1.6, 3.6)
    u = 1.5  # drawing units per water year

    def wy(year):
        return (year - 1997) * u

    rows = [("train", 1998, 1999, COLORS["physics"], 2.4),
            ("validation", 2000, 2001, COLORS["extra"], 1.4),
            ("test", 2002, 2004, "#E69F00", 0.4)]
    for name, first, last, col, y in rows:
        ax.add_patch(Rectangle((wy(first - 1), y - 0.3), u, 0.6, fc="#f2f2f2",
                               ec=MUTED, hatch="////", lw=0.8, zorder=2))
        ax.add_patch(Rectangle((wy(first), y - 0.3), wy(last + 1) - wy(first), 0.6,
                               fc=_tint(col, 0.35), ec=col, lw=1.4, zorder=2))
        text(ax, (wy(first) + wy(last + 1)) / 2, y, f"WY{first}–{str(last)[2:]}", fs=9)
        text(ax, -0.2, y, name, fs=10, ha="right")
    for yr in range(1997, 2005):
        text(ax, wy(yr) + u / 2, -0.25, f"WY{yr}", fs=8.5, color=MUTED)
        ax.plot([wy(yr)] * 2, [-0.05, 2.85], color="#e6e6e6", lw=0.8, zorder=1)
    ax.add_patch(Rectangle((0.0, -1.2), 0.6, 0.35, fc="#f2f2f2", ec=MUTED,
                           hatch="////", lw=0.8))
    text(ax, 0.75, -1.02, "365-day spin-up: simulated, not scored. It may overlap an "
         "earlier block; it only sets the initial states.", fs=8.5, ha="left")
    text(ax, 5.6, 3.3, "Chronological split by water year (1 Oct–30 Sep)", fs=10.5,
         fontweight="bold")
    return fig


def dynamic_parameterization():
    fig, ax = canvas(-0.4, 11.4, -0.4, 4.9)
    xb = box(ax, 1.0, 3.2, "$x_t$: T, R, P", kind="data", w=1.7, h=0.7)
    sw = box(ax, 1.0, 1.8, "SWE$_t$", w=1.7, h=0.7)
    nn = box(ax, 3.3, 2.5, "NN$_\\varphi$", kind="nn", w=1.3, h=0.8)
    pm = box(ax, 5.3, 2.5, "ParamMap\n0.5–8", w=1.5, h=0.9, fs=9)
    ph = box(ax, 7.9, 2.5, "physics step\n$f(u_t, x_t, \\mathrm{DDF}_t)$", kind="physics",
             w=2.4, h=0.95, fs=9)
    un = box(ax, 10.3, 2.5, "$u_{t+1}$", w=1.0, h=0.7)
    qb = box(ax, 10.3, 3.9, "$Q_t$", kind="loss", w=1.0, h=0.6)

    arrow(ax, xb.r, nn.at("l", 0.7))
    arrow(ax, sw.r, nn.at("l", 0.3))
    arrow(ax, nn.r, pm.l)
    arrow(ax, pm.r, ph.l, label="DDF$_t$", loff=0.25, fs=9)
    arrow(ax, ph.r, un.l)
    path_arrow(ax, [xb.t, (1.0, 4.35), (7.4, 4.35), ph.at("t", 0.3)])
    path_arrow(ax, [ph.at("t", 0.8), (8.82, 3.9), qb.l])
    path_arrow(ax, [un.b, (10.3, 0.55), (1.0, 0.55), sw.b], color=COLORS["physics"],
               ls=DASHED)
    text(ax, 5.6, 0.2, "next day: the simulated SWE$_{t+1}$ is a network input",
         fs=9, color=COLORS["physics"])
    return fig


def twin_experiment():
    fig, ax = canvas(-0.1, 13.5, 0.2, 4.4)
    specs = [("known truth\ne.g. DDF$(t)$, $q(S)$", "hybrid"),
             ("truth model\n+ real forcings", "physics"),
             ("pseudo-observations\n(+ noise)", "data"),
             ("train hybrid on\npseudo-observations", "nn"),
             ("learned function", "nn")]
    bs = [box(ax, 1.3 + 2.7 * i, 2.0, s, kind=k, w=2.4, h=1.0, fs=9)
          for i, (s, k) in enumerate(specs)]
    for a, b in zip(bs[:-1], bs[1:]):
        arrow(ax, a.r, b.l)
    path_arrow(ax, [bs[-1].t, (bs[-1].x, 3.4), (bs[0].x, 3.4), bs[0].t],
               color=COLORS["hybrid"], ls=DASHED)
    text(ax, 6.7, 3.65, "grade directly: recovered vs. true", fs=9, color=COLORS["hybrid"])
    text(ax, 6.7, 0.75, "With real data there is no such check: the true function is "
         "unknown.", fs=9, color=MUTED)
    return fig


# --------------------------------------------------------------------------
# Advanced: notebooks 06 and 07
# --------------------------------------------------------------------------

def truncated_bptt():
    fig, ax = canvas(-0.4, 14.0, -1.3, 3.8)
    n_win, n_node, gap, step = 3, 4, 1.3, 0.95
    x = 0.4
    starts = []
    for k in range(n_win):
        xs = [x + j * step for j in range(n_node)]
        starts.append(xs)
        ax.add_patch(FancyBboxPatch((xs[0] - 0.45, 0.35), xs[-1] - xs[0] + 0.9, 1.3,
                                    boxstyle="round,pad=0,rounding_size=0.15",
                                    fc="#fafafa", ec="#cccccc", lw=1.0, zorder=1))
        for j, xx in enumerate(xs):
            ax.add_patch(Circle((xx, 1.0), 0.18, fc=COLORS["physics"], ec="none", zorder=3))
            if j:
                arrow(ax, (xs[j - 1] + 0.2, 1.0), (xx - 0.2, 1.0), color=COLORS["physics"])
                arrow(ax, (xx - 0.08, 1.22), (xs[j - 1] + 0.08, 1.22), color=BWD,
                      ls=DASHED, rad=0.5)
        text(ax, (xs[0] + xs[-1]) / 2, 0.0,
             f"$\\mathcal{{L}}_{k + 1}$: backward + optimizer step", fs=8.5)
        text(ax, (xs[0] + xs[-1]) / 2, 2.35, f"window {k + 1}", fs=9, color=MUTED)
        x = xs[-1] + gap + 0.9
        if k < n_win - 1:
            bx = xs[-1] + 0.45 + gap / 2 + 0.0
            arrow(ax, (xs[-1] + 0.2, 1.0), (x - 0.2, 1.0), color=COLORS["physics"])
            ax.plot([bx, bx], [0.4, 1.9], color=BWD, lw=1.4, ls=(0, (2, 2)), zorder=2)
            text(ax, bx, 2.15, ".detach()", fs=8.5, color=BWD, family="monospace")
    legend(ax, 0.2, 3.35, [("state carries across windows", COLORS["physics"], "-")])
    legend(ax, 6.2, 3.35, [("gradient stops at each window boundary", BWD, DASHED)])
    text(ax, 6.8, -0.85, "Choose the window from the slowest store's residence time, "
         "not a round number.", fs=9, color=MUTED)
    return fig


def closure_to_equation():
    fig, ax = canvas(-0.1, 13.5, -0.4, 3.4)
    specs = [("trained closure\nNN$_\\varphi(S)$", "nn"),
             ("sample $q$ on visited\nstates (2nd–98th pct)", "plain"),
             ("candidate library\n$1,\\ S,\\ S^{1.5},\\ S^2,\\ S^3$", "plain"),
             ("STLSQ: fit, drop\nsmall terms, refit", "plain"),
             ("equation\n$q \\approx c\\,S^{1.5}$", "hybrid")]
    bs = [box(ax, 1.3 + 2.7 * i, 2.2, s, kind=k, w=2.4, h=1.05, fs=9)
          for i, (s, k) in enumerate(specs)]
    for a, b in zip(bs[:-1], bs[1:]):
        arrow(ax, a.r, b.l)
    chk = box(ax, 8.0, 0.35, "put it back in the model: re-simulate and compare\n"
              "against held-out data (or the truth, in a twin)", kind="hybrid",
              w=5.6, h=0.85, fs=9)
    path_arrow(ax, [bs[-1].b, (bs[-1].x, 0.35), chk.r], color=COLORS["hybrid"], ls=DASHED)
    text(ax, 2.6, 0.35, "library and threshold\nare modeling choices", fs=9, color=MUTED)
    return fig


def closure_absorbs():
    fig, ax = canvas(-0.2, 12.4, -1.0, 4.7)
    srcs = [("process error (what you want)", "hybrid"),
            ("solver / time-step error", "plain"),
            ("forcing error", "plain"),
            ("initial-state error", "plain"),
            ("observation error", "plain")]
    mis = box(ax, 6.1, 2.2, "model − observation\nmismatch", kind="loss", w=2.5, h=1.0)
    nn = box(ax, 9.6, 2.2, "learned closure\nNN$_\\varphi$ fits all of it", kind="nn",
             w=2.8, h=1.0)
    for i, (s, k) in enumerate(srcs):
        b = box(ax, 1.7, 4.0 - 0.9 * i, s, kind=k, w=3.2, h=0.65, fs=9,
                color=None if k != "plain" else "#999999")
        arrow(ax, b.r, mis.at("l", 0.85 - 0.175 * i),
              color=COLORS["hybrid"] if k == "hybrid" else "#999999", lw=1.2)
    arrow(ax, mis.r, nn.l)
    text(ax, 6.1, -0.55, "Notebook 07 section 2 changes only the solver and watches the "
         "learned closure change.", fs=9, color=MUTED)
    return fig


# --------------------------------------------------------------------------

DIAGRAMS = {
    # name: (function, alt text, where it goes)
    "computational_graph": (computational_graph,
                            "Computational graph of y = sin(x^2) + cos(x^2)", "01 section 1"),
    "forward_reverse_mode": (forward_reverse_mode,
                             "Forward-mode versus reverse-mode AD on a Jacobian", "01 section 3"),
    "unrolled_model": (unrolled_model,
                       "HBV-lite unrolled through time with a shared parameter vector", "01 section 4"),
    "dto_vs_adjoint": (dto_vs_adjoint,
                       "Discretize-then-optimize versus the continuous adjoint", "02 section 2"),
    "ude_spectrum": (ude_spectrum,
                     "Spectrum of universal differential equation forms", "02 section 4"),
    "neural_ode_vs_ude": (neural_ode_vs_ude,
                          "Neural ODE versus a UDE with a learned storage-discharge relation",
                          "02 section 5"),
    "hbv_lite": (hbv_lite, "HBV-lite model structure", "00 section 2"),
    "calibration_loop": (calibration_loop,
                         "Random-search versus gradient-based calibration loops", "00 sections 3–4"),
    "hbv_route_closure": (hbv_route_closure,
                          "HBV-lite with a learned fast-reservoir release", "03 section 1"),
    "split_timeline": (split_timeline,
                       "Chronological train/validation/test split with spin-up", "03 section 2"),
    "snow_bucket": (snow_bucket, "Degree-day snow store feeding HBV-lite", "04 section 2"),
    "dynamic_parameterization": (dynamic_parameterization,
                                 "Per-timestep data flow of dynamic parameterization", "04 section 4"),
    "twin_experiment": (twin_experiment, "Synthetic twin experiment workflow", "04 section 5"),
    "hbv_closure_slots": (hbv_closure_slots,
                          "HBV-lite with three candidate closure slots", "05 section 1"),
    "truncated_bptt": (truncated_bptt, "Truncated backpropagation through time", "06 section 2"),
    "closure_to_equation": (closure_to_equation,
                            "Distilling a learned closure into an equation", "07 section 1"),
    "closure_absorbs": (closure_absorbs,
                        "Error sources a learned closure absorbs", "07 section 2"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("names", nargs="*", help="diagrams to build (default: all)")
    ap.add_argument("--png", action="store_true", help="also write PNG previews")
    ap.add_argument("--markdown", action="store_true",
                    help="print notebook embed lines instead of drawing")
    args = ap.parse_args()

    names = args.names or list(DIAGRAMS)
    unknown = sorted(set(names) - set(DIAGRAMS))
    if unknown:
        ap.error(f"unknown diagram(s): {', '.join(unknown)}")

    if args.markdown:
        for name in names:
            _, alt, where = DIAGRAMS[name]
            print(f"{where:>8}  ![{alt}]({RAW_URL.format(name=name)})")
        return

    OUT.mkdir(parents=True, exist_ok=True)
    for name in names:
        fig = DIAGRAMS[name][0]()
        fig.savefig(OUT / f"{name}.svg", facecolor="white", metadata={"Date": None})
        if args.png:
            fig.savefig(OUT / f"{name}.png", facecolor="white", dpi=110)
        plt.close(fig)
        print(f"wrote {OUT.relative_to(ROOT)}/{name}.svg")


if __name__ == "__main__":
    main()
