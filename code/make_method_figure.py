"""make_method_figure.py -- Figure 1 (method block diagram), 600 dpi PNG + vector PDF."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import figstyle as FS

FS.use()

LEARNED = "#FBE3C8"   # tint of the orange used for learned filters in the other figures
CLASSIC = "#D6EFE6"   # tint of the green used for the classical fallback
GATE = "#FDF3CF"
CALIB = "#F1E4F3"
EDGE = "#3A3A3A"

fig, ax = plt.subplots(figsize=(FS.WIDTH_FULL, 3.15))
ax.set_xlim(0, 100)
ax.set_ylim(-0.5, 42)
ax.axis("off")
ax.grid(False)


def box(x, y, w, h, text, fc, fs=6.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc=fc, ec=EDGE, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.45)


def arrow(x1, y1, x2, y2, text=None, ls="-", col=EDGE):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9,
                                 lw=0.9, color=col, linestyle=ls))
    if text:
        ax.text((x1 + x2) / 2 + 1.5, (y1 + y2) / 2, text, ha="left", fontsize=6, color=col)


box(1, 19, 12, 9, "IMU\n(100 Hz)\n+\nGNSS fixes\n(1 Hz)", "#EAF0F6", 6.5)
box(21, 31, 22, 10, "Learned adaptive filter $F_L$\n(network sets $R$ from innovation\nstatistics + receiver quality)", LEARNED)
box(21, 7, 22, 10, "Classical adaptive filter $F_C$\n($\\chi^2$ test / quality rule)\n= trusted fallback", CLASSIC)
box(51, 18, 19, 11, "Solution-separation gate\n" + r"$d=\|p_L-p_C\|$ (metres)" + "\n" + r"$d>\lambda$ ?", GATE)
box(79, 27, 19, 9, "output $p_L$\n(gate open)", LEARNED)
box(79, 10, 19, 9, "output $p_C$\n" + r"(optional reset $F_L \leftarrow F_C$)", CLASSIC)
box(45, 0.5, 31, 8.5, "Learn-then-Test calibration of " + r"$\lambda$" + " (offline):\n"
    + r"$P(\mathrm{RMSE}_{g} > (1+\varepsilon)\,\mathrm{RMSE}_{C}) \leq \alpha$  w.p. $\geq 1-\delta$",
    CALIB, 6.0)
arrow(13.5, 25, 20.5, 35)
arrow(13.5, 22, 20.5, 12)
arrow(43.5, 36, 50.5, 26)
arrow(43.5, 12, 50.5, 21)
arrow(70.5, 26, 78.5, 31, "no")
arrow(70.5, 21, 78.5, 15, "yes")
arrow(60.5, 9.3, 60.5, 17.5, r"sets $\lambda$", ls="--", col=FS.PURPLE)
FS.save(fig, "fig_method")
