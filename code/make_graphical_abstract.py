"""make_graphical_abstract.py -- graphical abstract for IEEE Sensors Journal (3 in wide).

Top: the method in one line. Bottom: the headline result, worst-case error relative to the
classical filter, ungated versus certified, in simulation and on real IMU data. The four
numbers are the ones reported in the manuscript (Table 3 and Table 4 of the full version).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import figstyle as FS

FS.use()

# worst-case RMSE ratio vs. classical filter: (ungated learned filter, certified system)
HEADLINE = {"Simulation": (10.50, 1.69), "Real IMU (Tokyo)": (1.86, 1.38)}

fig = plt.figure(figsize=(3.0, 2.3))
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.95], hspace=0.12)

# ---------------- method
ax = fig.add_subplot(gs[0])
ax.set_xlim(0, 100)
ax.set_ylim(0, 44)
ax.axis("off")
ax.grid(False)


def box(x, y, w, h, text, fc, fs=5.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.5",
                                fc=fc, ec="#3A3A3A", lw=0.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.25)


def arrow(x1, y1, x2, y2, col="#3A3A3A", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=5,
                                 lw=0.6, color=col, linestyle=ls))


box(1, 27, 27, 12, "Learned\nfilter", "#FBE3C8")
box(1, 9, 27, 12, "Classical\nfilter", "#D6EFE6")
box(38, 17, 25, 14, "Gate\n" + r"$\|p_L - p_C\| > \lambda$ ?", "#FDF3CF")
box(73, 16, 26, 16, "Output:\nlearned or\nclassical", "#EAF0F6", fs=5.2)
arrow(28.5, 33, 37.5, 27)
arrow(28.5, 15, 37.5, 21)
arrow(63.5, 24, 72.5, 24)
ax.text(50.5, 3.0, r"$\lambda$ certified by Learn-then-Test:" + "\n"
        + r"P(harm) $\leq \alpha$ with probability $\geq 1-\delta$",
        ha="center", va="center", fontsize=5.2, color="#8E3F80")
arrow(50.5, 8.6, 50.5, 16.4, col="#8E3F80", ls="--")

# ---------------- headline result
bx = fig.add_subplot(gs[1])
labels = list(HEADLINE)
ys = range(len(labels))
h = 0.36
for i, lab in enumerate(labels):
    ung, cert = HEADLINE[lab]
    bx.barh(i - h / 2, ung, h, color=FS.VERM, label="ungated" if i == 0 else None)
    bx.barh(i + h / 2, cert, h, color=FS.BLUE, hatch="////", edgecolor="white", linewidth=0.3,
            label="certified" if i == 0 else None)
    bx.text(ung * 1.06, i - h / 2, f"{ung:.2f}" + r"$\times$", va="center", fontsize=5.4)
    bx.text(cert * 1.06, i + h / 2, f"{cert:.2f}" + r"$\times$", va="center", fontsize=5.4)
bx.set_xscale("log")
bx.set_xlim(1, 25)
bx.set_xticks([1, 2, 5, 10])
bx.set_xticklabels(["1", "2", "5", "10"], fontsize=5.4)
bx.set_yticks(list(ys))
bx.set_yticklabels(labels, fontsize=5.6)
bx.invert_yaxis()
bx.set_xlabel("worst-case error relative to the classical filter", fontsize=5.6, labelpad=1.5)
bx.tick_params(axis="both", length=2, pad=1.5)
bx.grid(axis="y", visible=False)
bx.legend(loc="lower right", fontsize=5.2, handlelength=1.4, borderpad=0.3, labelspacing=0.25)

fig.savefig(f"{FS.R}/fig_graphical_abstract.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
fig.savefig(f"{FS.R}/fig_graphical_abstract.pdf", bbox_inches="tight", pad_inches=0.02,
            metadata={"CreationDate": None, "ModDate": None})
plt.close(fig)
print("  fig_graphical_abstract.png (600 dpi) + fig_graphical_abstract.pdf (vector)")
