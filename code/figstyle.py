"""
figstyle.py
-----------
Shared style for every manuscript figure, so that the six figures look like one
set and survive both printing and colour-blind readers.

  * Okabe-Ito palette (colour-blind safe) with a distinct marker and dash pattern
    per series, so the figures also read in greyscale.
  * Serif text at MDPI's body size, Greek symbols set as mathtext rather than
    spelled out ("alpha" -> alpha).
  * One canvas width (170 mm, MDPI's text width) for every figure.
  * Every figure is written twice: a 600 dpi PNG for the Word manuscript and a
    vector PDF for the LaTeX manuscript.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "../results"
DPI = 600

# Okabe-Ito, colour-blind safe
BLUE = "#0072B2"
VERM = "#D55E00"
GREEN = "#009E73"
ORANGE = "#E69F00"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
GREY = "#4D4D4D"

WIDTH_FULL = 6.9   # in; MDPI text width is 170 mm
H_PAIR = 3.05      # two panels side by side
H_WIDE = 3.30      # one wide panel


def use():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Palatino Linotype", "DejaVu Serif"],  # MDPI body face, with a fallback
        "mathtext.fontset": "custom",
        "mathtext.rm": "Palatino Linotype",
        "mathtext.it": "Palatino Linotype:italic",
        "mathtext.bf": "Palatino Linotype:bold",
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.0,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "axes.grid": True,
        "grid.color": "#B0B0B0",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.35,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
        "lines.markersize": 4.0,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "#CCCCCC",
        "legend.borderpad": 0.35,
        "legend.handlelength": 2.2,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def panel(ax, label):
    """MDPI-style panel label, drawn just above the axes."""
    ax.text(0.0, 1.02, label, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8.5, fontweight="bold")


def save(fig, name, extra_bottom=0.0):
    """Write <name>.png (600 dpi, for Word) and <name>.pdf (vector, for LaTeX).

    extra_bottom leaves room under the axes for a figure-level legend.
    """
    fig.tight_layout(pad=0.4, rect=(0, extra_bottom, 1, 1))
    png, pdf = f"{R}/{name}.png", f"{R}/{name}.pdf"
    fig.savefig(png, dpi=DPI)
    fig.savefig(pdf, metadata={"CreationDate": None, "ModDate": None})  # byte-identical on rebuild
    plt.close(fig)
    print(f"  {os.path.basename(png)} ({DPI} dpi) + {os.path.basename(pdf)} (vector)")
