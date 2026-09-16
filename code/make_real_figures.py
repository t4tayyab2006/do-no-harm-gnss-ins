"""
make_real_figures.py
--------------------
Figures and a compact summary for the real-data (PPC-Dataset, Nagoya -> Tokyo)
experiment, from results/real_*.npz. No filter re-runs.

  results/fig_real_frontier.png  mean ratio and worst case vs risk budget alpha
  results/fig_real_tail.png      sorted per-segment ratios, ungated vs certified
  results/real_frontier.json
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import figstyle as FS
from certify import ltt_select

FS.use()

R = "../results"
ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
SETS = [("learned_k0", r"learned, no indicator ($d'$ = 0)", FS.ORANGE, "s", (0, (4, 2))),
        ("learned_k3", r"learned, strong indicator ($d'$ = 3)", FS.GREEN, "^", (0, (1, 1.2))),
        ("perfectR_k3", "perfect-$R$ component", FS.BLUE, "o", "-")]


def frontier(C, G, eps=0.10, n_splits=200, seed=0):
    rng = np.random.default_rng(seed)
    N = len(C)
    splits = [rng.permutation(N) for _ in range(n_splits)]  # same splits for every alpha
    out = []
    for a in ALPHAS:
        rec = []
        for p in splits:
            cal, te = p[:N // 2], p[N // 2:]
            j = ltt_select(G, C, cal, a, 0.10, eps)
            r = G[te, j] / C[te]
            rec.append((r.mean(), np.mean(r > 1 + eps), r.max(), np.mean(r > 1 + eps) > a))
        m = np.array(rec, float)
        out.append(dict(alpha=a, mean_ratio=float(m[:, 0].mean()), harm=float(m[:, 1].mean()),
                        worst=float(m[:, 2].mean()), p_exceed=float(m[:, 3].mean())))
    return out, splits


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FS.WIDTH_FULL, FS.H_PAIR))
    summary = {}
    for key, label, col, mk, ls in SETS:
        Z = np.load(f"{R}/real_{key}.npz")
        C, G, L = Z["classical"], Z["gated"], Z["learned"]
        fr, _ = frontier(C, G)
        summary[key] = dict(ungated_ratio=float(np.mean(L / C)), ungated_worst=float(np.max(L / C)),
                            ungated_harm10=float(np.mean(L > 1.1 * C)), frontier=fr)
        ax1.plot(ALPHAS, [f["mean_ratio"] for f in fr], marker=mk, ls=ls, color=col, label=label)
        ax1.axhline(np.mean(L / C), color=col, ls=":", lw=1.0)
        ax2.plot(ALPHAS, [f["worst"] for f in fr], marker=mk, ls=ls, color=col, label=label)
        N = len(C); rng = np.random.default_rng(0)
        uw = np.mean([np.max((L / C)[rng.permutation(N)[N // 2:]]) for _ in range(200)])
        ax2.axhline(uw, color=col, ls=":", lw=1.0)
    ax1.axhline(1.0, color=FS.GREY, ls=(0, (4, 2)), lw=1.0)
    ax1.set_xlabel(r"risk budget $\alpha$ ($\varepsilon$ = 0.10)")
    ax1.set_ylabel("mean RMSE ratio vs. classical")
    FS.panel(ax1, "(a)")
    ax2.axhline(1.0, color=FS.GREY, ls=(0, (4, 2)), lw=1.0)
    ax2.set_xlabel(r"risk budget $\alpha$")
    ax2.set_ylabel("worst-case ratio (mean over splits)")
    FS.panel(ax2, "(b)")
    # one legend for both panels, below the axes: the three series are the same
    h, lab = ax1.get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, 0.035), ncol=3, frameon=False)
    FS.save(fig, "fig_real_frontier", extra_bottom=0.06)

    Z = np.load(f"{R}/real_learned_k3.npz")
    C, G, L = Z["classical"], Z["gated"], Z["learned"]
    j = int(np.where(Z["lambdas"] == 6.0)[0][0])  # median certified lambda at alpha=eps=0.10
    fig2, ax = plt.subplots(figsize=(FS.WIDTH_FULL, FS.H_WIDE))
    for name, r, col, ls in [(r"ungated learned ($d'$ = 3)", L / C, FS.VERM, "-"),
                             (r"certified gate ($\lambda$ = 6 m)", G[:, j] / C, FS.BLUE, (0, (4, 2)))]:
        ax.plot(np.arange(1, len(r) + 1), np.sort(r), color=col, ls=ls, lw=1.5, label=name)
    ax.axhline(1.0, color=FS.GREY, ls=":", lw=0.9)
    ax.set_xlabel(f"Tokyo segments sorted by ratio ({len(C)} segments)")
    ax.set_ylabel("RMSE ratio vs. classical")
    ax.legend(loc="upper left")
    FS.save(fig2, "fig_real_tail")

    json.dump(summary, open(f"{R}/real_frontier.json", "w"), indent=2)
    for k, v in summary.items():
        print(k, {kk: round(vv, 3) for kk, vv in v.items() if kk != "frontier"})
        for f in v["frontier"]:
            print("   ", {kk: round(vv, 3) for kk, vv in f.items()})


if __name__ == "__main__":
    main()
