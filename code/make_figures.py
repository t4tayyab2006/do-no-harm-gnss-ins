"""
make_figures.py
---------------
Final analysis and figures for the certified-gating paper, built from saved
result matrices (no filter re-runs). Run after experiment_passthrough.py,
experiment_quality.py <kappa> (for each kappa) and ablation.py.

  results/fig_frontier.png   risk budget alpha -> share of gain kept, worst case
  results/fig_validity.png   the guarantee holds: test harm vs alpha, P(harm>alpha)
  results/fig_tail.png       sorted per-trajectory ratios: ungated vs gated vs no-reset
  results/fig_kappa.png      learned detector across indicator informativeness
  results/final_summary.json key numbers quoted in the paper

Every figure is written twice: a 600 dpi PNG (for the Word manuscript) and a
vector PDF (for the LaTeX manuscript). The shared style lives in figstyle.py.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import figstyle as FS
from certify import ltt_select

FS.use()

R = "../results"
DELTA = 0.10
ALPHAS = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]
N_SPLITS, N_CAL = 200, 300


def split_eval(G, C, alpha, eps, naive=False, seed=0):
    rng = np.random.default_rng(seed)
    N = len(C)
    recs = []
    for _ in range(N_SPLITS):
        perm = rng.permutation(N)
        cal, te = perm[:N_CAL], perm[N_CAL:]
        j = int(np.argmin(G[cal].mean(axis=0))) if naive else ltt_select(G, C, cal, alpha, DELTA, eps)
        g = G[te, j]
        recs.append((np.mean(g / C[te]), np.mean(g > (1 + eps) * C[te]), np.max(g / C[te]), j))
    a = np.array(recs)
    return dict(mean_ratio=float(a[:, 0].mean()), harm=float(a[:, 1].mean()),
                p_exceed=float(np.mean(a[:, 1] > alpha)), worst=float(a[:, 2].mean()),
                median_j=int(np.median(a[:, 3])))


def ungated_split_worst(L, C, seed=0):
    """Worst-case ratio of an UNGATED component, averaged over the same 200 test
    halves used for the gated results, so figures and tables share one basis."""
    rng = np.random.default_rng(seed)
    N = len(C)
    return float(np.mean([np.max((L / C)[rng.permutation(N)[N_CAL:]]) for _ in range(N_SPLITS)]))


def main():
    P = np.load(f"{R}/passthrough.npz")
    C, G, comp, lam = P["classical"], P["gated"], P["oracle"], P["lambdas"]
    avail = 1 - np.mean(comp / C)
    summary = {"passthrough": {"available_gain": float(avail),
                               "component_harm5": float(np.mean(comp > 1.05 * C)),
                               "component_harm10": float(np.mean(comp > 1.10 * C)),
                               "component_worst": float(np.max(comp / C)), "by_alpha": {}}}

    # ---------- frontier + validity (pass-through) ----------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FS.WIDTH_FULL, FS.H_PAIR))
    figv, (bx1, bx2) = plt.subplots(1, 2, figsize=(FS.WIDTH_FULL, FS.H_PAIR))
    for eps, col, mk, ls in [(0.05, FS.BLUE, "o", "-"), (0.10, FS.ORANGE, "s", (0, (4, 2)))]:
        res = [split_eval(G, C, a, eps) for a in ALPHAS]
        naive = split_eval(G, C, 0.10, eps, naive=True)
        kept = [(1 - r["mean_ratio"]) / avail * 100 for r in res]
        lab = rf"certified gate, $\varepsilon$ = {eps:.2f}"
        ax1.plot(ALPHAS, kept, marker=mk, ls=ls, color=col, label=lab)
        ax2.plot(ALPHAS, [r["worst"] for r in res], marker=mk, ls=ls, color=col, label=lab)
        bx1.plot(ALPHAS, [r["harm"] for r in res], marker=mk, ls=ls, color=col,
                 label=rf"certified, $\varepsilon$ = {eps:.2f}")
        bx1.scatter([0.10], [naive["harm"]], marker="X", s=48, color=col, zorder=5,
                    edgecolor="white", linewidth=0.5,
                    label=rf"naive threshold, $\varepsilon$ = {eps:.2f}")
        bx2.plot(ALPHAS, [r["p_exceed"] for r in res], marker=mk, ls=ls, color=col,
                 label=rf"certified, $\varepsilon$ = {eps:.2f}")
        summary["passthrough"]["by_alpha"][f"eps{eps}"] = {
            str(a): dict(r, gain_kept_pct=k) for a, r, k in zip(ALPHAS, res, kept)}
        summary["passthrough"][f"naive_eps{eps}"] = naive
    ax1.axhline(100, color=FS.GREY, ls=":", lw=0.9)
    ax1.set_xlabel(r"risk budget $\alpha$ (max share of harmed trajectories)")
    ax1.set_ylabel("share of the component's gain kept (%)")
    ax1.set_ylim(0, 108)
    ax1.legend(loc="lower right")
    FS.panel(ax1, "(a)")
    ax2.axhline(ungated_split_worst(comp, C), color=FS.VERM, ls=(0, (5, 2)), lw=1.2,
                label="ungated component")
    ax2.axhline(1.0, color=FS.GREY, ls=":", lw=0.9)
    ax2.set_xlabel(r"risk budget $\alpha$")
    ax2.set_ylabel("worst-case RMSE ratio vs. classical")
    ax2.legend(loc="center left")
    FS.panel(ax2, "(b)")
    FS.save(fig, "fig_frontier")

    xs = np.array(ALPHAS)
    bx1.plot(xs, xs, color=FS.GREY, ls=(0, (4, 2)), lw=1.0, label=r"harm = $\alpha$ (the bound)")
    bx1.set_xlabel(r"certified level $\alpha$")
    bx1.set_ylabel("mean test harm rate")
    bx1.legend(loc="lower right")
    FS.panel(bx1, "(a)")
    bx2.axhline(DELTA, color=FS.GREY, ls=(0, (4, 2)), lw=1.0, label=r"$\delta$ = 0.10")
    bx2.set_ylim(0, 0.16)
    bx2.set_xlabel(r"certified level $\alpha$")
    bx2.set_ylabel(r"share of the 200 splits with harm > $\alpha$")
    bx2.legend(loc="upper left")
    FS.panel(bx2, "(b)")
    FS.save(figv, "fig_validity")

    # ---------- tail: ungated vs gated vs no-reset ----------
    j12 = int(np.where(lam == 12.0)[0][0])
    fig3, ax3 = plt.subplots(figsize=(FS.WIDTH_FULL, FS.H_WIDE))
    curves = [("ungated perfect-$R$ component", comp / C, FS.VERM, "-"),
              (r"gated + reset ($\lambda$ = 12 m)", G[:, j12] / C, FS.BLUE, "-")]
    if os.path.exists(f"{R}/ablation_noreset.npz"):
        NR = np.load(f"{R}/ablation_noreset.npz")["gated"]
        curves.append((r"gated, no reset ($\lambda$ = 12 m)", NR[:, j12] / C, FS.ORANGE, (0, (4, 2))))
        summary["ablation_reset"] = {
            str(l): dict(reset_ratio=float(np.mean(G[:, i] / C)), reset_harm5=float(np.mean(G[:, i] > 1.05 * C)),
                         reset_worst=float(np.max(G[:, i] / C)),
                         noreset_ratio=float(np.mean(NR[:, i] / C)), noreset_harm5=float(np.mean(NR[:, i] > 1.05 * C)),
                         noreset_worst=float(np.max(NR[:, i] / C)))
            for i, l in enumerate(lam)}
    for name, r, col, ls in curves:
        ax3.plot(np.arange(1, len(r) + 1), np.sort(r), color=col, ls=ls, lw=1.5, label=name)
    ax3.axhline(1.0, color=FS.GREY, ls=":", lw=0.9)
    ax3.set_yscale("log")
    ax3.set_xlim(len(C) * 0.6, len(C) + 5)
    hi = max(float(np.max(r)) for _, r, _, _ in curves)
    ax3.set_ylim(0.88, hi * 1.18)
    ax3.set_yticks([0.9, 1.0, 1.25, 1.5, 2.0, 3.0])
    ax3.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax3.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax3.set_xlabel("pool trajectories sorted by ratio (worst 40% shown)")
    ax3.set_ylabel("RMSE ratio vs. classical (log scale)")
    ax3.legend(loc="upper left")
    FS.save(fig3, "fig_tail")

    # ---------- kappa sweep (learned detector) ----------
    ks = [k for k in [0.0, 1.0, 2.0, 3.0] if os.path.exists(f"{R}/quality_k{k}.npz")]
    summary["kappa"] = {}
    if ks:
        fig4, (cx1, cx2) = plt.subplots(1, 2, figsize=(FS.WIDTH_FULL, FS.H_PAIR))
        rows = []
        for k in ks:
            Z = np.load(f"{R}/quality_k{k}.npz")
            Ck, Gk, Lk = Z["classical"], Z["gated"], Z["learned"]
            r = split_eval(Gk, Ck, 0.10, 0.10)
            acc = json.load(open(f"{R}/quality_k{k}.json"))["config"]["detector_val_acc"]
            rows.append((k, np.mean(Lk / Ck), np.mean(Lk > 1.10 * Ck), ungated_split_worst(Lk, Ck),
                         r["mean_ratio"], r["harm"], r["worst"], r["p_exceed"], acc))
            summary["kappa"][str(k)] = dict(detector_acc=acc, learned_ratio=rows[-1][1], learned_harm10=rows[-1][2],
                                            learned_worst=rows[-1][3], gated_ratio=r["mean_ratio"],
                                            gated_harm10=r["harm"], gated_worst=r["worst"],
                                            gated_p_exceed=r["p_exceed"])
        rows = np.array(rows)
        x = np.arange(len(ks)); w = 0.38
        bkw = dict(edgecolor="white", linewidth=0.6)
        b1 = cx1.bar(x - w / 2, rows[:, 3], w, color=FS.VERM, label="learned, ungated", **bkw)
        b2 = cx1.bar(x + w / 2, rows[:, 6], w, color=FS.BLUE, hatch="//",
                     label=r"certified gate ($\alpha$ = $\varepsilon$ = 0.10)", **bkw)
        cx1.bar_label(b1, fmt="%.1f", fontsize=6.5, padding=1.5)
        cx1.bar_label(b2, fmt="%.1f", fontsize=6.5, padding=1.5)
        cx1.axhline(1, color=FS.GREY, ls=":", lw=0.9)
        cx1.set_xticks(x)
        cx1.set_xticklabels([f"$d\'$ = {k:g}\n({a*100:.0f}% acc.)" for k, a in zip(ks, rows[:, 8])])
        cx1.set_ylim(0, rows[:, 3].max() * 1.20)
        cx1.set_ylabel("worst-case ratio vs. classical")
        cx1.legend(loc="upper left")
        FS.panel(cx1, "(a)")
        b3 = cx2.bar(x - w / 2, rows[:, 2] * 100, w, color=FS.VERM, label="learned, ungated", **bkw)
        b4 = cx2.bar(x + w / 2, rows[:, 5] * 100, w, color=FS.BLUE, hatch="//",
                     label="certified gate", **bkw)
        cx2.bar_label(b3, fmt="%.1f", fontsize=6.5, padding=1.5)
        cx2.bar_label(b4, fmt="%.1f", fontsize=6.5, padding=1.5)
        cx2.axhline(10, color=FS.GREY, ls=(0, (4, 2)), lw=1.0, label=r"$\alpha$ = 10%")
        cx2.set_xticks(x)
        cx2.set_xticklabels([f"$d\'$ = {k:g}" for k in ks])
        cx2.set_ylim(0, rows[:, 2].max() * 100 * 1.24)
        cx2.set_ylabel("trajectories >10% worse than classical (%)")
        cx2.legend(loc="upper right")
        FS.panel(cx2, "(b)")
        FS.save(fig4, "fig_kappa")

    # ---------- two-level certificate (frequency + severity) ----------
    # Pre-specified pairing: >5% worse on at most 20% of trajectories AND
    # >50% worse on at most 2%. Compared with the single-level alpha=0.20 case,
    # which readmits the blow-ups because it bounds only frequency.
    from certify import ltt_select_multi
    cons = [(0.20, 0.05), (0.02, 0.50)]
    summary["two_level"] = {}
    settings = [("perfect-R (pass-through)", C, G)]
    for k in ks:
        Z = np.load(f"{R}/quality_k{k}.npz")
        settings.append((f"learned d'={k:g}", Z["classical"], Z["gated"]))
    for name, Cs, Gs in settings:
        rng = np.random.default_rng(0)
        single, multi = [], []
        for _ in range(N_SPLITS):
            perm = rng.permutation(len(Cs)); cal, te = perm[:N_CAL], perm[N_CAL:]
            for sel, store in [(ltt_select(Gs, Cs, cal, 0.20, DELTA, 0.05), single),
                               (ltt_select_multi(Gs, Cs, cal, cons, DELTA), multi)]:
                r = Gs[te, sel] / Cs[te]
                store.append((r.mean(), np.mean(r > 1.05), np.mean(r > 1.50), r.max()))
        s, m = np.array(single), np.array(multi)
        summary["two_level"][name] = {
            "single_alpha0.2": dict(mean_ratio=float(s[:, 0].mean()), harm5=float(s[:, 1].mean()),
                                    severe50=float(s[:, 2].mean()), worst=float(s[:, 3].mean())),
            "two_level": dict(mean_ratio=float(m[:, 0].mean()), harm5=float(m[:, 1].mean()),
                              severe50=float(m[:, 2].mean()), worst=float(m[:, 3].mean()),
                              p_severe_exceeds=float(np.mean(m[:, 2] > 0.02)))}

    json.dump(summary, open(f"{R}/final_summary.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "ablation_reset"}, indent=1)[:4000])


if __name__ == "__main__":
    main()
