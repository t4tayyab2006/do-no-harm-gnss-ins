"""
plot_results.py
----------------
Figures for the write-up. Run AFTER train_eval.py and safeguard_study.py.

Fig 1 trajectory_comparison.png : worst-divergence test trajectory, top-down
Fig 2 error_over_time.png       : position error vs time, degraded windows shaded
Fig 3 rmse_comparison.png       : mean/median test RMSE per method + oracle bound
Fig 4 tail_risk.png             : sorted per-trajectory RMSE ratio vs classical EKF
                                  (the key figure: means hide the tail)
"""
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulate import make_dataset
from ekf import run_ekf
from features import decimate_array
from ankf import DegradationDetector, make_classical_iae_fn

DUR = 90.0


def main():
    with open("../results/results.json") as f:
        S = json.load(f)
    with open("../results/safeguard_study.json") as f:
        SG = json.load(f)

    det = DegradationDetector()
    det.load_state_dict(torch.load("../results/ankf_detector.pt"))
    det.eval()

    q = {"q_accel": S["tuned_Q"]["q_accel"], "q_gyro": S["tuned_Q"]["q_gyro"]}
    vi2 = S["cfg"]["v2"][0]
    iae_vi = S["cfg"]["iae"][0]
    sg_vi, sg_tg, sg_mc = SG["selected_cfg"]

    rows = S["per_trajectory"]
    worst = max(rows, key=lambda r: r["ankf_rmse"] / r["ekf_rmse"])
    seed = worst["seed"]
    print(f"Plotting worst divergence trajectory: seed={seed} "
          f"(learned {worst['ankf_rmse']/worst['ekf_rmse']:.2f}x the classical EKF)")

    ref, sens = make_dataset(seed=seed, duration_s=DUR)
    n = len(ref["t"])
    runs = {
        "Classical EKF (tuned)": run_ekf(sens, n, **q),
        "Classical IAE (no ML)": run_ekf(sens, n, r_adapt_fn=make_classical_iae_fn(iae_vi), **q),
        "Learned R (no safeguard)": run_ekf(sens, n, r_adapt_fn=det.make_r_scale_fn(vi2, 1.0), **q),
        "Learned R + safeguard": run_ekf(sens, n, r_adapt_fn=det.make_safeguarded_fn(sg_vi, 1.0, sg_tg, sg_mc), **q),
        "Oracle R (bound)": run_ekf(sens, n, r_gps_base=None, **q),
    }
    colors = {"Classical EKF (tuned)": "tab:red", "Classical IAE (no ML)": "tab:purple",
              "Learned R (no safeguard)": "tab:orange", "Learned R + safeguard": "tab:green",
              "Oracle R (bound)": "tab:blue"}

    rx, ry, t = decimate_array(ref["x"]), decimate_array(ref["y"]), decimate_array(ref["t"])
    outage = decimate_array(sens["outage_mask_full"])

    # Fig 1
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.plot(rx, ry, "k-", lw=2.5, label="Ground truth", zorder=5)
    for name, out in runs.items():
        ax.plot(decimate_array(out["x"]), decimate_array(out["y"]), lw=1.4, color=colors[name], label=name)
    ax.scatter(sens["gps_x"][sens["gps_available"]], sens["gps_y"][sens["gps_available"]],
               s=8, color="gray", alpha=0.35, label="GPS fixes", zorder=1)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.axis("equal")
    ax.set_title(f"Worst-divergence test trajectory (seed {seed})")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig("../results/trajectory_comparison.png", dpi=150)

    # Fig 2
    fig2, ax2 = plt.subplots(figsize=(9.5, 4.2))
    emax = 0
    for name, out in runs.items():
        e = np.sqrt((decimate_array(out["x"]) - rx) ** 2 + (decimate_array(out["y"]) - ry) ** 2)
        emax = max(emax, e.max())
        ax2.plot(t, e, lw=1.3, color=colors[name], label=name)
    ax2.fill_between(t, 0, emax * 1.05, where=outage, color="gray", alpha=0.2, label="GPS degraded")
    ax2.set_xlabel("time (s)"); ax2.set_ylabel("position error (m)"); ax2.set_ylim(0, emax * 1.05)
    ax2.set_title("Position error over time")
    ax2.legend(fontsize=8)
    fig2.tight_layout(); fig2.savefig("../results/error_over_time.png", dpi=150)

    # Fig 3
    methods = [("Classical\nEKF", "ekf"), ("Classical\nIAE", "iae"), ("ANKF v1\nresidual", "resid"),
               ("Learned R\n(v2)", "ankf"), ("Oracle R\n(bound)", "oracle")]
    fig3, ax3 = plt.subplots(figsize=(8.5, 4.5))
    xp = np.arange(len(methods)); w = 0.38
    ax3.bar(xp - w / 2, [S[k + "_rmse"] for _, k in methods], width=w, label="mean RMSE", color="tab:blue")
    ax3.bar(xp + w / 2, [S[k + "_median_rmse"] for _, k in methods], width=w, label="median RMSE", color="tab:gray")
    ax3.set_xticks(xp); ax3.set_xticklabels([m for m, _ in methods], fontsize=8)
    ax3.set_ylabel("RMSE (m)")
    ax3.set_title(f"Test set (n={S['n_test']}): mean is far above median — rare blow-ups dominate")
    ax3.legend(fontsize=8)
    fig3.tight_layout(); fig3.savefig("../results/rmse_comparison.png", dpi=150)

    # Fig 4 -- the key figure
    fig4, ax4 = plt.subplots(figsize=(9, 4.8))
    for lbl, k, c in [("Classical IAE (no ML)", "iae", "tab:purple"),
                      ("Learned R (no safeguard)", "ankf", "tab:orange"),
                      ("Oracle R (bound)", "oracle", "tab:blue")]:
        r = np.sort(np.array([row[k + "_rmse"] / row["ekf_rmse"] for row in rows]))
        ax4.plot(np.arange(1, len(r) + 1), r, marker="o", ms=3, lw=1.3, color=c, label=lbl)
    ax4.axhline(1.0, color="k", ls="--", lw=1.2, label="classical EKF (=1.0)")
    ax4.set_yscale("log")
    ax4.set_xlabel("test trajectories, sorted by ratio")
    ax4.set_ylabel("RMSE ratio vs classical EKF (log)")
    ax4.set_title("Tail risk: the learned filter wins on average but blows up on the worst case")
    ax4.legend(fontsize=8)
    fig4.tight_layout(); fig4.savefig("../results/tail_risk.png", dpi=150)

    print("Saved 4 figures to ../results/")


if __name__ == "__main__":
    main()
