"""
dev_gate.py
-----------
Chooses the gate STATISTIC on a separate development set (seeds 2000-2199),
disjoint from the certification pool (1000-1599), so that the design choice is
not tuned on the data used for the reported guarantee.

For each (statistic, threshold) it reports the mean RMSE ratio vs the classical
fallback and the harm rate. A good statistic reaches a low mean ratio while
keeping harm low enough to be certifiable: with 300 calibration trajectories at
alpha = 0.10, delta = 0.10, the empirical harm rate must be below about 7%.
"""
import json
import multiprocessing as mp
import numpy as np

DEV_SEEDS = list(range(2000, 2200))
GRID = {
    ("sep_norm", 0.7): [0.5, 1.0, 2.0, 4.0],
    ("sep_m", 0.7): [2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0],
    ("pred_resid", 0.5): [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
    ("pred_resid", 0.8): [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
}
_W = {}


def _init():
    import torch
    torch.set_num_threads(1)
    from ankf import DegradationDetector, make_classical_iae_fn
    S = json.load(open("../results/results.json"))
    det = DegradationDetector()
    det.load_state_dict(torch.load("../results/ankf_detector.pt"))
    det.eval()
    _W["q"] = {"q_accel": S["tuned_Q"]["q_accel"], "q_gyro": S["tuned_Q"]["q_gyro"]}
    _W["L"] = lambda: det.make_r_scale_fn(S["cfg"]["v2"][0], 1.0)
    _W["C"] = lambda: make_classical_iae_fn(S["cfg"]["iae"][0])


def _one(seed):
    from simulate import make_dataset
    from gated import run_gated
    q, L, C = _W["q"], _W["L"], _W["C"]
    ref, sens = make_dataset(seed=seed, duration_s=90.0)
    n = len(ref["t"])
    r = lambda p: float(np.sqrt(np.mean((p[:, 0] - ref["x"]) ** 2 + (p[:, 1] - ref["y"]) ** 2)))
    base = run_gated(sens, n, q, L, C, np.inf)
    row = {"classical": r(base["classical"]), "learned": r(base["learned"])}
    for (stat, beta), lams in GRID.items():
        for lam in lams:
            g = run_gated(sens, n, q, L, C, lam, reset=True, stat=stat, beta=beta)
            row[f"{stat}|{beta}|{lam}"] = (r(g["gated"]), g["open_frac"])
    return row


if __name__ == "__main__":
    with mp.Pool(mp.cpu_count() - 1, initializer=_init) as pool:
        rows = pool.map(_one, DEV_SEEDS, chunksize=4)
    C = np.array([r["classical"] for r in rows])
    Lr = np.array([r["learned"] for r in rows])
    print(f"DEV set ({len(rows)} trajectories). learned alone: mean ratio {np.mean(Lr/C):.3f}, "
          f"harm5 {np.mean(Lr > 1.05*C)*100:.1f}%, worst {np.max(Lr/C):.2f}x\n")
    print(f"{'statistic':<16}{'beta':>5}{'lambda':>8}{'open%':>7}{'mean ratio':>12}{'harm5%':>8}{'harm10%':>9}{'worst':>7}")
    out = {}
    for (stat, beta), lams in GRID.items():
        for lam in lams:
            key = f"{stat}|{beta}|{lam}"
            G = np.array([r[key][0] for r in rows]); op = np.mean([r[key][1] for r in rows])
            ratio = G / C
            res = dict(open=float(op), mean_ratio=float(ratio.mean()),
                       harm5=float(np.mean(ratio > 1.05)), harm10=float(np.mean(ratio > 1.10)),
                       worst=float(ratio.max()))
            out[key] = res
            print(f"{stat:<16}{beta:>5}{lam:>8}{op*100:>6.1f}%{res['mean_ratio']:>12.3f}"
                  f"{res['harm5']*100:>7.1f}%{res['harm10']*100:>8.1f}%{res['worst']:>6.2f}x")
        print()
    json.dump(out, open("../results/dev_gate.json", "w"), indent=2)
    print("Wrote ../results/dev_gate.json")
