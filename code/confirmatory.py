"""
confirmatory.py
---------------
Pre-registered confirmatory test of the certified predictive switch on a FRESH
simulated pool. The hypotheses, pass criteria and analysis are fixed in
results/PREREGISTRATION_confirmatory.md, written (with SHA-256 fingerprints of this
file and of every module it uses) BEFORE this script was run.

Frozen components (no retraining, no retuning): the d'=3 learned-R detector and its
classical fallback (results/detector_k3.0.pt, results/quality_k3.0.json), the
KalmanNet-style filter (results/knet_sim.npz), the tuned process noise
(results/results.json), the threshold grids and the certificate settings.

Fresh pool: seeds 5000-5599, never used by any earlier experiment.

Usage:  python confirmatory.py run        (~15 min, 11 workers)
        python confirmatory.py analyze
"""
import sys
import json
import time
import multiprocessing as mp
import numpy as np

import benchmark as B
from benchmark_pred import PS_GRID

R = "../results"
SEEDS = list(range(5000, 5600))
CONS = [(0.20, 0.05), (0.02, 0.50)]   # two-level budget, as in the simulation benchmarks
DELTA = 0.10
N_SPLITS, N_CAL = 200, 300
_W = B._W


def _row(seed):
    from simulate import make_dataset
    from gated import run_gated
    ref, sens = make_dataset(seed=seed, kappa=B.KAPPA)
    q, L, C = _W["q"], _W["L"], _W["C"]
    n = len(ref["t"])
    kf = B._knet_factory()
    row = {"seed": seed, "classical": B.rmse(run_gated(sens, n, q, L, C, 0.0)["classical"], ref)}
    for name, fac in [("learnedR", None), ("knet", kf)]:
        row[f"{name}_ps"] = [B.rmse(run_gated(sens, n, q, L, C, lam, reset=False, stat="pred_switch",
                                              make_learned_filter=fac)["gated"], ref) for lam in PS_GRID]
        row[f"{name}_sep"] = [B.rmse(run_gated(sens, n, q, L, C, lam, reset=True, stat="sep_m",
                                               make_learned_filter=fac)["gated"], ref) for lam in B.LAMBDAS]
    return row


def run():
    setup = json.load(open(f"{R}/bench_sim_setup.json"))
    t0 = time.time()
    with mp.Pool(mp.cpu_count() - 1, initializer=B._init, initargs=(setup,)) as pool:
        rows = pool.map(_row, SEEDS, chunksize=2)
    json.dump(rows, open(f"{R}/confirmatory_rows.json", "w"))
    print(f"{len(rows)} fresh trajectories in {time.time()-t0:.0f}s")


def analyze():
    from certify import ltt_select_multi
    rows = json.load(open(f"{R}/confirmatory_rows.json"))
    C = np.array([r["classical"] for r in rows])
    N = len(C)
    rng = np.random.default_rng(0)
    splits = [rng.permutation(N) for _ in range(N_SPLITS)]
    i0 = PS_GRID.index(0.0)
    out = {"n": N, "splits": N_SPLITS, "n_cal": N_CAL}

    def evaluate(G, certified):
        rec = []
        for p in splits:
            cal, te = p[:N_CAL], p[N_CAL:]
            j = ltt_select_multi(G, C, cal, CONS, DELTA) if certified else None
            g = G[te, j] if certified else G[te]
            r = g / C[te]
            rec.append((r.mean(), np.mean(r > 1.05), np.mean(r > 1.10), np.mean(r > 1.50), r.max(),
                        (np.mean(r > 1.05) > CONS[0][0]) or (np.mean(r > 1.50) > CONS[1][0])))
        a = np.array(rec, float)
        return dict(mean_ratio=float(a[:, 0].mean()), harm5=float(a[:, 1].mean()), harm10=float(a[:, 2].mean()),
                    severe50=float(a[:, 3].mean()), worst=float(a[:, 4].mean()),
                    violation_frequency=float(a[:, 5].mean()),
                    share_splits_better_than_classical=float(np.mean(a[:, 0] < 1.0)))

    for name in ["learnedR", "knet"]:
        PS = np.array([r[f"{name}_ps"] for r in rows])
        SP = np.array([r[f"{name}_sep"] for r in rows])
        res = {"certified_predictive_switch": evaluate(PS, True),
               "certified_metre_gate": evaluate(SP, True),
               "imm_heuristic": evaluate(PS[:, i0], False),
               "ungated": evaluate(PS[:, -1], False)}
        cps, cmg = res["certified_predictive_switch"], res["certified_metre_gate"]
        res["H1_validity_pass"] = cps["violation_frequency"] <= DELTA
        if name == "knet":
            res["H2_gain_pass"] = (cps["mean_ratio"] < 1.0) and (cps["share_splits_better_than_classical"] >= 0.95)
            res["H3_beats_metre_gate_pass"] = cps["mean_ratio"] < cmg["mean_ratio"]
        out[name] = res
    json.dump(out, open(f"{R}/confirmatory.json", "w"), indent=2, default=bool)

    print(f"CONFIRMATORY TEST on {N} fresh trajectories (seeds 5000-5599), {N_SPLITS} splits of {N_CAL}/{N-N_CAL}")
    for name in ["learnedR", "knet"]:
        print(f"\n-- {name}")
        for m in ["ungated", "imm_heuristic", "certified_metre_gate", "certified_predictive_switch"]:
            v = out[name][m]
            print(f"   {m:<30} ratio {v['mean_ratio']:.3f}  >10% {v['harm10']*100:4.1f}%  >50% {v['severe50']*100:4.1f}%  "
                  f"worst {v['worst']:6.2f}x  budget violated in {v['violation_frequency']*100:5.1f}% of splits")
        print(f"   H1 validity (violations <= {DELTA:.0%}): {'PASS' if out[name]['H1_validity_pass'] else 'FAIL'}")
        if name == "knet":
            print(f"   H2 gain (mean ratio < 1 in >= 95% of splits): {'PASS' if out[name]['H2_gain_pass'] else 'FAIL'}"
                  f"  [{out[name]['certified_predictive_switch']['share_splits_better_than_classical']*100:.1f}% of splits]")
            print(f"   H3 beats certified metre gate: {'PASS' if out[name]['H3_beats_metre_gate_pass'] else 'FAIL'}")


if __name__ == "__main__":
    {"run": run, "analyze": analyze}[sys.argv[1]]()
