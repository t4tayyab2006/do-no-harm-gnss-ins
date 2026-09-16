"""
calibration_size.py
-------------------
Pre-registered secondary analysis: HOW MUCH CALIBRATION DATA DOES THE CERTIFICATE
NEED?

The confirmatory test (confirmatory.py) rejected H2: with 300 calibration
trajectories the certified predictive switch was better than the classical filter
in only 75.5% of splits, because in the remaining 24.5% the calibration half could
not certify any threshold beyond the classical fallback. This analysis asks whether
that is a property of the method or of the calibration sample size.

It re-uses the SAME frozen per-trajectory RMSE matrix produced by the confirmatory
run (results/confirmatory_rows.json, 600 fresh trajectories, seeds 5000-5599). No
filter is re-run, no model is re-trained: only the split sizes change.

Design (fixed before running, see results/PREREGISTRATION_calibration_size.md):
  calibration sizes m in {150, 300, 450}; test size fixed at 150 trajectories so
  that test-set noise is identical across m; 200 random splits per m (rng seed 1);
  same two-level budget, same delta, same threshold grid as the confirmatory test.

Usage:  python calibration_size.py            (seconds)
"""
import json
import numpy as np

from certify import ltt_select_multi
from benchmark_pred import PS_GRID
from confirmatory import CONS, DELTA

R = "../results"
CAL_SIZES = [150, 300, 450]
N_TEST = 150
N_SPLITS = 200
RNG_SEED = 1


def main():
    rows = json.load(open(f"{R}/confirmatory_rows.json"))
    C = np.array([r["classical"] for r in rows])
    N = len(C)
    rng = np.random.default_rng(RNG_SEED)
    perms = [rng.permutation(N) for _ in range(N_SPLITS)]
    out = {"n": N, "splits": N_SPLITS, "n_test": N_TEST, "cal_sizes": CAL_SIZES, "rng_seed": RNG_SEED}

    for name in ["learnedR", "knet"]:
        PS = np.array([r[f"{name}_ps"] for r in rows])
        per_size = {}
        for m in CAL_SIZES:
            rec = []
            for p in perms:
                cal, te = p[:m], p[m:m + N_TEST]
                j = ltt_select_multi(PS, C, cal, CONS, DELTA)
                r = PS[te, j] / C[te]
                rec.append((r.mean(), j > 0, np.mean(r > 1.05), np.mean(r > 1.50), r.max(),
                            (np.mean(r > 1.05) > CONS[0][0]) or (np.mean(r > 1.50) > CONS[1][0])))
            a = np.array(rec, float)
            per_size[str(m)] = dict(
                mean_ratio=float(a[:, 0].mean()),
                share_certified_beyond_classical=float(a[:, 1].mean()),
                share_splits_better_than_classical=float(np.mean(a[:, 0] < 1.0)),
                share_splits_worse_than_classical=float(np.mean(a[:, 0] > 1.0)),
                mean_ratio_when_certified=float(a[a[:, 1] > 0, 0].mean()) if a[:, 1].any() else float("nan"),
                worst=float(a[:, 4].mean()),
                violation_frequency=float(a[:, 5].mean()))
        out[name] = per_size

    s = [out["knet"][str(m)]["share_splits_better_than_classical"] for m in CAL_SIZES]
    out["H4_more_calibration_helps_pass"] = bool(all(b >= a for a, b in zip(s, s[1:])) and s[-1] > s[0])
    out["H5_validity_retained_pass"] = bool(all(
        out[nm][str(m)]["violation_frequency"] <= DELTA for nm in ["learnedR", "knet"] for m in CAL_SIZES))
    json.dump(out, open(f"{R}/calibration_size.json", "w"), indent=2)

    print(f"CALIBRATION-SIZE ANALYSIS on the {N} fresh confirmatory trajectories, "
          f"{N_SPLITS} splits, test size {N_TEST}")
    for name in ["learnedR", "knet"]:
        print(f"\n-- {name}")
        for m in CAL_SIZES:
            v = out[name][str(m)]
            print(f"   n_cal {m:>3}  ratio {v['mean_ratio']:.3f}  certified beyond classical "
                  f"{v['share_certified_beyond_classical']*100:5.1f}%  better than classical "
                  f"{v['share_splits_better_than_classical']*100:5.1f}%  worse {v['share_splits_worse_than_classical']*100:4.1f}%"
                  f"  worst {v['worst']:5.2f}x  budget violated {v['violation_frequency']*100:4.1f}%")
    print(f"\n   H4 more calibration helps: {'PASS' if out['H4_more_calibration_helps_pass'] else 'FAIL'}")
    print(f"   H5 validity retained at every size: {'PASS' if out['H5_validity_retained_pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
