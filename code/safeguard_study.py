"""
safeguard_study.py
------------------
Focused experiment: do the structural safeguards reduce TAIL risk?

Motivation (decided before looking at test results): the main run selected the
safeguard configuration by mean validation RMSE, on a validation set that turned
out to under-sample the heavy tail. Selection therefore switched the safeguards
OFF (cfg = [9.0, None, None]) and the learned filter kept a 3.15x worst-case
blow-up versus the classical EKF -- while the non-learned classical IAE stayed at
1.11x.

Mean RMSE is the wrong selection criterion for a safety property. Here the
safeguard configuration is selected on validation by a TAIL criterion (90th
percentile of the per-trajectory RMSE ratio vs the classical EKF), then reported
on the untouched test set for both mean and tail.
"""
import json
import numpy as np
import torch

from simulate import make_dataset
from ekf import run_ekf
from ankf import DegradationDetector, make_classical_iae_fn

DUR = 90.0
VAL_SEEDS = list(range(100, 120))
TEST_SEEDS = list(range(200, 240))


def load_cfg():
    with open("../results/results.json") as f:
        s = json.load(f)
    det = DegradationDetector()
    det.load_state_dict(torch.load("../results/ankf_detector.pt"))
    det.eval()
    return s, det


def rmse_of(out, ref):
    return float(np.sqrt(np.mean((out["x"] - ref["x"]) ** 2 + (out["y"] - ref["y"]) ** 2)))


def ratios_for(data, qcfg, make_fn):
    """Per-trajectory RMSE ratio vs the classical fixed-R EKF."""
    out = []
    for ref, sens in data:
        n = len(ref["t"])
        base = rmse_of(run_ekf(sens, n, **qcfg), ref)
        adapted = rmse_of(run_ekf(sens, n, r_adapt_fn=make_fn(), **qcfg), ref)
        out.append(adapted / base)
    return np.array(out)


def main():
    summary, det = load_cfg()
    qcfg = {"q_accel": summary["tuned_Q"]["q_accel"], "q_gyro": summary["tuned_Q"]["q_gyro"]}
    vi_sel = summary["cfg"]["v2"][0]

    val = [make_dataset(seed=s, duration_s=DUR) for s in VAL_SEEDS]
    test = [make_dataset(seed=s, duration_s=DUR) for s in TEST_SEEDS]

    candidates = []
    for vi in [4.0, 9.0]:
        for tg in [None, 1.5, 2.0, 2.5]:
            for mc in [None, 2, 3, 6]:
                candidates.append((vi, tg, mc))

    print(f"Selecting safeguard config on VALIDATION by tail criterion (p90 ratio); "
          f"{len(candidates)} candidates")
    best, best_score = None, float("inf")
    for vi, tg, mc in candidates:
        r = ratios_for(val, qcfg, lambda vi=vi, tg=tg, mc=mc: det.make_safeguarded_fn(vi, 1.0, tg, mc))
        score = float(np.percentile(r, 90))
        if score < best_score:
            best_score, best = score, (vi, tg, mc)
    print(f"  selected {best}  (val p90 ratio {best_score:.3f})")

    print("\nEvaluating on untouched TEST set...")
    policies = {
        "learned, no safeguard": lambda: det.make_r_scale_fn(vi_sel, 1.0),
        "learned + safeguard (tail-selected)": lambda: det.make_safeguarded_fn(best[0], 1.0, best[1], best[2]),
        "classical IAE (no ML)": lambda: make_classical_iae_fn(summary["cfg"]["iae"][0]),
    }

    print(f"\n{'Policy':<38}{'mean ratio':>12}{'p90':>8}{'worst':>8}{'wins':>8}")
    out_rows = {}
    for name, fn in policies.items():
        r = ratios_for(test, qcfg, fn)
        out_rows[name] = dict(mean_ratio=float(r.mean()), p90=float(np.percentile(r, 90)),
                              worst=float(r.max()), wins=int((r < 1).sum()), n=len(r))
        print(f"{name:<38}{r.mean():>12.3f}{np.percentile(r,90):>8.2f}{r.max():>8.2f}{(r<1).sum():>5}/{len(r)}")

    with open("../results/safeguard_study.json", "w") as f:
        json.dump(dict(selected_cfg=list(best), val_p90=best_score, test=out_rows), f, indent=2)
    print("\nWrote ../results/safeguard_study.json")


if __name__ == "__main__":
    main()
