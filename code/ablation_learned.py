"""
ablation_learned.py
-------------------
Reset ablation for the LEARNED detector at a given kappa (default 3, the case
with the 16.7x tail). ablation.py showed the reset does NOT help the perfect-R
component -- which has no innovation-driven feedback loop to break. The reset is
motivated by that loop, so its proper test is a component that has it.

Usage: python ablation_learned.py [kappa]   ->  results/ablation_noreset_k<kappa>.npz
"""
import sys
import json
import multiprocessing as mp
import numpy as np

import experiment_quality as E


def _job(seed):
    from simulate import make_dataset
    from gated import run_gated
    ref, sens = make_dataset(seed=seed, duration_s=E.DUR, kappa=E._W["kappa"])
    n = len(ref["t"])
    return [E._rmse(run_gated(sens, n, E._W["q"], E._W["L"], E._W["C"], lam,
                              reset=False, stat=E.STAT)["gated"], ref) for lam in E.LAMBDAS]


if __name__ == "__main__":
    kappa = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
    cfg = json.load(open(f"../results/quality_k{kappa}.json"))["config"]
    with mp.Pool(mp.cpu_count() - 1, initializer=E._init,
                 initargs=(kappa, f"../results/detector_k{kappa}.pt", cfg)) as pool:
        NR = np.array(pool.map(_job, E.POOL_SEEDS, chunksize=4))
    np.savez(f"../results/ablation_noreset_k{kappa}.npz", lambdas=np.array(E.LAMBDAS), gated=NR)
    Z = np.load(f"../results/quality_k{kappa}.npz")
    C, G = Z["classical"], Z["gated"]
    print(f"learned detector, kappa={kappa}: reset vs NO reset (all 600 pool trajectories)")
    print(f"{'lambda':>7} | {'reset: ratio  harm5  worst':>28} | {'no-reset: ratio  harm5  worst':>31}")
    for i, l in enumerate(E.LAMBDAS):
        a, b = G[:, i] / C, NR[:, i] / C
        print(f"{l:>7} | {a.mean():>12.3f} {np.mean(a>1.05)*100:>5.1f}% {a.max():>6.2f}x |"
              f" {b.mean():>15.3f} {np.mean(b>1.05)*100:>5.1f}% {b.max():>6.2f}x")
