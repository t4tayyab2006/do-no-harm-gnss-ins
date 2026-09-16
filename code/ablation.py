"""
ablation.py
-----------
Ablation of the RESET mechanism on the pass-through setting (perfect-R component,
kappa = 2 quality-aware classical fallback, sep_m gate). Without reset, the gate
only switches the OUTPUT; the component keeps running from its own (possibly
diverged) state and can re-enter the divergence loop.

Writes results/ablation_noreset.npz (same pool and thresholds as passthrough.npz).
"""
import json
import multiprocessing as mp
import numpy as np

import experiment_quality as E
import experiment_passthrough as P


def _job(seed):
    from simulate import make_dataset
    from gated import run_gated
    ref, sens = make_dataset(seed=seed, duration_s=E.DUR, kappa=P.KAPPA)
    n = len(ref["t"])
    out = []
    for lam in E.LAMBDAS:
        g = run_gated(sens, n, P._W["q"], None, P._W["C"], lam, reset=False,
                      stat=E.STAT, learned_is_oracle=True)
        out.append(E._rmse(g["gated"], ref))
    return out


if __name__ == "__main__":
    cfg = json.load(open(f"../results/quality_k{P.KAPPA}.json"))["config"]
    with mp.Pool(mp.cpu_count() - 1, initializer=P._init, initargs=(cfg,)) as pool:
        G = np.array(pool.map(_job, E.POOL_SEEDS, chunksize=4))
    np.savez("../results/ablation_noreset.npz", lambdas=np.array(E.LAMBDAS), gated=G)
    C = np.load("../results/passthrough.npz")["classical"]
    for i, l in enumerate(E.LAMBDAS):
        r = G[:, i] / C
        print(f"  no-reset lambda {l:>6}: mean ratio {r.mean():.3f}  harm5 {np.mean(r>1.05)*100:5.1f}%  worst {r.max():.2f}x")
