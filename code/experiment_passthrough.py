"""
experiment_passthrough.py
-------------------------
Pass-through test: when the gated component GENUINELY has gains, does the
certified gate keep them while removing its tail?

The component is the oracle-R filter -- the limit of a perfect learned noise
predictor. It is 8.7% better than the strongest classical fallback on average,
yet harms 15.3% of trajectories with a 3.48x worst case (Q mismatch and IMU
drift make "correct R" not MSE-optimal everywhere). So tail risk is not only a
learning failure; even a perfect covariance predictor needs certification.

Fallback: the quality-aware classical rule tuned at kappa = 2 (the strongest
classical baseline). Gate: sep_m + reset, fixed in advance on the dev set.
Same pool (1000-1599) and the same LTT analysis as experiment_quality.py.
"""
import json
import multiprocessing as mp
import numpy as np

import experiment_quality as E

KAPPA = 2.0
_W = {}


def _init(cfg):
    from ankf import make_classical_quality_fn
    _W["q"] = E._q()
    _W["C"] = lambda: make_classical_quality_fn(cfg["classical_vi"], cfg["classical_qthr"])


def _job(seed):
    from simulate import make_dataset
    from gated import run_gated, run_single
    q, C = _W["q"], _W["C"]
    ref, sens = make_dataset(seed=seed, duration_s=E.DUR, kappa=KAPPA)
    n = len(ref["t"])
    row = {"fixed": E._rmse(run_single(sens, n, q), ref)}
    gated, opens = [], []
    for lam in E.LAMBDAS:
        g = run_gated(sens, n, q, None, C, lam, reset=True, stat=E.STAT, learned_is_oracle=True)
        gated.append(E._rmse(g["gated"], ref)); opens.append(g["open_frac"])
        if lam == 0.0:
            row["classical"] = E._rmse(g["classical"], ref)
        if np.isinf(lam):
            row["oracle"] = row["learned"] = E._rmse(g["learned"], ref)
    row["gated"], row["open"] = gated, opens
    return row


if __name__ == "__main__":
    cfg = json.load(open(f"../results/quality_k{KAPPA}.json"))["config"]
    print(f"Fallback: classical quality rule vi={cfg['classical_vi']}, q_thr={cfg['classical_qthr']}", flush=True)
    with mp.Pool(mp.cpu_count() - 1, initializer=_init, initargs=(cfg,)) as pool:
        rows = pool.map(_job, E.POOL_SEEDS, chunksize=4)
    Z = {k: np.array([r[k] for r in rows]) for k in ["fixed", "oracle", "classical", "learned", "gated", "open"]}
    np.savez("../results/passthrough.npz", lambdas=np.array(E.LAMBDAS), **Z)
    out = E.analyze(Z)
    out["open_frac_by_lambda"] = {str(l): float(Z["open"][:, i].mean()) for i, l in enumerate(E.LAMBDAS)}
    json.dump(out, open("../results/passthrough.json", "w"), indent=2)

    C = Z["classical"]
    gain_avail = 1 - np.mean(Z["oracle"] / C)
    print(f"\nperfect-R component vs classical: ratio {np.mean(Z['oracle']/C):.3f}  "
          f"harm5 {np.mean(Z['oracle'] > 1.05*C)*100:.1f}%  worst {np.max(Z['oracle']/C):.2f}x")
    print("per-threshold (all 600): lambda, gate-open %, mean ratio, harm5, worst")
    for i, l in enumerate(E.LAMBDAS):
        r = Z["gated"][:, i] / C
        print(f"  {l:>6}  {Z['open'][:, i].mean()*100:5.1f}%  {r.mean():.3f}  {np.mean(r > 1.05)*100:5.1f}%  {r.max():.2f}x")
    print("\ncertified (200 splits, delta=0.10):")
    for key, v in out["certified"].items():
        kept = (1 - v["mean_ratio"]) / gain_avail * 100
        print(f"  {key:<16} ratio {v['mean_ratio']:.3f} (keeps {kept:5.1f}% of available gain)  harm {v['harm']*100:5.1f}%  "
              f"worst {v['worst']:.2f}x  P(harm>alpha) {v['p_harm_exceeds_alpha']*100:5.1f}%  median lambda {v['median_lambda']}")
