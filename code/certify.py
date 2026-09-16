"""
certify.py
----------
Learn-then-Test (LTT) calibration of the solution-separation gate, with a
finite-sample, distribution-free guarantee on HARM relative to the classical
fallback.

Harm on a trajectory:  RMSE_gated > (1 + EPS) * RMSE_classical
Guarantee:             with probability >= 1 - DELTA over the calibration draw,
                       the population harm probability of the selected
                       threshold is <= ALPHA.

Why LTT and not a Gaussian threshold: classical solution-separation monitors
set thresholds from assumed Gaussian statistics with correct covariances. A
learned covariance is exactly what breaks that assumption. LTT needs only that
trajectories are exchangeable (i.i.d. episodes), and is valid for ANY score.

Procedure (fixed-sequence testing, controls family-wise error at DELTA):
  thresholds ordered conservative -> aggressive (lambda = 0 is pure classical,
  so its harm is exactly 0 and it is always certifiable -- the method can
  always fall back). For each lambda in order, test H0: harm(lambda) > ALPHA
  with an exact binomial p-value; certify while p <= DELTA; stop at the first
  failure. Deploy the certified lambda with the lowest calibration RMSE.

Usage:
  python certify.py compute   # ~10 min on 12 cores: builds results/certify_matrix.npz
  python certify.py analyze   # repeated random splits -> results/certify.json
"""
import sys
import json
import math
import multiprocessing as mp
import numpy as np

POOL_SEEDS = list(range(1000, 1600))   # 600 trajectories, disjoint from train/val/test
LAMBDAS = [0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, np.inf]
DUR = 90.0

_W = {}


def _init_worker():
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
    from ekf import run_ekf
    from gated import run_gated
    q, L, C = _W["q"], _W["L"], _W["C"]
    ref, sens = make_dataset(seed=seed, duration_s=DUR)
    n = len(ref["t"])

    def r(px, py):
        return float(np.sqrt(np.mean((px - ref["x"]) ** 2 + (py - ref["y"]) ** 2)))

    row = {"seed": seed}
    e = run_ekf(sens, n, **q)
    row["fixed"] = r(e["x"], e["y"])
    o = run_ekf(sens, n, r_gps_base=None, **q)
    row["oracle"] = r(o["x"], o["y"])

    reset_rmse, noreset_rmse, open_frac = [], [], []
    for lam in LAMBDAS:
        g = run_gated(sens, n, q, L, C, lam, reset=True)
        reset_rmse.append(r(g["gated"][:, 0], g["gated"][:, 1]))
        open_frac.append(g["open_frac"])
        if lam == 0.0:
            row["classical"] = r(g["classical"][:, 0], g["classical"][:, 1])
        if np.isinf(lam):
            row["learned"] = r(g["learned"][:, 0], g["learned"][:, 1])
            noreset_rmse.append(reset_rmse[-1])  # identical: gate never closes
        elif lam == 0.0:
            noreset_rmse.append(reset_rmse[-1])  # identical: always classical
        else:
            g2 = run_gated(sens, n, q, L, C, lam, reset=False)
            noreset_rmse.append(r(g2["gated"][:, 0], g2["gated"][:, 1]))
    row["gated_reset"] = reset_rmse
    row["gated_noreset"] = noreset_rmse
    row["open_frac"] = open_frac
    return row


def compute():
    print(f"Computing {len(POOL_SEEDS)} trajectories x {len(LAMBDAS)} thresholds "
          f"(reset + no-reset) on {mp.cpu_count() - 1} workers...", flush=True)
    with mp.Pool(mp.cpu_count() - 1, initializer=_init_worker) as pool:
        rows = []
        for i, row in enumerate(pool.imap_unordered(_one, POOL_SEEDS, chunksize=4)):
            rows.append(row)
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(POOL_SEEDS)}", flush=True)
    rows.sort(key=lambda r: r["seed"])
    np.savez("../results/certify_matrix.npz",
             seeds=np.array([r["seed"] for r in rows]),
             lambdas=np.array(LAMBDAS),
             fixed=np.array([r["fixed"] for r in rows]),
             oracle=np.array([r["oracle"] for r in rows]),
             classical=np.array([r["classical"] for r in rows]),
             learned=np.array([r["learned"] for r in rows]),
             gated_reset=np.array([r["gated_reset"] for r in rows]),
             gated_noreset=np.array([r["gated_noreset"] for r in rows]),
             open_frac=np.array([r["open_frac"] for r in rows]))
    print("Wrote ../results/certify_matrix.npz")


# ---------------------------------------------------------------- analysis
def binom_cdf(k, n, p):
    """P(Binomial(n, p) <= k), computed in log space."""
    if k >= n:
        return 1.0
    lp, lq = math.log(p), math.log1p(-p)
    terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq
             for i in range(k + 1)]
    m = max(terms)
    return min(1.0, math.exp(m) * sum(math.exp(t - m) for t in terms))


def ltt_select(G, C, idx, alpha, delta, eps):
    """Fixed-sequence LTT over thresholds (columns of G, conservative first).
    Returns index of the deployed threshold."""
    n = len(idx)
    certified = []
    for j in range(G.shape[1]):
        k = int(np.sum(G[idx, j] > (1 + eps) * C[idx]))
        if binom_cdf(k, n, alpha) <= delta:
            certified.append(j)
        else:
            break
    if not certified:           # cannot happen: lambda=0 has zero harm, but be safe
        return 0
    return min(certified, key=lambda j: G[idx, j].mean())


def ltt_select_multi(G, C, idx, constraints, delta):
    """Multi-risk LTT. constraints = [(alpha_1, eps_1), (alpha_2, eps_2), ...]:
    certify a threshold only if EVERY risk P(RMSE_g > (1+eps_i) RMSE_C) <= alpha_i
    is rejected at level delta/m (Bonferroni within each step, fixed-sequence
    across thresholds), so family-wise error stays <= delta.

    Motivation: a single alpha bounds how OFTEN harm occurs, not how BAD it is.
    Pairing a frequency constraint (e.g. >5% worse on <= 20%) with a severity
    constraint (e.g. >50% worse on <= 2%) stops a large alpha from readmitting
    the blow-ups."""
    n = len(idx)
    m = len(constraints)
    certified = []
    for j in range(G.shape[1]):
        ok = all(binom_cdf(int(np.sum(G[idx, j] > (1 + e) * C[idx])), n, a) <= delta / m
                 for a, e in constraints)
        if ok:
            certified.append(j)
        else:
            break
    if not certified:
        return 0
    return min(certified, key=lambda j: G[idx, j].mean())


def analyze(n_splits=200, n_cal=300, alpha=0.10, delta=0.10, eps=0.05, seed=0):
    Z = np.load("../results/certify_matrix.npz")
    C, Lr, F, O = Z["classical"], Z["learned"], Z["fixed"], Z["oracle"]
    lambdas = Z["lambdas"]
    N = len(C)
    rng = np.random.default_rng(seed)

    def stats(pred, idx):
        ratio = pred[idx] / C[idx]
        return dict(mean_rmse=float(pred[idx].mean()),
                    mean_ratio_vs_classical=float(ratio.mean()),
                    harm_rate=float(np.mean(pred[idx] > (1 + eps) * C[idx])),
                    worst_ratio=float(ratio.max()),
                    win_rate=float(np.mean(pred[idx] < C[idx])))

    out = {"config": dict(alpha=alpha, delta=delta, eps=eps, n_cal=n_cal,
                          n_test=N - n_cal, n_splits=n_splits, n_pool=N)}
    agg = {k: [] for k in ["certified_reset", "certified_noreset", "naive_reset",
                           "learned", "classical", "fixed", "oracle"]}
    chosen = []
    for _ in range(n_splits):
        perm = rng.permutation(N)
        cal, te = perm[:n_cal], perm[n_cal:]
        j = ltt_select(Z["gated_reset"], C, cal, alpha, delta, eps)
        jn = ltt_select(Z["gated_noreset"], C, cal, alpha, delta, eps)
        jnaive = int(np.argmin(Z["gated_reset"][cal].mean(axis=0)))  # no guarantee
        chosen.append(float(lambdas[j]))
        agg["certified_reset"].append(stats(Z["gated_reset"][:, j], te))
        agg["certified_noreset"].append(stats(Z["gated_noreset"][:, jn], te))
        agg["naive_reset"].append(stats(Z["gated_reset"][:, jnaive], te))
        agg["learned"].append(stats(Lr, te))
        agg["classical"].append(stats(C, te))
        agg["fixed"].append(stats(F, te))
        agg["oracle"].append(stats(O, te))

    summary = {}
    for k, v in agg.items():
        summary[k] = {m: float(np.mean([s[m] for s in v])) for m in v[0]}
        summary[k]["frac_splits_harm_exceeds_alpha"] = float(np.mean([s["harm_rate"] > alpha for s in v]))
    out["summary"] = summary
    out["chosen_lambda_distribution"] = {str(l): chosen.count(float(l)) for l in sorted(set(chosen))}
    out["open_frac_mean_by_lambda"] = {str(l): float(Z["open_frac"][:, i].mean()) for i, l in enumerate(lambdas)}
    return out


def report(res):
    s, c = res["summary"], res["config"]
    print(f"\n=== {c['n_splits']} random splits: {c['n_cal']} calibration / {c['n_test']} test trajectories "
          f"(alpha={c['alpha']}, delta={c['delta']}, eps={c['eps']}) ===")
    print(f"{'Method':<34}{'mean RMSE':>10}{'vs classical':>14}{'harm rate':>11}{'worst':>8}{'P(harm>a)':>11}")
    names = [("Fixed-R EKF (tuned)", "fixed"), ("Classical chi2 IAE [fallback]", "classical"),
             ("Learned adaptive R", "learned"), ("Gate, naive threshold (no cert.)", "naive_reset"),
             ("Gate, certified, no reset", "certified_noreset"),
             ("Gate, certified + reset [PROPOSED]", "certified_reset"), ("Oracle R (bound)", "oracle")]
    for name, k in names:
        v = s[k]
        print(f"{name:<34}{v['mean_rmse']:>10.3f}{(v['mean_ratio_vs_classical']-1)*100:>+13.1f}%"
              f"{v['harm_rate']*100:>10.1f}%{v['worst_ratio']:>7.2f}x{v['frac_splits_harm_exceeds_alpha']*100:>10.1f}%")
    print(f"\nchosen threshold distribution: {res['chosen_lambda_distribution']}")
    print(f"guarantee check: P(test harm > alpha) for PROPOSED = "
          f"{s['certified_reset']['frac_splits_harm_exceeds_alpha']*100:.1f}%  (must be <= delta = {c['delta']*100:.0f}%, "
          f"up to test-set sampling noise)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "compute":
        compute()
    else:
        results = {}
        for a in [0.05, 0.10, 0.20]:
            res = analyze(alpha=a)
            report(res)
            results[f"alpha_{a}"] = res
        with open("../results/certify.json", "w") as f:
            json.dump(results, f, indent=2)
        print("\nWrote ../results/certify.json")
