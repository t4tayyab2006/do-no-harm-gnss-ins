"""
benchmark_pred.py
-----------------
Follow-up to benchmark.py, added AFTER its results were seen, and labelled as such
in the manuscript. benchmark.py showed that the IMM-style predictive switch (which
filter predicted the recent GNSS fixes better) captures more gain than the
pre-specified metre-scale gate when the learned filter is good (simulated
KalmanNet), but has no guarantee and fails badly when it is not (real KalmanNet:
37% of segments harmed). Because Learn-then-Test is agnostic to the statistic, the
predictive statistic can itself be certified ("pred_switch" in gated.py).

To keep the choice of statistic off the certification data, it is made per learned
component on DEVELOPMENT data only, with a rule fixed before running this script:

  candidates: sep_m (metre-scale separation, reset)  |  pred_switch (no reset)
  rule:       two-level certificate (domain constraints as in benchmark.py), 200
              random 50/50 splits of the development set; choose the statistic with
              the lower mean test RMSE ratio; tie -> sep_m.
  dev sets:   simulation seeds 2000-2099 (never in the pool); real: Nagoya drive 3
              (the tuning drive; never in the Tokyo pool).

The pool results of BOTH statistics are reported, plus the dev-selected one.

Usage:
  python benchmark_pred.py dev-sim
  python benchmark_pred.py pool-sim 0 2   (and 1 2)
  python benchmark_pred.py real
  python benchmark_pred.py analyze        -> results/benchmark_pred.json
"""
import sys
import json
import time
import multiprocessing as mp
import numpy as np

import benchmark as B

R = "../results"
PS_GRID = [-np.inf, -8.0, -4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0, 8.0, np.inf]  # conservative -> aggressive
_W = B._W


def _grids(args):
    """pred_switch grid (and optionally the sep_m grid) for learned-R and KalmanNet."""
    from gated import run_gated
    ref, sens, with_sep = args
    q, L, C = _W["q"], _W["L"], _W["C"]
    n = len(ref["t"])
    kf = B._knet_factory()
    out = {"classical": B.rmse(run_gated(sens, n, q, L, C, 0.0)["classical"], ref)}
    for name, fac in [("learnedR", None), ("knet", kf)]:
        out[f"{name}_ps"] = [B.rmse(run_gated(sens, n, q, L, C, lam, reset=False, stat="pred_switch",
                                              make_learned_filter=fac)["gated"], ref) for lam in PS_GRID]
        if with_sep:
            out[f"{name}_sep"] = [B.rmse(run_gated(sens, n, q, L, C, lam, reset=True, stat="sep_m",
                                                   make_learned_filter=fac)["gated"], ref) for lam in B.LAMBDAS]
    return out


def run_pool(pairs, setup, with_sep):
    with mp.Pool(mp.cpu_count() - 1, initializer=B._init, initargs=(setup,)) as pool:
        return pool.map(_grids, [(r, s, with_sep) for r, s in pairs], chunksize=2)


def dev_sim():
    from simulate import make_dataset
    setup = json.load(open(f"{R}/bench_sim_setup.json"))
    pairs = [make_dataset(seed=s, kappa=B.KAPPA) for s in range(2000, 2100)]
    t0 = time.time()
    rows = run_pool(pairs, setup, with_sep=True)
    json.dump(rows, open(f"{R}/bench_pred_dev_sim.json", "w"))
    print(f"[dev-sim] {len(rows)} trajectories in {time.time()-t0:.0f}s")


def pool_sim(chunk, nchunks):
    from simulate import make_dataset
    setup = json.load(open(f"{R}/bench_sim_setup.json"))
    seeds = [s for i, s in enumerate(range(1000, 1600)) if i % nchunks == chunk]
    t0 = time.time()
    rows = run_pool([make_dataset(seed=s, kappa=B.KAPPA) for s in seeds], setup, with_sep=False)
    for r, s in zip(rows, seeds):
        r["seed"] = s
    json.dump(rows, open(f"{R}/bench_pred_sim_chunk{chunk}.json", "w"))
    print(f"[pool-sim] chunk {chunk}: {len(rows)} trajectories in {time.time()-t0:.0f}s")


def real():
    from realdata import segments, load_run
    setup = json.load(open(f"{R}/bench_real_setup.json"))
    dev = segments(load_run("nagoya", 3), 53_000, B.KAPPA)
    tokyo = []
    for run in (1, 2, 3):
        tokyo += segments(load_run("tokyo", run), 90_000 + 1000 * run, B.KAPPA)
    t0 = time.time()
    json.dump(run_pool(dev, setup, with_sep=True), open(f"{R}/bench_pred_dev_real.json", "w"))
    json.dump(run_pool(tokyo, setup, with_sep=False), open(f"{R}/bench_pred_real_rows.json", "w"))
    print(f"[real] dev {len(dev)} + pool {len(tokyo)} segments in {time.time()-t0:.0f}s")


def certified_stats(G, C, cons, seed=0, n_splits=200):
    from certify import ltt_select_multi
    rng = np.random.default_rng(seed)
    N = len(C)
    rec = []
    for _ in range(n_splits):
        p = rng.permutation(N); cal, te = p[:N // 2], p[N // 2:]
        j = ltt_select_multi(G, C, cal, cons, B.DELTA)
        r = G[te, j] / C[te]
        rec.append((r.mean(), np.mean(r > 1.10), np.mean(r > 1.50), r.max(), np.mean(r > 1.05) > cons[0][0],
                    np.mean(r > 1.50) > cons[1][0], G[te, j].mean(), j))
    a = np.array(rec, float)
    return dict(mean_ratio=float(a[:, 0].mean()), harm10=float(a[:, 1].mean()), severe50=float(a[:, 2].mean()),
                worst=float(a[:, 3].mean()), viol_freq=float(a[:, 4].mean()), viol_sev=float(a[:, 5].mean()),
                mean_rmse=float(a[:, 6].mean()), median_index=int(np.median(a[:, 7])))


def analyze():
    out = {}
    for dom in ["sim", "real"]:
        cons = B.TWO_LEVEL[dom]
        dev = json.load(open(f"{R}/bench_pred_dev_{dom}.json"))
        Cd = np.array([r["classical"] for r in dev])
        if dom == "sim":
            pool = json.load(open(f"{R}/bench_pred_sim_chunk0.json")) + json.load(open(f"{R}/bench_pred_sim_chunk1.json"))
            pool.sort(key=lambda r: r["seed"])
            base = json.load(open(f"{R}/bench_sim_chunk0.json")) + json.load(open(f"{R}/bench_sim_chunk1.json"))
            base.sort(key=lambda r: r["seed"])
            sep = {"learnedR": np.load(f"{R}/quality_k{B.KAPPA}.npz")["gated"],
                   "knet": np.array([r["knet_gated"] for r in base])}
        else:
            pool = json.load(open(f"{R}/bench_pred_real_rows.json"))
            base = json.load(open(f"{R}/bench_real_rows.json"))
            sep = {"learnedR": np.load(f"{R}/real_learned_k{B.KAPPA:g}.npz")["gated"],
                   "knet": np.array([r["knet_gated"] for r in base])}
        Cp = np.array([r["classical"] for r in pool])
        assert np.allclose(Cp, [r["classical"] for r in base]), f"{dom}: pools differ"
        out[dom] = {}
        for name in ["learnedR", "knet"]:
            d_sep = certified_stats(np.array([r[f"{name}_sep"] for r in dev]), Cd, cons)
            d_ps = certified_stats(np.array([r[f"{name}_ps"] for r in dev]), Cd, cons)
            choice = "pred_switch" if d_ps["mean_ratio"] < d_sep["mean_ratio"] else "sep_m"
            p_sep = certified_stats(sep[name], Cp, cons)
            p_ps = certified_stats(np.array([r[f"{name}_ps"] for r in pool]), Cp, cons)
            out[dom][name] = dict(dev_sep=d_sep, dev_ps=d_ps, choice=choice, pool_sep=p_sep, pool_ps=p_ps,
                                  pool_selected=p_ps if choice == "pred_switch" else p_sep)
            print(f"\n[{dom}] {name}: dev certified mean ratio  sep_m {d_sep['mean_ratio']:.3f}  "
                  f"pred_switch {d_ps['mean_ratio']:.3f}  -> selected {choice}")
            for lab, v in [("pool, certified sep_m", p_sep), ("pool, certified pred_switch", p_ps)]:
                print(f"   {lab:<30} ratio {v['mean_ratio']:.3f}  >10% {v['harm10']*100:4.1f}%  >50% {v['severe50']*100:4.1f}%"
                      f"  worst {v['worst']:.2f}x  viol(freq) {v['viol_freq']*100:4.1f}%  viol(sev) {v['viol_sev']*100:4.1f}%")
    json.dump(out, open(f"{R}/benchmark_pred.json", "w"), indent=2)
    print("\nWrote ../results/benchmark_pred.json")


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"dev-sim": dev_sim, "real": real, "analyze": analyze}.get(cmd, lambda: None)()
    if cmd == "pool-sim":
        pool_sim(int(sys.argv[2]), int(sys.argv[3]))
