"""
experiment_quality.py
---------------------
Certified gating as a function of how informative the receiver quality
indicator is (detectability d' = kappa).

For each kappa, all on seeds DISJOINT from the certification pool:
  1. train the learned degradation detector on train seeds (0-29), with the
     indicator as a 6th feature;
  2. tune the classical fallback (chi-square IAE, optionally + quality-threshold
     rule) and the learned filter's inflation on validation seeds (100-119);
  3. on the certification pool (1000-1599) compute, per trajectory: fixed-R EKF,
     oracle, classical fallback, learned, and the gated system over a threshold
     grid. The gate statistic (sep_m, metres) and reset=True were fixed in
     advance on the development set (dev_gate.py) and are not re-tuned here.
  4. Learn-then-Test certification over 200 random calibration/test splits.

Usage:  python experiment_quality.py <kappa>
Writes: results/quality_k<kappa>.npz and results/quality_k<kappa>.json
"""
import sys
import json
import multiprocessing as mp
import numpy as np

from certify import ltt_select

TRAIN_SEEDS = list(range(0, 30))
VAL_SEEDS = list(range(100, 120))
POOL_SEEDS = list(range(1000, 1600))
LAMBDAS = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, np.inf]
STAT = "sep_m"
DUR = 90.0
_W = {}


def _q():
    S = json.load(open("../results/results.json"))
    return {"q_accel": S["tuned_Q"]["q_accel"], "q_gyro": S["tuned_Q"]["q_gyro"]}


def _policy_fns(det, cfg):
    from ankf import make_classical_quality_fn
    L = lambda: det.make_r_scale_fn(cfg["learned_vi"], 1.0)
    C = lambda: make_classical_quality_fn(cfg["classical_vi"], cfg["classical_qthr"])
    return L, C


def _init(kappa, det_path, cfg):
    import torch
    torch.set_num_threads(1)
    from ankf import DegradationDetector
    det = DegradationDetector(in_dim=6)
    det.load_state_dict(torch.load(det_path))
    det.eval()
    _W.update(kappa=kappa, q=_q(), cfg=cfg)
    _W["L"], _W["C"] = _policy_fns(det, cfg)


def _rmse(track, ref):
    return float(np.sqrt(np.mean((track[:, 0] - ref["x"]) ** 2 + (track[:, 1] - ref["y"]) ** 2)))


def _val_job(args):
    """Validation RMSE for one (trajectory, policy) pair."""
    seed, kind, a, b = args
    from simulate import make_dataset
    from gated import run_single
    from ankf import make_classical_quality_fn
    ref, sens = make_dataset(seed=seed, duration_s=DUR, kappa=_W["kappa"])
    n = len(ref["t"])
    if kind == "classical":
        fn = lambda: make_classical_quality_fn(a, b)
    else:
        fn = lambda: _W["det"].make_r_scale_fn(a, 1.0)
    return (kind, a, b, seed, _rmse(run_single(sens, n, _W["q"], fn), ref))


def _init_val(kappa, det_path):
    import torch
    torch.set_num_threads(1)
    from ankf import DegradationDetector
    det = DegradationDetector(in_dim=6)
    det.load_state_dict(torch.load(det_path))
    det.eval()
    _W.update(kappa=kappa, q=_q(), det=det)


def _pool_job(seed):
    from simulate import make_dataset
    from gated import run_gated, run_single
    q, L, C = _W["q"], _W["L"], _W["C"]
    ref, sens = make_dataset(seed=seed, duration_s=DUR, kappa=_W["kappa"])
    n = len(ref["t"])
    row = {"fixed": _rmse(run_single(sens, n, q), ref),
           "oracle": _rmse(run_single(sens, n, q, oracle=True), ref)}
    gated = []
    for lam in LAMBDAS:
        g = run_gated(sens, n, q, L, C, lam, reset=True, stat=STAT)
        gated.append(_rmse(g["gated"], ref))
        if lam == 0.0:
            row["classical"] = _rmse(g["classical"], ref)
        if np.isinf(lam):
            row["learned"] = _rmse(g["learned"], ref)
    row["gated"] = gated
    return row


def train_and_tune(kappa):
    import torch
    from simulate import make_dataset
    from gated import run_single
    from ankf import train_detector
    q = _q()

    def fix_data(seeds):
        X, Y = [], []
        for s in seeds:
            ref, sens = make_dataset(seed=s, duration_s=DUR, kappa=kappa)
            _, feats = run_single(sens, len(ref["t"]), q, collect=True)
            for k, f in feats:
                X.append(f); Y.append(float(sens["outage_mask_full"][k]))
        return np.array(X, np.float32), np.array(Y, np.float32)

    print(f"[kappa={kappa}] training detector (6 features incl. quality indicator)...", flush=True)
    Xtr, Ytr = fix_data(TRAIN_SEEDS)
    Xva, Yva = fix_data(VAL_SEEDS)
    torch.manual_seed(0)
    det = train_detector(Xtr, Ytr, Xva, Yva, verbose=False)
    with torch.no_grad():
        p = torch.sigmoid(det(torch.tensor(Xva))).numpy()
    acc = float(np.mean((p > 0.5) == (Yva > 0.5)))
    print(f"  detector val accuracy {acc:.3f} ({int(Yva.sum())} degraded / {len(Yva)} fixes)", flush=True)
    det_path = f"../results/detector_k{kappa}.pt"
    torch.save(det.state_dict(), det_path)

    jobs = [(s, "classical", vi, qt) for vi in [4.0, 9.0, 16.0]
            for qt in [np.inf, 0.5, 1.0, 1.5, 2.0, 3.0] for s in VAL_SEEDS]
    jobs += [(s, "learned", vi, 0.0) for vi in [4.0, 9.0, 16.0] for s in VAL_SEEDS]
    with mp.Pool(mp.cpu_count() - 1, initializer=_init_val, initargs=(kappa, det_path)) as pool:
        res = pool.map(_val_job, jobs, chunksize=8)
    agg = {}
    for kind, a, b, _, r in res:
        agg.setdefault((kind, a, b), []).append(r)
    means = {k: float(np.mean(v)) for k, v in agg.items()}
    bc = min((k for k in means if k[0] == "classical"), key=means.get)
    bl = min((k for k in means if k[0] == "learned"), key=means.get)
    cfg = dict(kappa=kappa, detector_val_acc=acc,
               classical_vi=bc[1], classical_qthr=bc[2], classical_val_rmse=means[bc],
               learned_vi=bl[1], learned_val_rmse=means[bl])
    print(f"  classical fallback: vi={bc[1]}, q_thr={bc[2]}  val RMSE {means[bc]:.3f}", flush=True)
    print(f"  learned:            vi={bl[1]}              val RMSE {means[bl]:.3f}", flush=True)
    return det_path, cfg


def analyze(Z, n_splits=200, n_cal=300, seed=0):
    C, G = Z["classical"], Z["gated"]
    N = len(C)
    rng = np.random.default_rng(seed)
    table = {}
    for eps in [0.05, 0.10]:
        for alpha in [0.05, 0.10, 0.20]:
            recs, lams = [], []
            for _ in range(n_splits):
                perm = rng.permutation(N)
                cal, te = perm[:n_cal], perm[n_cal:]
                j = ltt_select(G, C, cal, alpha, 0.10, eps)
                lams.append(float(LAMBDAS[j]))
                g = G[te, j]
                recs.append(dict(ratio=float(np.mean(g / C[te])), harm=float(np.mean(g > (1 + eps) * C[te])),
                                 worst=float(np.max(g / C[te])), rmse=float(g.mean())))
            table[f"eps{eps}_alpha{alpha}"] = dict(
                mean_ratio=float(np.mean([r["ratio"] for r in recs])),
                mean_rmse=float(np.mean([r["rmse"] for r in recs])),
                harm=float(np.mean([r["harm"] for r in recs])),
                worst=float(np.mean([r["worst"] for r in recs])),
                p_harm_exceeds_alpha=float(np.mean([r["harm"] > alpha for r in recs])),
                median_lambda=float(np.median(lams)))
    base = {}
    for k in ["fixed", "classical", "learned", "oracle"]:
        v = Z[k]
        base[k] = dict(mean_rmse=float(v.mean()), mean_ratio=float(np.mean(v / C)),
                       harm5=float(np.mean(v > 1.05 * C)), harm10=float(np.mean(v > 1.10 * C)),
                       worst=float(np.max(v / C)))
    return dict(baselines=base, certified=table)


def main(kappa):
    det_path, cfg = train_and_tune(kappa)
    print(f"[kappa={kappa}] computing {len(POOL_SEEDS)} pool trajectories x {len(LAMBDAS)} thresholds...", flush=True)
    with mp.Pool(mp.cpu_count() - 1, initializer=_init, initargs=(kappa, det_path, cfg)) as pool:
        rows = pool.map(_pool_job, POOL_SEEDS, chunksize=4)
    Z = {k: np.array([r[k] for r in rows]) for k in ["fixed", "oracle", "classical", "learned", "gated"]}
    np.savez(f"../results/quality_k{kappa}.npz", lambdas=np.array(LAMBDAS), **Z)
    out = dict(config=cfg, **analyze(Z))
    json.dump(out, open(f"../results/quality_k{kappa}.json", "w"), indent=2)

    b = out["baselines"]
    print(f"\n[kappa={kappa}] pool baselines (ratio vs classical fallback; harm = >5% worse):")
    for k in ["fixed", "classical", "learned", "oracle"]:
        print(f"  {k:<10} RMSE {b[k]['mean_rmse']:7.3f}  ratio {b[k]['mean_ratio']:.3f}  "
              f"harm5 {b[k]['harm5']*100:5.1f}%  worst {b[k]['worst']:.2f}x")
    print(f"[kappa={kappa}] certified gate, 200 splits (delta=0.10):")
    for key, v in out["certified"].items():
        print(f"  {key:<16} ratio {v['mean_ratio']:.3f}  harm {v['harm']*100:5.1f}%  worst {v['worst']:.2f}x  "
              f"P(harm>alpha) {v['p_harm_exceeds_alpha']*100:5.1f}%  median lambda {v['median_lambda']}")


if __name__ == "__main__":
    main(float(sys.argv[1]))
