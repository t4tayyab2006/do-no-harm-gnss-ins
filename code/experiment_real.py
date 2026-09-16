"""
experiment_real.py
------------------
Certified gating on REAL IMU data (PPC-Dataset), cross-city.

PRE-SPECIFIED DESIGN (fixed before any Tokyo result was seen):
  * Nagoya runs 1-2: detector training. Nagoya run 3: tuning of process noise Q
    (q_accel, q_gyro, q_bias), the classical fallback and the learned inflation.
  * Tokyo runs 1-3 (242 segments): certification pool, never used for tuning.
    200 random 50/50 calibration/test splits (121 / 121).
  * Gate: sep_m + reset, threshold grid as in the simulation study.
  * Certificates, delta = 0.10:
      single     alpha = 0.10, eps = 0.10
      two-level  (alpha 0.20, eps 0.05) AND (alpha 0.05, eps 0.50)
    The severity budget is 5% rather than the simulation's 2% because with 121
    calibration trajectories a 2% bound is not certifiable even with zero observed
    severe cases (P(Bin(121, 0.02) = 0) = 0.087 > delta/2).
  * Indicator informativeness d' in {0, 3}; perfect-R pass-through with the d'=3
    classical fallback.

Usage: python experiment_real.py      -> results/real_*.npz, results/real_summary.json
"""
import json
import multiprocessing as mp
import numpy as np

from certify import ltt_select, ltt_select_multi

LAMBDAS = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, np.inf]
KAPPAS = [0.0, 3.0]
N_SPLITS, DELTA = 200, 0.10
SINGLE = (0.10, 0.10)
TWO_LEVEL = [(0.20, 0.05), (0.05, 0.50)]
_W = {}


def rmse(track, ref):
    return float(np.sqrt(np.mean((track[:, 0] - ref["x"]) ** 2 + (track[:, 1] - ref["y"]) ** 2)))


# ------------------------------------------------------------------ workers
def _init(q, det_state, cfg):
    import torch
    torch.set_num_threads(1)
    _W["q"], _W["cfg"] = q, cfg
    if det_state is not None:
        from ankf import DegradationDetector
        det = DegradationDetector(in_dim=6)
        det.load_state_dict(det_state)
        det.eval()
        _W["det"] = det


def _fns():
    from ankf import make_classical_quality_fn
    cfg = _W["cfg"]
    C = lambda: make_classical_quality_fn(cfg["classical_vi"], cfg["classical_qthr"])
    L = (lambda: _W["det"].make_r_scale_fn(cfg["learned_vi"], 1.0)) if "det" in _W else None
    return L, C


def _eval_job(args):
    """Mean-RMSE job for tuning: args = (ref, sens, kind, params)."""
    from gated import run_single
    from ankf import make_classical_quality_fn
    ref, sens, kind, p = args
    n = len(ref["t"])
    if kind == "Q":
        return rmse(run_single(sens, n, p), ref)
    if kind == "classical":
        return rmse(run_single(sens, n, _W["q"], lambda: make_classical_quality_fn(*p)), ref)
    return rmse(run_single(sens, n, _W["q"], lambda: _W["det"].make_r_scale_fn(p, 1.0)), ref)


def _pool_job(args):
    from gated import run_gated, run_single
    ref, sens, oracle_component = args
    q = _W["q"]
    L, C = _fns()
    n = len(ref["t"])
    row = {"fixed": rmse(run_single(sens, n, q), ref),
           "oracle": rmse(run_single(sens, n, q, oracle=True), ref)}
    g_reset, g_noreset = [], []
    for lam in LAMBDAS:
        a = run_gated(sens, n, q, L, C, lam, reset=True, stat="sep_m", learned_is_oracle=oracle_component)
        g_reset.append(rmse(a["gated"], ref))
        if lam in (0.0, np.inf):
            g_noreset.append(g_reset[-1])
        else:
            b = run_gated(sens, n, q, L, C, lam, reset=False, stat="sep_m", learned_is_oracle=oracle_component)
            g_noreset.append(rmse(b["gated"], ref))
        if lam == 0.0:
            row["classical"] = rmse(a["classical"], ref)
        if np.isinf(lam):
            row["learned"] = rmse(a["learned"], ref)
    row["gated"], row["gated_noreset"] = g_reset, g_noreset
    return row


# ------------------------------------------------------------------ driver
def pmap(fn, jobs, q, det_state=None, cfg=None):
    with mp.Pool(mp.cpu_count() - 1, initializer=_init, initargs=(q, det_state, cfg or {})) as pool:
        return pool.map(fn, jobs, chunksize=4)


def tune_Q(tune):
    grid = [{"q_accel": qa, "q_gyro": qg, "q_bias": qb}
            for qa in [0.05, 0.2, 0.5, 1.0, 2.0] for qg in [0.002, 0.02, 0.1] for qb in [1e-6, 1e-5, 1e-4]]
    jobs = [(r, s, "Q", g) for g in grid for r, s in tune]
    res = np.array(pmap(_eval_job, jobs, grid[0])).reshape(len(grid), len(tune)).mean(axis=1)
    best = grid[int(np.argmin(res))]
    print(f"  tuned Q on Nagoya run 3: {best}  (mean RMSE {res.min():.3f}; worst grid point {res.max():.3f})", flush=True)
    return best


def train_tune(kappa, q):
    import torch
    from realdata import segments, load_run
    from gated import run_single
    from ankf import train_detector
    train = segments(load_run("nagoya", 1), 51_000, kappa) + segments(load_run("nagoya", 2), 52_000, kappa)
    tune = segments(load_run("nagoya", 3), 53_000, kappa)

    def fixes(segs):
        X, Y = [], []
        for ref, sens in segs:
            _, feats = run_single(sens, len(ref["t"]), q, collect=True)
            for k, f in feats:
                X.append(f); Y.append(float(sens["outage_mask_full"][k]))
        return np.array(X, np.float32), np.array(Y, np.float32)

    Xtr, Ytr = fixes(train)
    Xva, Yva = fixes(tune)
    torch.manual_seed(0)
    det = train_detector(Xtr, Ytr, Xva, Yva, verbose=False)
    with torch.no_grad():
        acc = float(np.mean((torch.sigmoid(det(torch.tensor(Xva))).numpy() > 0.5) == (Yva > 0.5)))
    state = {k: v.clone() for k, v in det.state_dict().items()}

    cgrid = [(vi, qt) for vi in [4.0, 9.0, 16.0] for qt in [np.inf, 0.5, 1.0, 1.5, 2.0, 3.0]]
    lgrid = [4.0, 9.0, 16.0]
    cres = np.array(pmap(_eval_job, [(r, s, "classical", c) for c in cgrid for r, s in tune], q, state)
                    ).reshape(len(cgrid), len(tune)).mean(axis=1)
    lres = np.array(pmap(_eval_job, [(r, s, "learned", v) for v in lgrid for r, s in tune], q, state)
                    ).reshape(len(lgrid), len(tune)).mean(axis=1)
    bc, bl = cgrid[int(np.argmin(cres))], lgrid[int(np.argmin(lres))]
    cfg = dict(kappa=kappa, detector_acc=acc, n_train_fixes=len(Ytr), n_degraded=int(Ytr.sum()),
               classical_vi=bc[0], classical_qthr=bc[1], classical_tune_rmse=float(cres.min()),
               learned_vi=bl, learned_tune_rmse=float(lres.min()))
    print(f"  [d'={kappa}] detector acc {acc:.3f}; classical vi={bc[0]} q_thr={bc[1]} ({cres.min():.3f} m); "
          f"learned vi={bl} ({lres.min():.3f} m)", flush=True)
    return state, cfg


def analyse(Z):
    C, G, Lr = Z["classical"], Z["gated"], Z["learned"]
    N = len(C); n_cal = N // 2
    rng = np.random.default_rng(0)
    rec = {"single": [], "two_level": [], "naive": [], "ungated": []}
    for _ in range(N_SPLITS):
        p = rng.permutation(N); cal, te = p[:n_cal], p[n_cal:]
        picks = {"single": ltt_select(G, C, cal, SINGLE[0], DELTA, SINGLE[1]),
                 "two_level": ltt_select_multi(G, C, cal, TWO_LEVEL, DELTA),
                 "naive": int(np.argmin(G[cal].mean(axis=0)))}
        for k, j in picks.items():
            r = G[te, j] / C[te]
            rec[k].append((r.mean(), np.mean(r > 1.05), np.mean(r > 1.10), np.mean(r > 1.50), r.max(),
                           float(LAMBDAS[j]), G[te, j].mean()))
        r = Lr[te] / C[te]
        rec["ungated"].append((r.mean(), np.mean(r > 1.05), np.mean(r > 1.10), np.mean(r > 1.50), r.max(),
                               np.inf, Lr[te].mean()))
    out = {}
    for k, v in rec.items():
        a = np.array(v, dtype=float)
        out[k] = dict(mean_ratio=float(a[:, 0].mean()), harm5=float(a[:, 1].mean()), harm10=float(a[:, 2].mean()),
                      severe50=float(a[:, 3].mean()), worst=float(a[:, 4].mean()),
                      median_lambda=float(np.median(a[:, 5])), mean_rmse=float(a[:, 6].mean()),
                      p_harm10_exceeds_0p10=float(np.mean(a[:, 2] > 0.10)),
                      p_severe_exceeds_0p05=float(np.mean(a[:, 3] > 0.05)))
    out["pool"] = {m: float(Z[m].mean()) for m in ["fixed", "classical", "learned", "oracle"]}
    return out


def report(name, res):
    print(f"\n=== {name}: Tokyo pool, {N_SPLITS} splits (121 cal / 121 test) ===")
    print(f"  pool mean RMSE (m): " + ", ".join(f"{k} {v:.2f}" for k, v in res["pool"].items()))
    print(f"  {'':<12}{'ratio':>8}{'>5%':>7}{'>10%':>7}{'>50%':>7}{'worst':>8}{'lambda':>8}{'P(>10%>a)':>11}{'P(sev>5%)':>11}")
    for k in ["ungated", "naive", "single", "two_level"]:
        v = res[k]
        print(f"  {k:<12}{v['mean_ratio']:>8.3f}{v['harm5']*100:>6.1f}%{v['harm10']*100:>6.1f}%{v['severe50']*100:>6.1f}%"
              f"{v['worst']:>7.2f}x{v['median_lambda']:>8}{v['p_harm10_exceeds_0p10']*100:>10.1f}%{v['p_severe_exceeds_0p05']*100:>10.1f}%")


def main():
    from realdata import segments, load_run
    tune0 = segments(load_run("nagoya", 3), 53_000, None)
    print("Tuning process noise on Nagoya run 3 (fixed-R EKF)...", flush=True)
    q = tune_Q(tune0)

    summary = {"tuned_Q": q, "design": {"single": SINGLE, "two_level": TWO_LEVEL, "delta": DELTA}}
    for kappa in KAPPAS:
        print(f"\n[d'={kappa}] training on Nagoya runs 1-2, tuning on run 3...", flush=True)
        state, cfg = train_tune(kappa, q)
        tokyo = []
        for run in (1, 2, 3):
            tokyo += segments(load_run("tokyo", run), 90_000 + 1000 * run, kappa)
        print(f"  computing Tokyo pool ({len(tokyo)} segments x {len(LAMBDAS)} thresholds, reset + no-reset)...", flush=True)
        for label, oracle in [("learned", False)] + ([("perfectR", True)] if kappa == 3.0 else []):
            rows = pmap(_pool_job, [(r, s, oracle) for r, s in tokyo], q, state, cfg)
            Z = {k: np.array([r[k] for r in rows]) for k in ["fixed", "oracle", "classical", "learned",
                                                              "gated", "gated_noreset"]}
            key = f"{label}_k{kappa:g}"
            np.savez(f"../results/real_{key}.npz", lambdas=np.array(LAMBDAS), **Z)
            res = analyse(Z)
            res["config"] = cfg
            res["reset_ablation"] = {str(l): dict(
                reset_ratio=float(np.mean(Z["gated"][:, i] / Z["classical"])),
                reset_worst=float(np.max(Z["gated"][:, i] / Z["classical"])),
                noreset_ratio=float(np.mean(Z["gated_noreset"][:, i] / Z["classical"])),
                noreset_worst=float(np.max(Z["gated_noreset"][:, i] / Z["classical"])))
                for i, l in enumerate(LAMBDAS)}
            summary[key] = res
            report(f"{label} component, d'={kappa:g}", res)
    json.dump(summary, open("../results/real_summary.json", "w"), indent=2, default=float)
    print("\nWrote ../results/real_summary.json")


if __name__ == "__main__":
    main()
