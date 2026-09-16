"""
benchmark.py
------------
Benchmarks requested for the manuscript, on the simulated pool (600 trajectories,
d' = 3) and on the real Tokyo pool (242 segments, d' = 3):

  (B1) classical adaptive filters, no learning: fixed-R EKF; the paper's fallback
       (chi-square / quality rule); Sage-Husa adaptive R; covariance matching [1].
  (B2) learned filters: the learned-R filter used in the paper; a KalmanNet-style
       filter (re-implementation of [3], knet.py).
  (B3) safety mechanisms wrapped around EACH learned filter:
       none (ungated) | naive threshold (min calibration RMSE, no guarantee) |
       Gaussian solution separation (chi-square 99.9%, fixed threshold, switch+reset) |
       IMM-style predictive switching (use whichever filter predicted the last fixes
       better; EMA of predicted-residual difference, threshold 0, no reset) |
       certified gate, single (alpha = eps = 0.10) | certified gate, two-level.
  (B4) runtime per IMU step and per GNSS epoch.

PRE-SPECIFIED: all settings below and the tuning grids were fixed before any
benchmark result was computed. Tuning uses validation data only (simulation: seeds
100-119; real: Nagoya drive 3); pools were never used for tuning. The learned-R
gated matrices are those of the main experiments (results/quality_k3.0.npz and
results/real_learned_k3.npz), so the benchmark and the paper share one learned-R.

Usage:
  python benchmark.py sim-setup            # train KNet (sim), tune Sage-Husa / cov-matching
  python benchmark.py sim-pool 0 2         # pool chunk 0 of 2   (and: sim-pool 1 2)
  python benchmark.py real                 # setup + pool for the Tokyo experiment
  python benchmark.py runtime
  python benchmark.py analyze              # -> results/benchmark.json + tables
"""
import sys
import json
import time
import multiprocessing as mp
import numpy as np

R = "../results"
KAPPA = 3.0
LAMBDAS = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, np.inf]
SH_GRID = [0.90, 0.95, 0.98, 0.99]
CM_GRID = [5, 10, 20]
N_SPLITS, DELTA = 200, 0.10
SINGLE = (0.10, 0.10)
TWO_LEVEL = {"sim": [(0.20, 0.05), (0.02, 0.50)], "real": [(0.20, 0.05), (0.05, 0.50)]}
IMM_LAM = 1e-9  # "use the better predictor": gate open while the EMA difference <= 0
_W = {}


def rmse(track, ref):
    return float(np.sqrt(np.mean((track[:, 0] - ref["x"]) ** 2 + (track[:, 1] - ref["y"]) ** 2)))


# ------------------------------------------------------------------ shared worker
def _init(setup):
    import torch
    torch.set_num_threads(1)
    from ankf import DegradationDetector, make_classical_quality_fn
    _W["setup"] = setup
    _W["q"] = setup["q"]
    det = DegradationDetector(in_dim=6)
    det.load_state_dict(torch.load(setup["detector_path"]))
    det.eval()
    cfg = setup["cfg"]
    _W["L"] = lambda: det.make_r_scale_fn(cfg["learned_vi"], 1.0)
    _W["C"] = lambda: make_classical_quality_fn(cfg["classical_vi"], cfg["classical_qthr"])
    _W["knet"] = {k: np.array(v) for k, v in np.load(setup["knet_path"]).items()}


def _knet_factory():
    from knet import StepKNet
    p = _W["knet"]
    return lambda sens: StepKNet(p, x0=sens.get("x0"))


def _row(args):
    """All benchmark quantities for one (ref, sens)."""
    from gated import run_single, run_gated, SageHusaR, CovMatchR, CHI2_2DOF_999
    ref, sens = args
    q, L, C, st = _W["q"], _W["L"], _W["C"], _W["setup"]
    n = len(ref["t"])
    kf = _knet_factory()
    row = {"fixed": rmse(run_single(sens, n, q), ref),
           "sagehusa": rmse(run_single(sens, n, q, make_estimator=lambda: SageHusaR(st["sh_b"])), ref),
           "covmatch": rmse(run_single(sens, n, q, make_estimator=lambda: CovMatchR(st["cm_N"])), ref)}
    g = run_gated(sens, n, q, L, C, np.inf)
    row["classical"], row["learnedR"] = rmse(g["classical"], ref), rmse(g["learned"], ref)
    kg = []
    for lam in LAMBDAS:
        a = run_gated(sens, n, q, None, C, lam, reset=True, stat="sep_m", make_learned_filter=kf)
        kg.append(rmse(a["gated"], ref))
        if np.isinf(lam):
            row["knet"] = rmse(a["learned"], ref)
    row["knet_gated"] = kg
    for name, fac in [("learnedR", None), ("knet", kf)]:
        a = run_gated(sens, n, q, L, C, CHI2_2DOF_999, reset=True, stat="gauss_ss", make_learned_filter=fac)
        row[f"{name}_gauss_ss"] = rmse(a["gated"], ref)
        b = run_gated(sens, n, q, L, C, IMM_LAM, reset=False, stat="pred_resid", make_learned_filter=fac)
        row[f"{name}_imm"] = rmse(b["gated"], ref)
    return row


def pool_rows(pairs, setup):
    import multiprocessing as mp
    with mp.Pool(mp.cpu_count() - 1, initializer=_init, initargs=(setup,)) as pool:
        return pool.map(_row, pairs, chunksize=2)


def tune_classical(pairs, q):
    """Mean validation RMSE for the Sage-Husa and covariance-matching grids."""
    from gated import run_single, SageHusaR, CovMatchR
    sh = {b: np.mean([rmse(run_single(s, len(r["t"]), q, make_estimator=lambda: SageHusaR(b)), r) for r, s in pairs])
          for b in SH_GRID}
    cm = {N: np.mean([rmse(run_single(s, len(r["t"]), q, make_estimator=lambda: CovMatchR(N)), r) for r, s in pairs])
          for N in CM_GRID}
    b, N = min(sh, key=sh.get), min(cm, key=cm.get)
    print(f"  Sage-Husa b: {({k: round(v, 3) for k, v in sh.items()})} -> {b}")
    print(f"  cov-matching N: {({k: round(v, 3) for k, v in cm.items()})} -> {N}")
    return float(b), int(N)


def save_knet(model, path):
    from knet import export_params
    np.savez(path, **export_params(model))


# ------------------------------------------------------------------ simulation
def sim_setup():
    from simulate import make_dataset
    from knet import train_knet
    S = json.load(open(f"{R}/results.json"))
    q = {"q_accel": S["tuned_Q"]["q_accel"], "q_gyro": S["tuned_Q"]["q_gyro"]}
    cfg = json.load(open(f"{R}/quality_k{KAPPA}.json"))["config"]
    # KalmanNet has far more parameters than the learned-R detector and overfits
    # 30 trajectories (first attempt: best validation MSE 193 m^2 at epoch 7, rising
    # after). It is given 150 training trajectories -- seeds 0-29 plus 4000-4119,
    # disjoint from validation (100-119), development (2000-2199) and pool (1000-1599).
    train = [make_dataset(seed=s, kappa=KAPPA) for s in list(range(0, 30)) + list(range(4000, 4120))]
    val = [make_dataset(seed=s, kappa=KAPPA) for s in range(100, 120)]
    print("[sim] tuning classical adaptive baselines on validation seeds 100-119...")
    b, N = tune_classical(val, q)
    print(f"[sim] training KalmanNet-style filter on {len(train)} trajectories (val 100-119)...")
    t0 = time.time()
    setup = dict(q=q, cfg=cfg, detector_path=f"{R}/detector_k{KAPPA}.pt", knet_path=f"{R}/knet_sim.npz",
                 sh_b=b, cm_N=N, knet_train_trajectories=len(train))
    json.dump(setup, open(f"{R}/bench_sim_setup.json", "w"), indent=2, default=float)
    model = train_knet(train, val, q, epochs=80, patience=12, ckpt=f"{R}/knet_sim.npz")
    print(f"  trained in {time.time()-t0:.0f}s")
    save_knet(model, f"{R}/knet_sim.npz")


def sim_pool(chunk, nchunks):
    from simulate import make_dataset
    setup = json.load(open(f"{R}/bench_sim_setup.json"))
    seeds = [s for i, s in enumerate(range(1000, 1600)) if i % nchunks == chunk]
    pairs = [make_dataset(seed=s, kappa=KAPPA) for s in seeds]
    t0 = time.time()
    rows = pool_rows(pairs, setup)
    for r, s in zip(rows, seeds):
        r["seed"] = s
    json.dump(rows, open(f"{R}/bench_sim_chunk{chunk}.json", "w"))
    print(f"[sim] chunk {chunk}/{nchunks}: {len(rows)} trajectories in {time.time()-t0:.0f}s")


# ------------------------------------------------------------------ real data
def real():
    import torch
    import experiment_real as ER
    from realdata import segments, load_run
    from knet import train_knet
    RS = json.load(open(f"{R}/real_summary.json"))
    q = RS["tuned_Q"]
    print("[real] re-deriving the d'=3 detector and fallback (same seed, same data)...")
    state, cfg = ER.train_tune(KAPPA, q)
    saved = RS[f"learned_k{KAPPA:g}"]["config"]
    same = all(abs(float(cfg[k]) - float(saved[k])) < 1e-9 if isinstance(saved[k], (int, float)) else cfg[k] == saved[k]
               for k in ["detector_acc", "classical_vi", "learned_vi", "classical_tune_rmse", "learned_tune_rmse"])
    print(f"  reproduces the saved configuration exactly: {same}")
    if not same:
        raise SystemExit("detector/fallback not reproducible -- refusing to benchmark against a different model")
    torch.save(state, f"{R}/detector_real_k{KAPPA}.pt")
    train = segments(load_run("nagoya", 1), 51_000, KAPPA) + segments(load_run("nagoya", 2), 52_000, KAPPA)
    val = segments(load_run("nagoya", 3), 53_000, KAPPA)
    print("[real] tuning classical adaptive baselines on Nagoya drive 3...")
    b, N = tune_classical(val, q)
    print("[real] training KalmanNet-style filter on Nagoya drives 1-2 (val: drive 3)...")
    t0 = time.time()
    model = train_knet(train, val, q, epochs=80, patience=12, ckpt=f"{R}/knet_real.npz")
    print(f"  trained in {time.time()-t0:.0f}s")
    save_knet(model, f"{R}/knet_real.npz")
    setup = dict(q=q, cfg=cfg, detector_path=f"{R}/detector_real_k{KAPPA}.pt", knet_path=f"{R}/knet_real.npz",
                 sh_b=b, cm_N=N)
    json.dump(setup, open(f"{R}/bench_real_setup.json", "w"), indent=2, default=float)
    tokyo = []
    for run in (1, 2, 3):
        tokyo += segments(load_run("tokyo", run), 90_000 + 1000 * run, KAPPA)
    t0 = time.time()
    rows = pool_rows(tokyo, setup)
    json.dump(rows, open(f"{R}/bench_real_rows.json", "w"))
    print(f"[real] {len(rows)} Tokyo segments in {time.time()-t0:.0f}s")


# ------------------------------------------------------------------ runtime
def runtime():
    from simulate import make_dataset
    from gated import run_single, run_gated
    setup = json.load(open(f"{R}/bench_sim_setup.json"))
    _init(setup)
    q, L, C = _W["q"], _W["L"], _W["C"]
    kf = _knet_factory()
    pairs = [make_dataset(seed=s, kappa=KAPPA) for s in range(3000, 3005)]
    steps = sum(len(r["t"]) for r, _ in pairs)
    epochs = sum(len(s["gps_idx"]) for _, s in pairs)
    cases = {"fixed-R EKF": lambda r, s: run_single(s, len(r["t"]), q),
             "classical adaptive (fallback)": lambda r, s: run_single(s, len(r["t"]), q, C),
             "learned-R filter": lambda r, s: run_single(s, len(r["t"]), q, L),
             "KalmanNet-style filter": lambda r, s: run_gated(s, len(r["t"]), q, None, C, np.inf, make_learned_filter=kf),
             "certified gate (learned-R + fallback)": lambda r, s: run_gated(s, len(r["t"]), q, L, C, 6.0, stat="sep_m"),
             "certified gate (KalmanNet + fallback)": lambda r, s: run_gated(s, len(r["t"]), q, None, C, 6.0, stat="sep_m",
                                                                             make_learned_filter=kf)}
    out = {}
    for name, fn in cases.items():
        t0 = time.perf_counter()
        for r, s in pairs:
            fn(r, s)
        dt = time.perf_counter() - t0
        out[name] = dict(us_per_imu_step=dt / steps * 1e6, x_realtime_at_100Hz=(steps * 0.01) / dt)
        print(f"  {name:<40} {dt/steps*1e6:7.1f} us per IMU step  ({(steps*0.01)/dt:6.0f}x faster than real time)")
    # certification cost: LTT over the saved matrix
    from certify import ltt_select
    Z = np.load(f"{R}/quality_k{KAPPA}.npz")
    t0 = time.perf_counter()
    for _ in range(100):
        ltt_select(Z["gated"], Z["classical"], np.arange(300), 0.10, 0.10, 0.10)
    out["LTT calibration (300 trajectories, 11 thresholds)"] = dict(ms=(time.perf_counter() - t0) / 100 * 1e3)
    print(f"  LTT calibration (offline): {out['LTT calibration (300 trajectories, 11 thresholds)']['ms']:.2f} ms")
    json.dump(out, open(f"{R}/bench_runtime.json", "w"), indent=2)


# ------------------------------------------------------------------ analysis
def analyze_one(rows, Glr, domain):
    from certify import ltt_select, ltt_select_multi
    C = np.array([r["classical"] for r in rows])
    N = len(C); n_cal = N // 2
    methods = {k: np.array([r[k] for r in rows]) for k in
               ["fixed", "sagehusa", "covmatch", "learnedR", "knet", "learnedR_gauss_ss", "knet_gauss_ss",
                "learnedR_imm", "knet_imm"]}
    Gk = np.array([r["knet_gated"] for r in rows])
    rng = np.random.default_rng(0)
    splits = [rng.permutation(N) for _ in range(N_SPLITS)]

    def stat(pred_fn):
        rec = []
        for p in splits:
            cal, te = p[:n_cal], p[n_cal:]
            g = pred_fn(cal, te)
            r = g / C[te]
            rec.append((r.mean(), np.mean(r > 1.10), np.mean(r > 1.50), r.max(), np.mean(r > 1.10) > SINGLE[0], g.mean()))
        a = np.array(rec, float)
        return dict(mean_ratio=float(a[:, 0].mean()), harm10=float(a[:, 1].mean()), severe50=float(a[:, 2].mean()),
                    worst=float(a[:, 3].mean()), violations=float(a[:, 4].mean()), mean_rmse=float(a[:, 5].mean()))

    out = {"classical_baselines": {}, "learned": {}}
    for k in ["fixed", "sagehusa", "covmatch"]:
        out["classical_baselines"][k] = stat(lambda cal, te, k=k: methods[k][te])
    out["classical_baselines"]["fallback"] = stat(lambda cal, te: C[te])
    for name, G, ung in [("learnedR", Glr, methods["learnedR"]), ("knet", Gk, methods["knet"])]:
        out["learned"][name] = {
            "ungated": stat(lambda cal, te: ung[te]),
            "naive": stat(lambda cal, te: G[te, int(np.argmin(G[cal].mean(axis=0)))]),
            "gauss_ss": stat(lambda cal, te: methods[f"{name}_gauss_ss"][te]),
            "imm": stat(lambda cal, te: methods[f"{name}_imm"][te]),
            "certified_single": stat(lambda cal, te: G[te, ltt_select(G, C, cal, SINGLE[0], DELTA, SINGLE[1])]),
            "certified_two_level": stat(lambda cal, te: G[te, ltt_select_multi(G, C, cal, TWO_LEVEL[domain], DELTA)]),
        }
    out["pool_mean_rmse"] = {k: float(v.mean()) for k, v in methods.items()} | {"classical": float(C.mean())}
    return out


def analyze():
    res = {}
    rows = []
    for i in range(2):
        rows += json.load(open(f"{R}/bench_sim_chunk{i}.json"))
    rows.sort(key=lambda r: r["seed"])
    Z = np.load(f"{R}/quality_k{KAPPA}.npz")
    C_main = Z["classical"]
    C_bench = np.array([r["classical"] for r in rows])
    assert np.allclose(C_main, C_bench), "benchmark fallback differs from the main experiment"
    res["sim"] = analyze_one(rows, Z["gated"], "sim")
    rr = json.load(open(f"{R}/bench_real_rows.json"))
    Zr = np.load(f"{R}/real_learned_k{KAPPA:g}.npz")
    assert np.allclose(Zr["classical"], [r["classical"] for r in rr]), "real fallback differs from main experiment"
    res["real"] = analyze_one(rr, Zr["gated"], "real")
    res["setup"] = {"sim": json.load(open(f"{R}/bench_sim_setup.json")), "real": json.load(open(f"{R}/bench_real_setup.json"))}
    json.dump(res, open(f"{R}/benchmark.json", "w"), indent=2, default=float)
    for dom in ["sim", "real"]:
        d = res[dom]
        print(f"\n===== {dom.upper()} =====   (ratios vs classical fallback; mean over {N_SPLITS} test halves)")
        print(f"{'classical baseline':<34}{'ratio':>7}{'>10%':>7}{'worst':>8}{'RMSE':>7}")
        for k, v in d["classical_baselines"].items():
            print(f"{k:<34}{v['mean_ratio']:>7.3f}{v['harm10']*100:>6.1f}%{v['worst']:>7.2f}x{v['mean_rmse']:>7.2f}")
        for name in ["learnedR", "knet"]:
            print(f"-- {name}")
            for m, v in d["learned"][name].items():
                print(f"   {m:<31}{v['mean_ratio']:>7.3f}{v['harm10']*100:>6.1f}%{v['worst']:>7.2f}x{v['mean_rmse']:>7.2f}"
                      f"   >50%: {v['severe50']*100:4.1f}%   viol(single): {v['violations']*100:5.1f}%")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "sim-setup":
        sim_setup()
    elif cmd == "sim-pool":
        sim_pool(int(sys.argv[2]), int(sys.argv[3]))
    elif cmd == "real":
        real()
    elif cmd == "runtime":
        runtime()
    elif cmd == "analyze":
        analyze()
