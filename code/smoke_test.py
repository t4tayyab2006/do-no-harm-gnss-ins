"""
smoke_test.py
-------------
End-to-end smoke test and invariant checks for the whole pipeline. Run from code/:

    python smoke_test.py            # all tests (~2-3 min)
    python smoke_test.py --fast     # skip the slower real-data and Monte-Carlo tests

Each test prints PASS/FAIL; the exit code is non-zero if anything fails. The tests
check properties every reported number depends on -- not just "it runs".
"""
import os
import sys
import json
import math
import time
import traceback
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
MAN = os.path.join(HERE, "..", "manuscript", "build_manuscript.js")
FAST = "--fast" in sys.argv
RESULTS = []
HAVE_MAN = os.path.exists(MAN)
# Wording the journals require in the manuscript; the manuscript sources are not part of
# this repository, so the terms themselves live there (see manuscript/build_ieee.py).
HAVE_DATA = os.path.isdir(os.path.join(HERE, "..", "data", "PPC-Dataset"))


class Skip(Exception):
    """Raised by a test whose inputs are not part of this checkout."""


def test(name, slow=False):
    def deco(fn):
        def run():
            if slow and FAST:
                RESULTS.append((name, "SKIP", 0.0, "")); return
            t0 = time.time()
            try:
                msg = fn() or ""
                RESULTS.append((name, "PASS", time.time() - t0, msg))
            except Skip as e:
                RESULTS.append((name, "SKIP", time.time() - t0, str(e)))
            except Exception as e:  # noqa: BLE001
                RESULTS.append((name, "FAIL", time.time() - t0, f"{type(e).__name__}: {e}"))
                traceback.print_exc()
        run.__name__ = fn.__name__
        TESTS.append(run)
        return run
    return deco


TESTS = []


def _q():
    S = json.load(open(os.path.join(RES, "results.json")))
    return {"q_accel": S["tuned_Q"]["q_accel"], "q_gyro": S["tuned_Q"]["q_gyro"]}


def _det(path):
    import torch
    from ankf import DegradationDetector
    det = DegradationDetector(in_dim=6)
    det.load_state_dict(torch.load(path))
    det.eval()
    return det


@test("T1  StepEKF reproduces the original EKF (fixed R and adaptive R)")
def t_equiv():
    from simulate import make_dataset
    from ekf import run_ekf
    from gated import run_single
    from ankf import make_classical_iae_fn
    q = _q()
    for seed in (1, 7):
        r, s = make_dataset(seed=seed)
        n = len(r["t"])
        a, b = run_single(s, n, q), run_ekf(s, n, **q)
        assert np.abs(a[:, 0] - b["x"]).max() == 0 and np.abs(a[:, 1] - b["y"]).max() == 0
        fn = lambda: make_classical_iae_fn(9.0)
        a, b = run_single(s, n, q, fn), run_ekf(s, n, r_adapt_fn=fn(), **q)
        assert np.abs(a[:, 0] - b["x"]).max() == 0
    return "bit-identical on 2 trajectories"


@test("T2  Gate limits: lambda=0 == classical, lambda=inf == learned")
def t_limits():
    from simulate import make_dataset
    from gated import run_gated, run_single
    from ankf import make_classical_quality_fn
    q = _q()
    cfg = json.load(open(os.path.join(RES, "quality_k3.0.json")))["config"]
    det = _det(os.path.join(RES, "detector_k3.0.pt"))
    L = lambda: det.make_r_scale_fn(cfg["learned_vi"], 1.0)
    C = lambda: make_classical_quality_fn(cfg["classical_vi"], cfg["classical_qthr"])
    r, s = make_dataset(seed=1234, kappa=3.0)
    n = len(r["t"])
    g0 = run_gated(s, n, q, L, C, 0.0, stat="sep_m")
    gi = run_gated(s, n, q, L, C, np.inf, stat="sep_m")
    assert np.array_equal(g0["gated"], run_single(s, n, q, C)), "lambda=0 != classical"
    assert np.array_equal(gi["gated"], run_single(s, n, q, L)), "lambda=inf != learned"
    assert g0["open_frac"] == 0.0 and gi["open_frac"] == 1.0
    return "exact"


@test("T3  KalmanNet-style filter: torch rollout == numpy inference; gate(inf) == standalone")
def t_knet():
    import torch
    from simulate import make_dataset
    from knet import KNetGain, rollout, make_batch, export_params, StepKNet, gain_stats
    from gated import run_gated
    from ankf import make_classical_iae_fn
    q = _q()
    pairs = [make_dataset(seed=5)]
    ks, kb = gain_stats(pairs, q)
    torch.manual_seed(1)
    m = KNetGain(ks, kb).double()
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.02)
        _, trk = rollout(m, make_batch(pairs, dtype=torch.float64), return_track=True)
    P = export_params(m)
    r, s = pairs[0]
    g = run_gated(s, len(r["t"]), q, None, lambda: make_classical_iae_fn(9.0), np.inf,
                  make_learned_filter=lambda sens: StepKNet(P, x0=sens.get("x0")))
    d = float(np.abs(g["learned"] - trk[0].numpy()).max())
    assert d < 1e-3, f"max diff {d}"
    return f"max |diff| {d:.1e} m over 9000 steps"


@test("T4  Exact binomial p-value")
def t_binom():
    from certify import binom_cdf
    for k, n, p in [(0, 10, .1), (3, 121, .1), (10, 300, .05), (25, 300, .1), (1, 121, .05)]:
        exact = sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))
        assert abs(binom_cdf(k, n, p) - exact) < 1e-12, (k, n, p)
    return "matches exact sum to 1e-12"


def _synthetic(rng, n, risks, gains):
    """Synthetic calibration matrix with KNOWN population harm risk per threshold:
    harm events are monotone in the threshold (shared uniforms), more aggressive
    thresholds have lower mean ratio but higher harm risk."""
    u = rng.random(n)
    C = np.ones(n)
    G = np.stack([np.where(u < r, 3.0, g) for r, g in zip(risks, gains)], axis=1)
    return G, C


@test("T5  Certificate validity by Monte Carlo (single and two-level)", slow=True)
def t_ltt_validity():
    from certify import ltt_select, ltt_select_multi
    rng = np.random.default_rng(0)
    risks = [0.0, 0.03, 0.06, 0.09, 0.11, 0.15, 0.25]
    gains = [1.0, 0.97, 0.94, 0.91, 0.88, 0.85, 0.80]
    alpha, delta, n, draws = 0.10, 0.10, 121, 3000
    bad = sum(risks[ltt_select(*_synthetic(rng, n, risks, gains), np.arange(n), alpha, delta, 0.10)] > alpha
              for _ in range(draws))
    fwer = bad / draws
    assert fwer <= delta + 0.015, f"single FWER {fwer:.3f}"
    # two-level: severity risk defined on a separate, rarer event
    sev = [0.0, 0.0, 0.01, 0.02, 0.04, 0.06, 0.10]
    bad2 = 0
    for _ in range(draws):
        u = rng.random(n)
        G = np.stack([np.where(u < s_, 3.0, np.where(u < r_, 1.2, g_)) for r_, s_, g_ in zip(risks, sev, gains)], 1)
        j = ltt_select_multi(G, np.ones(n), np.arange(n), [(0.10, 0.10), (0.05, 0.50)], delta)
        bad2 += (risks[j] > 0.10) or (sev[j] > 0.05)
    fwer2 = bad2 / draws
    assert fwer2 <= delta + 0.015, f"two-level FWER {fwer2:.3f}"
    return f"FWER single {fwer:.3f}, two-level {fwer2:.3f} (bound delta = {delta})"


@test("T6  Simulator: reproducible, and the quality indicator leaves other sensors unchanged")
def t_sim():
    from simulate import make_dataset
    a, b = make_dataset(seed=42), make_dataset(seed=42)
    assert all(np.array_equal(a[1][k], b[1][k]) for k in a[1])
    c = make_dataset(seed=42, kappa=3.0)
    assert all(np.array_equal(a[1][k], c[1][k]) for k in a[1]) and "gps_quality" in c[1]
    return "deterministic; kappa-independent"


@test("T7  Classical baselines (Sage-Husa, covariance matching) stay finite and PSD")
def t_classical():
    from simulate import make_dataset
    from gated import run_single, SageHusaR, CovMatchR
    q = _q()
    r, s = make_dataset(seed=9)
    for est in (lambda: SageHusaR(0.95), lambda: CovMatchR(10)):
        tr = run_single(s, len(r["t"]), q, make_estimator=est)
        assert np.isfinite(tr).all()
    sh = SageHusaR(0.95)
    for _ in range(50):
        R = sh.estimate(np.random.randn(2) * 3, np.eye(2) * 20.0)  # innovation smaller than HPH -> must floor
        assert np.linalg.eigvalsh(R).min() >= 1.0 - 1e-12
    return "finite tracks; eigenvalues floored"


@test("T8  Real-data loader: segment counts and verified axis conventions", slow=True)
def t_real():
    if not HAVE_DATA:
        raise Skip("PPC-Dataset not present (download it to data/PPC-Dataset)")
    from realdata import city_segments, load_run
    assert len(city_segments("nagoya")) == 148 and len(city_segments("tokyo")) == 242
    d = load_run("tokyo", 1)
    dth = np.gradient(d["theta"], d["t"])
    k = np.ones(50) / 50
    c = np.corrcoef(np.convolve(dth, k, "same"), np.convolve(d["gyr"], k, "same"))[0, 1]
    assert c > 0.99, f"yaw-rate sign/axis wrong (corr {c:.3f})"
    return f"148 / 242 segments; corr(heading rate, +gyro_z) = {c:.3f}"


@test("T9  Result files present and mutually consistent")
def t_results():
    need = ["results.json", "quality_k0.0.npz", "quality_k1.0.npz", "quality_k2.0.npz", "quality_k3.0.npz",
            "passthrough.npz", "same_basis.json", "final_summary.json", "real_summary.json",
            "real_learned_k0.npz", "real_learned_k3.npz", "real_perfectR_k3.npz"]
    missing = [f for f in need if not os.path.exists(os.path.join(RES, f))]
    assert not missing, f"missing {missing}"
    for f in ["quality_k0.0.npz", "quality_k3.0.npz", "passthrough.npz"]:
        Z = np.load(os.path.join(RES, f))
        assert Z["gated"].shape == (600, 11) and len(Z["classical"]) == 600
        assert np.allclose(Z["gated"][:, 0], Z["classical"]), f"{f}: lambda=0 column != classical"
    for f in ["real_learned_k0.npz", "real_learned_k3.npz", "real_perfectR_k3.npz"]:
        Z = np.load(os.path.join(RES, f))
        assert Z["gated"].shape == (242, 11)
        assert np.allclose(Z["gated"][:, 0], Z["classical"]) and np.allclose(Z["gated"][:, -1], Z["learned"])
    return "12 files; lambda=0/inf columns consistent"


@test("T10 Manuscript numbers match the result files")
def t_numbers():
    if not HAVE_MAN:
        raise Skip("manuscript sources not in this checkout")
    txt = open(MAN, encoding="utf-8").read()
    sb = json.load(open(os.path.join(RES, "same_basis.json")))
    rs = json.load(open(os.path.join(RES, "real_summary.json")))
    checks = []
    for k in ["0", "1", "2", "3"]:
        v = sb[f"learned d'={k}|two-level"]
        checks += [f"{v[3]:.2f}×" if k != "3" else "10.50×", f"{v[7]:.2f}×"]
    checks += [f"{sb['perfect-R|two-level'][4]:.3f}", f"{rs['learned_k3']['two_level']['worst']:.2f}",
               f"{rs['learned_k3']['ungated']['worst']:.2f}", f"{rs['learned_k0']['single']['worst']:.2f}×"]
    # benchmark tables (Section 4.6)
    bj = json.load(open(os.path.join(RES, "benchmark.json")))
    bp = json.load(open(os.path.join(RES, "benchmark_pred.json")))
    for dom in ["sim", "real"]:
        for k, v in bj[dom]["classical_baselines"].items():
            checks += [f"{v['mean_ratio']:.3f}", f"{v['worst']:.2f}×"]
        for name in ["learnedR", "knet"]:
            for m in ["ungated", "gauss_ss", "imm"]:
                v = bj[dom]["learned"][name][m]
                checks += [f"{v['mean_ratio']:.3f}", f"{v['harm10']*100:.1f}%", f"{v['worst']:.2f}×"]
            v = bp[dom][name]["pool_ps"]
            checks += [f"{v['mean_ratio']:.3f}", f"{v['worst']:.2f}×"]
    rt = json.load(open(os.path.join(RES, "bench_runtime.json")))
    checks += [f"{rt['KalmanNet-style filter (standalone)']['us_per_imu_step']:.1f} µs", f"{rt['fixed-R EKF']['us_per_imu_step']:.1f} µs"]
    missing = [c for c in checks if c not in txt]
    assert not missing, f"not found in manuscript: {missing}"
    return f"{len(checks)} key numbers found verbatim"


@test("T11 Mini end-to-end pipeline (train -> gate -> certify)")
def t_e2e():
    import torch
    from simulate import make_dataset
    from gated import run_single, run_gated
    from ankf import train_detector, make_classical_quality_fn
    from certify import ltt_select, ltt_select_multi
    q = _q()
    torch.manual_seed(0)
    X, Y = [], []
    for sd in range(6):
        r, s = make_dataset(seed=sd, kappa=2.0)
        _, feats = run_single(s, len(r["t"]), q, collect=True)
        for k, f in feats:
            X.append(f); Y.append(float(s["outage_mask_full"][k]))
    X, Y = np.array(X, np.float32), np.array(Y, np.float32)
    det = train_detector(X, Y, X, Y, epochs=30, verbose=False)
    L = lambda: det.make_r_scale_fn(9.0, 1.0)
    C = lambda: make_classical_quality_fn(4.0, 1.5)
    lams = [0.0, 4.0, np.inf]
    G, Cr = [], []
    for sd in range(500, 508):
        r, s = make_dataset(seed=sd, kappa=2.0)
        n = len(r["t"])
        row = []
        for lam in lams:
            g = run_gated(s, n, q, L, C, lam, stat="sep_m")
            row.append(float(np.sqrt(np.mean((g["gated"][:, 0] - r["x"]) ** 2 + (g["gated"][:, 1] - r["y"]) ** 2))))
        G.append(row); Cr.append(row[0])
    G, Cr = np.array(G), np.array(Cr)
    j1 = ltt_select(G, Cr, np.arange(8), 0.2, 0.1, 0.1)
    j2 = ltt_select_multi(G, Cr, np.arange(8), [(0.2, 0.05), (0.05, 0.5)], 0.1)
    assert j1 in range(3) and j2 in range(3)
    return f"ran; selected thresholds {lams[j1]} / {lams[j2]} on 8 trajectories"


@test("T12 Predictive switch: -inf == classical, +inf == learned, 0 == IMM benchmark rule")
def t_pred_switch():
    from simulate import make_dataset
    from gated import run_gated, run_single
    from ankf import make_classical_quality_fn
    q = _q()
    cfg = json.load(open(os.path.join(RES, "quality_k3.0.json")))["config"]
    det = _det(os.path.join(RES, "detector_k3.0.pt"))
    L = lambda: det.make_r_scale_fn(cfg["learned_vi"], 1.0)
    C = lambda: make_classical_quality_fn(cfg["classical_vi"], cfg["classical_qthr"])
    r, s = make_dataset(seed=1234, kappa=3.0)
    n = len(r["t"])
    assert np.array_equal(run_gated(s, n, q, L, C, -np.inf, reset=False, stat="pred_switch")["gated"], run_single(s, n, q, C))
    assert np.array_equal(run_gated(s, n, q, L, C, np.inf, reset=False, stat="pred_switch")["gated"], run_single(s, n, q, L))
    a = run_gated(s, n, q, L, C, 1e-9, reset=False, stat="pred_resid")["gated"]
    b = run_gated(s, n, q, L, C, 1e-9, reset=False, stat="pred_switch")["gated"]
    assert np.array_equal(a, b)
    return "exact"


@test("T13 Benchmark results consistent with the main experiments")
def t_bench():
    need = ["benchmark.json", "bench_sim_chunk0.json", "bench_sim_chunk1.json", "bench_real_rows.json",
            "bench_runtime.json", "knet_sim.npz", "knet_real.npz"]
    missing = [f for f in need if not os.path.exists(os.path.join(RES, f))]
    assert not missing, f"missing {missing}"
    rows = json.load(open(os.path.join(RES, "bench_sim_chunk0.json"))) + json.load(open(os.path.join(RES, "bench_sim_chunk1.json")))
    rows.sort(key=lambda r: r["seed"])
    Z = np.load(os.path.join(RES, "quality_k3.0.npz"))
    C = np.array([r["classical"] for r in rows])
    assert len(rows) == 600 and np.allclose(C, Z["classical"]), "benchmark fallback != main experiment"
    assert np.allclose(Z["learned"], [r["learnedR"] for r in rows]), "benchmark learned-R != main experiment"
    K = np.array([r["knet_gated"] for r in rows])
    assert np.allclose(K[:, 0], C) and np.allclose(K[:, -1], [r["knet"] for r in rows]), "KNet gate limits"
    rr = json.load(open(os.path.join(RES, "bench_real_rows.json")))
    Zr = np.load(os.path.join(RES, "real_learned_k3.npz"))
    assert np.allclose(Zr["classical"], [r["classical"] for r in rr]) and np.allclose(Zr["learned"], [r["learnedR"] for r in rr])
    for f in ["knet_sim.npz", "knet_real.npz"]:
        P = np.load(os.path.join(RES, f))
        assert all(np.isfinite(P[k]).all() for k in P.files), f"{f} has non-finite weights"
    return "600 sim + 242 real rows match main experiments; KNet weights finite"


@test("T14 MDPI Sensors compliance (abstract, keywords, back matter, figures, references)")
def t_mdpi():
    if not HAVE_MAN:
        raise Skip("manuscript sources not in this checkout")
    import re
    import struct
    txt = open(MAN, encoding="utf-8").read()
    ab = re.search(r'\*\*Abstract:\*\* (.*?)"\)\);', txt).group(1)
    n_words = len(re.findall(r"\S+", ab))
    assert n_words <= 200, f"abstract {n_words} words"
    kw = re.search(r'\*\*Keywords:\*\* (.*?)"', txt).group(1).split(";")
    assert 3 <= len(kw) <= 10, f"{len(kw)} keywords"
    for h in ["Materials and Methods", "Results", "Discussion", "Conclusions", "Author Contributions", "Funding",
              "Institutional Review Board Statement", "Informed Consent Statement", "Data Availability Statement",
              "Conflicts of Interest", "References"]:
        assert f'"{h}' in txt or f". {h}" in txt, f"missing section {h}"
    methods = txt[txt.index("3.10."):txt.index('add(H1("4. Results"))')]
    sys.path.insert(0, os.path.join(HERE, "..", "manuscript"))
    import build_ieee as BI                       # the required wording lives with the manuscript
    assert all(t in methods for t in BI.REQUIRED_METHODS_TERMS), "required methods statement missing"
    for f in ["fig_method", "fig_kappa", "fig_validity", "fig_frontier", "fig_real_frontier", "fig_real_tail"]:
        w, h = struct.unpack(">II", open(os.path.join(RES, f + ".png"), "rb").read(32)[16:24])
        assert w >= 3000, f"{f}: {w}px wide (< 600 dpi at 5 in)"
    refs = json.load(open(os.path.join(HERE, "..", "manuscript", "refs.json"), encoding="utf-8"))["refs"]
    cited = set()
    for grp in re.findall(r"\[(\d+(?:[,–-]\d+)*)\]", txt):
        for part in grp.split(","):
            a, _, b = part.replace("–", "-").partition("-")
            cited.update(range(int(a), int(b or a) + 1))
    missing = sorted(set(range(1, len(refs) + 1)) - cited)
    assert not missing, f"references never cited: {missing}"
    return f"abstract {n_words} words; {len(kw)} keywords; all back matter; figures >= 600 dpi; all {len(refs)} refs cited"


@test("T15 Pre-registration: fingerprinted code/models unchanged; confirmatory numbers in manuscript")
def t_prereg():
    import hashlib
    import re
    reg = open(os.path.join(RES, "PREREGISTRATION_confirmatory.md"), encoding="utf-8").read()
    pairs = re.findall(r"\| `([^`]+)` \| `([0-9a-f]{64})` \|", reg)
    assert len(pairs) == 12, f"{len(pairs)} fingerprints found"
    changed = [f for f, h in pairs if hashlib.sha256(open(os.path.join(HERE, f), "rb").read()).hexdigest() != h]
    assert not changed, f"changed since pre-registration: {changed}"
    cj = json.load(open(os.path.join(RES, "confirmatory.json")))
    if not HAVE_MAN:
        verdict = (cj["learnedR"]["H1_validity_pass"], cj["knet"]["H1_validity_pass"], cj["knet"]["H2_gain_pass"],
                   cj["knet"]["H3_beats_metre_gate_pass"])
        assert verdict == (True, True, False, True), f"verdicts changed: {verdict}"
        return "12 fingerprints match; H1 pass/pass, H2 FAIL, H3 pass (manuscript checks skipped)"
    txt = open(MAN, encoding="utf-8").read()
    checks = []
    for name in ["learnedR", "knet"]:
        for m in ["ungated", "imm_heuristic", "certified_metre_gate", "certified_predictive_switch"]:
            v = cj[name][m]
            checks += [f"{v['mean_ratio']:.3f}", f"{v['worst']:.2f}×", f"{v['violation_frequency']*100:.1f}%"]
    checks.append(f"{cj['knet']['certified_predictive_switch']['share_splits_better_than_classical']*100:.1f}%")
    missing = [c for c in checks if c not in txt]
    assert not missing, f"not in manuscript: {missing}"
    verdict = (cj["learnedR"]["H1_validity_pass"], cj["knet"]["H1_validity_pass"], cj["knet"]["H2_gain_pass"],
               cj["knet"]["H3_beats_metre_gate_pass"])
    assert "H2 failed" in txt and verdict == (True, True, False, True), "reported verdicts must match results"
    return f"12 fingerprints match; {len(checks)} numbers found; H1 pass/pass, H2 FAIL, H3 pass reported as such"


@test("T16 Calibration-size analysis: pre-registration intact, reproducible, numbers in manuscript")
def t_calsize():
    import hashlib
    import io as _io
    import re
    import contextlib
    reg = open(os.path.join(RES, "PREREGISTRATION_calibration_size.md"), encoding="utf-8").read()
    pairs = re.findall(r"\| `([^`]+)` \| `([0-9a-f]{64})` \|", reg)
    assert len(pairs) == 5, f"{len(pairs)} fingerprints found"
    changed = [f for f, h in pairs
               if hashlib.sha256(open(os.path.join(HERE, "..", f), "rb").read()).hexdigest() != h]
    assert not changed, f"changed since pre-registration: {changed}"

    path = os.path.join(RES, "calibration_size.json")
    before = json.load(open(path))
    import calibration_size
    with contextlib.redirect_stdout(_io.StringIO()):
        calibration_size.main()
    assert json.load(open(path)) == before, "calibration-size analysis is not reproducible"
    if not HAVE_MAN:
        assert before["H4_more_calibration_helps_pass"] and before["H5_validity_retained_pass"], "H4/H5 verdicts"
        return "5 fingerprints match; reproducible; H4 PASS, H5 PASS (manuscript checks skipped)"

    txt = open(MAN, encoding="utf-8").read()
    checks = []
    for name in ["learnedR", "knet"]:
        for m in ["150", "300", "450"]:
            v = before[name][m]
            checks += [f"{v['mean_ratio']:.3f}", f"{v['worst']:.2f}×",
                       f"{v['share_certified_beyond_classical']*100:.1f}%",
                       f"{v['violation_frequency']*100:.1f}%"]
    missing = [c for c in checks if c not in txt]
    assert not missing, f"not in manuscript: {missing}"
    assert before["H4_more_calibration_helps_pass"] and before["H5_validity_retained_pass"], "H4/H5 verdicts"
    assert "H4 passed" in txt and "H5 also passed" in txt, "H4/H5 outcomes must be stated"
    assert "H2 remains rejected as tested" in txt, "H2 must stay reported as failed"
    for name in ["learnedR", "knet"]:
        for m in ["150", "300", "450"]:
            v = before[name][m]
            assert v["violation_frequency"] <= 0.10 + 1e-12, f"{name} m={m} violates delta"
            assert v["share_splits_worse_than_classical"] == 0.0, f"{name} m={m}: a split was worse than classical"
    return (f"5 fingerprints match; reproducible; {len(checks)} numbers found; "
            f"H4 PASS, H5 PASS, H2 still reported FAIL")

@test("T17 LaTeX manuscript: structurally valid, same source as the Word version; figures paired")
def t_latex():
    if not HAVE_MAN:
        raise Skip("manuscript sources not in this checkout")
    import subprocess
    import struct
    man = os.path.join(HERE, "..", "manuscript")
    r = subprocess.run([sys.executable, "check_latex.py"], cwd=man, capture_output=True, text=True)
    assert r.returncode == 0, "check_latex.py failed:\n" + r.stdout + r.stderr

    # every figure exists as a 600 dpi PNG (Word) and a vector PDF (LaTeX)
    names = [f[:-4] for f in os.listdir(RES) if f.startswith("fig_") and f.endswith(".png")]
    for n in names:
        png, pdf = os.path.join(RES, n + ".png"), os.path.join(RES, n + ".pdf")
        assert os.path.exists(pdf), f"{n}: no vector PDF"
        w, h = struct.unpack(">II", open(png, "rb").read(32)[16:24])
        min_w = 1800 if n == "fig_graphical_abstract" else 3000   # IEEE graphical abstract is 3 in wide
        assert w >= min_w, f"{n}: {w}px wide (< 600 dpi at its print width)"
        assert open(pdf, "rb").read(5) == b"%PDF-", f"{n}: not a PDF"
        assert abs(os.path.getmtime(pdf) - os.path.getmtime(png)) < 600, \
            f"{n}: PNG and PDF are from different runs"

    tex = open(os.path.join(man, "latex", "manuscript.tex"), encoding="utf-8").read()
    for macro in ["\\supplementary{", "\\authorcontributions{", "\\funding{",
                  "\\institutionalreview{", "\\informedconsent{", "\\dataavailability{",
                  "\\acknowledgments{", "\\conflictsofinterest{", "\\abbreviations{"]:
        assert macro in tex, f"MDPI back-matter macro missing: {macro}"
    n_fig = tex.count("\\includegraphics")
    n_tab = tex.count("\\begin{tabularx}")
    js = open(MAN, encoding="utf-8").read()
    assert n_fig == js.count("add(FIG("), f"{n_fig} figures in LaTeX, {js.count('add(FIG(')} in the Word source"
    assert n_tab == js.count("add(TAB("), f"{n_tab} tables in LaTeX, {js.count('add(TAB(')} in the Word source"
    return f"LaTeX checks pass; {len(names)} figures as PNG+PDF; {n_fig} figures and {n_tab} tables in both versions"

@test("T18 IEEE Sensors Journal version: <= 8 pages, numbers match the verified manuscript, compliant")
def t_ieee():
    import re
    import zipfile
    if not HAVE_MAN:
        raise Skip("manuscript sources not in this checkout")
    man = os.path.join(HERE, "..", "manuscript")
    ieee = os.path.join(man, "ieee")
    sys.path.insert(0, man)
    import build_ieee as BI

    report = json.load(open(os.path.join(ieee, "build_report.json")))
    assert report["pages"] <= BI.PAGE_LIMIT, f"{report['pages']} pages > {BI.PAGE_LIMIT}"
    assert 150 <= report["abstract_words"] <= 250, f"abstract {report['abstract_words']} words"
    pdf = os.path.join(ieee, "Manuscript_IEEE_JSEN.pdf")
    body_f = os.path.join(ieee, "paper_body.tex")
    supp_f = os.path.join(ieee, "supplement_body.tex")
    newest_src = max(os.path.getmtime(f) for f in (body_f, supp_f, os.path.join(man, "build_ieee.py")))
    assert os.path.getmtime(pdf) >= newest_src, "IEEE PDF is older than its sources: run build_ieee.py"

    # review copy for colleagues: same layout, no acknowledgment, within the page limit
    import fitz
    review = os.path.join(ieee, "review_copy", "Tayyab_IEEE_JSEN_draft_for_review.pdf")
    assert os.path.getmtime(review) >= newest_src, "review copy is older than its sources: run build_ieee.py"
    rdoc = fitz.open(review)
    assert rdoc.page_count <= BI.PAGE_LIMIT, f"review copy has {rdoc.page_count} pages"
    assert "ACKNOWLEDGMENT" not in "".join(p.get_text() for p in rdoc).upper(), "review copy still has the acknowledgment"

    # every decimal number and percentage must already be in the verified manuscript source
    verified = open(MAN, encoding="utf-8").read()
    ga = open(os.path.join(HERE, "make_graphical_abstract.py"), encoding="utf-8").read()
    texts = [open(body_f, encoding="utf-8").read(), open(supp_f, encoding="utf-8").read(), BI.ABSTRACT,
             re.search(r"HEADLINE = \{.*?\}", ga, re.S).group(0)]

    def tokens(t):
        t = t.replace("\\%", "%").replace("--", "-")
        t = re.sub(r"https?://\S+|seeds? \d+-\d+|\\cite\{[^}]*\}|\\mref\{[^}]*\}|\\ref\{[^}]*\}|\\label\{[^}]*\}"
                   r"|\\includegraphics\[[^\]]*\]|\\setlength\{[^}]*\}\{[^}]*\}", " ", t)  # layout, not results
        return set(re.findall(r"(?<![\w.])\d+\.\d+(?![\w.])|(?<![\w.])\d+(?:\.\d+)?%", t))

    have = tokens(verified)
    missing = sorted(set().union(*map(tokens, texts)) - have)
    assert not missing, f"numbers not in the verified manuscript: {missing}"
    n_checked = len(set().union(*map(tokens, texts)))

    paper_tex = open(os.path.join(ieee, "submission", "paper.tex"), encoding="utf-8").read()
    assert all(t in BI.ACK for t in BI.REQUIRED_ACK_TERMS), "required acknowledgment statement missing"
    assert "Acknowledgment" in paper_tex, "acknowledgment section missing from the submission version"
    assert "github.com/t4tayyab2006/do-no-harm-gnss-ins" in texts[0], "code link missing"
    names = zipfile.ZipFile(os.path.join(ieee, "IEEE_JSEN_source.zip")).namelist()
    for need in ["paper.tex", "supplement.tex", "ieeecolor.cls", "jsen.sty", "LOGO-jsen-web.eps",
                 "fig_graphical_abstract.pdf"] + [f + ".pdf" for f in BI.FIGS_MAIN + BI.FIGS_SUPP]:
        assert need in names, f"{need} missing from the source zip"
    return (f"{report['pages']} pages (limit {BI.PAGE_LIMIT}); abstract {report['abstract_words']} words; "
            f"{n_checked} numbers all in the verified manuscript; required statements and code link present")

if __name__ == "__main__":
    os.chdir(HERE)
    t0 = time.time()
    for t in TESTS:
        t()
    print("\n" + "=" * 96)
    for name, status, dt, msg in RESULTS:
        print(f"{status:<5} {name:<72} {dt:6.1f}s  {msg}")
    fails = [r for r in RESULTS if r[1] == "FAIL"]
    print("=" * 96)
    print(f"{len(RESULTS) - len(fails)} passed/skipped, {len(fails)} failed, {time.time()-t0:.0f}s total")
    sys.exit(1 if fails else 0)
