"""
train_eval.py
-------------
Controlled comparison of classical and learned adaptive-covariance filters for
IMU/GNSS fusion, against an oracle-R upper bound.

METHODOLOGY NOTES (these were mistakes in earlier versions, fixed here):

1. FAIR BASELINE. The classical EKF's process noise Q is now TUNED on the
   validation set, exactly like every other method's hyperparameters. Earlier
   runs used a hand-picked Q; tuning it improved the baseline by 12.6%, meaning
   the previously reported comparisons were against a strawman. The tuned Q is
   then used by EVERY method so all comparisons are apples-to-apples.

2. SELECTION OVERFITTING. Earlier runs swept 27 adaptation configs against only
   6 validation trajectories; the winner generalized badly (best on validation,
   worst on test). Validation is enlarged and the sweep trimmed.

3. TEST SET SIZE. 8 test trajectories let a single divergence dominate the mean.
   Enlarged so means and tail statistics mean something.

4. The null option (no adaptation) is always included in model selection, so if
   adaptation does not help, that is the reported outcome.
"""
import json
import numpy as np
import torch
import torch.nn as nn

from simulate import make_dataset, DT
from ekf import run_ekf
from features import build_ankf_features, build_lstm_inputs, decimate_array
from models import LSTMOnly, ANKFCorrector
from ankf import collect_fix_dataset, train_detector, make_classical_iae_fn

torch.manual_seed(0)
np.random.seed(0)

TRAIN_SEEDS = list(range(0, 30))
VAL_SEEDS = list(range(100, 120))
TEST_SEEDS = list(range(200, 240))
DURATION_S = 90.0

QCFG = {"q_accel": 0.05, "q_gyro": 0.02}  # replaced by tune_Q() on validation


def ekf(sens, n, **kw):
    """All filter runs go through here so every method shares the tuned Q."""
    return run_ekf(sens, n, q_accel=QCFG["q_accel"], q_gyro=QCFG["q_gyro"], **kw)


def traj_rmse(out, ref):
    return float(np.sqrt(np.mean((out["x"] - ref["x"]) ** 2 + (out["y"] - ref["y"]) ** 2)))


def tune_Q(val_raw):
    """Tune the classical filter's process noise on VALIDATION. The baseline
    gets the same tuning budget as the learned methods -- otherwise any reported
    improvement partly just reflects a badly-tuned baseline."""
    print("Tuning classical EKF process noise Q on validation...")
    best, best_v = None, float("inf")
    for qa in [0.02, 0.05, 0.1, 0.2, 0.35, 0.5]:
        for qg in [0.005, 0.02, 0.05, 0.2]:
            errs = [traj_rmse(run_ekf(s, len(r["t"]), q_accel=qa, q_gyro=qg), r) for r, s in val_raw]
            v = float(np.mean(errs))
            if v < best_v:
                best_v, best = v, (qa, qg)
    print(f"  tuned Q: q_accel={best[0]}, q_gyro={best[1]}  (val RMSE {best_v:.3f})")
    return {"q_accel": best[0], "q_gyro": best[1]}, best_v


def build_split(seeds):
    data = []
    for s in seeds:
        ref, sens = make_dataset(seed=s, duration_s=DURATION_S)
        n = len(ref["t"])
        ekf_out = ekf(sens, n)
        ankf_feats, ankf_target = build_ankf_features(ref, sens, ekf_out)
        lstm_feats, lstm_target = build_lstm_inputs(ref, sens)
        data.append(dict(seed=s, ref=ref, sens=sens, ekf=ekf_out,
                         ankf_feats=ankf_feats, ankf_target=ankf_target,
                         lstm_feats=lstm_feats, lstm_target=lstm_target,
                         outage_dec=decimate_array(sens["outage_mask_full"]).astype(np.float32)))
    return data


def to_batch(data, kf, kt):
    return (torch.tensor(np.stack([d[kf] for d in data])),
            torch.tensor(np.stack([d[kt] for d in data])))


def train_model(model, X, Y, Xval, Yval, epochs, lr, tag="",
                sample_weight=None, sample_weight_val=None, patience=60):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=300, gamma=0.5)

    def wmse(pred, target, w):
        se = (pred - target) ** 2
        return se.mean() if w is None else (se * w).sum() / (w.sum() * se.shape[-1])

    best_val, best_state, stale = float("inf"), None, 0
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        loss = wmse(model(X), Y, sample_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); sched.step()
        model.eval()
        with torch.no_grad():
            vloss = wmse(model(Xval), Yval, sample_weight_val).item()
        if vloss < best_val - 1e-6:
            best_val, stale = vloss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if (ep + 1) % 50 == 0 or ep == 0:
            print(f"  [{tag}] epoch {ep+1:4d}/{epochs}  train={loss.item():.5f}  val={vloss:.5f}  best={best_val:.5f}")
        if stale >= patience:
            print(f"  [{tag}] early stop at epoch {ep+1}")
            break
    model.load_state_dict(best_state)
    return model


def rmse(e):
    return float(np.sqrt(np.mean(e ** 2)))


def evaluate(test_data, lstm_model, ankf_model, policies):
    rows = []
    for d in test_data:
        ref, ekf_out, n_full = d["ref"], d["ekf"], len(d["ref"]["t"])
        rx, ry = decimate_array(ref["x"]), decimate_array(ref["y"])
        outage = decimate_array(d["sens"]["outage_mask_full"])

        preds = {"ekf": (decimate_array(ekf_out["x"]), decimate_array(ekf_out["y"]))}
        for name, make_fn in policies.items():
            out = ekf(d["sens"], n_full, r_adapt_fn=make_fn())
            preds[name] = (decimate_array(out["x"]), decimate_array(out["y"]))
        o_out = ekf(d["sens"], n_full, r_gps_base=None)
        preds["oracle"] = (decimate_array(o_out["x"]), decimate_array(o_out["y"]))

        with torch.no_grad():
            lp = lstm_model(torch.tensor(d["lstm_feats"])[None]).numpy()[0] * 100.0
            corr = ankf_model(torch.tensor(d["ankf_feats"])[None]).numpy()[0]
        preds["lstm"] = (lp[:, 0], lp[:, 1])
        preds["resid"] = (preds["ekf"][0] + corr[:, 0], preds["ekf"][1] + corr[:, 1])

        row = {"seed": d["seed"]}
        for k, (px, py) in preds.items():
            e = np.sqrt((px - rx) ** 2 + (py - ry) ** 2)
            row[f"{k}_rmse"] = rmse(e)
            row[f"{k}_outage_rmse"] = rmse(e[outage]) if outage.any() else float("nan")
        rows.append(row)
    return rows


def main():
    global QCFG
    val_raw = [make_dataset(seed=s, duration_s=DURATION_S) for s in VAL_SEEDS]
    QCFG, tuned_val = tune_Q(val_raw)

    print(f"\nBuilding splits ({len(TRAIN_SEEDS)} train / {len(VAL_SEEDS)} val / {len(TEST_SEEDS)} test)...")
    train_data, val_data, test_data = build_split(TRAIN_SEEDS), build_split(VAL_SEEDS), build_split(TEST_SEEDS)

    Xl_tr, Yl_tr = to_batch(train_data, "lstm_feats", "lstm_target")
    Xl_va, Yl_va = to_batch(val_data, "lstm_feats", "lstm_target")
    Xa_tr, Ya_tr = to_batch(train_data, "ankf_feats", "ankf_target")
    Xa_va, Ya_va = to_batch(val_data, "ankf_feats", "ankf_target")

    print("\nTraining pure end-to-end LSTM baseline...")
    lstm_model = train_model(LSTMOnly(5, 64, 2), Xl_tr, Yl_tr, Xl_va, Yl_va,
                             epochs=250, lr=3e-3, tag="LSTM-only", patience=60)

    w_tr = (1.0 + 5.0 * torch.tensor(np.stack([d["outage_dec"] for d in train_data]))).unsqueeze(-1)
    w_va = (1.0 + 5.0 * torch.tensor(np.stack([d["outage_dec"] for d in val_data]))).unsqueeze(-1)
    print("\nTraining ANKF v1 residual corrector (retained as negative result)...")
    ankf_model = train_model(ANKFCorrector(6, 32), Xa_tr, Ya_tr, Xa_va, Ya_va,
                             epochs=300, lr=1e-2, tag="ANKF-v1",
                             sample_weight=w_tr, sample_weight_val=w_va, patience=100)

    print("\nTraining degradation detector...")
    Xf_tr, Yf_tr = collect_fix_dataset(train_data)
    Xf_va, Yf_va = collect_fix_dataset(val_data)
    print(f"  {len(Xf_tr)} train fixes ({Yf_tr.sum():.0f} degraded), {len(Xf_va)} val fixes ({Yf_va.sum():.0f} degraded)")
    detector = train_detector(Xf_tr, Yf_tr, Xf_va, Yf_va)

    def val_rmse_of(make_fn):
        return float(np.mean([traj_rmse(ekf(d["sens"], len(d["ref"]["t"]), r_adapt_fn=make_fn()), d["ref"])
                              for d in val_data]))

    print("\nSelecting adaptation policy on validation (test never touched)...")
    null_val = val_rmse_of(lambda: None)
    print(f"  [null] no adaptation                 val_rmse={null_val:.3f}")
    results_val = {"none": null_val}

    best_iae, iae_cfg = float("inf"), None
    for vi in [4.0, 9.0, 16.0]:
        v = val_rmse_of(lambda vi=vi: make_classical_iae_fn(vi))
        print(f"  [classical IAE] vi={vi:5.1f}            val_rmse={v:.3f}")
        if v < best_iae:
            best_iae, iae_cfg = v, (vi,)
    results_val["classical_iae"] = best_iae

    best_v2, v2_cfg = float("inf"), None
    for vi in [4.0, 9.0]:
        v = val_rmse_of(lambda vi=vi: detector.make_r_scale_fn(vi, 1.0))
        print(f"  [learned, unsafeguarded] vi={vi:5.1f}   val_rmse={v:.3f}")
        if v < best_v2:
            best_v2, v2_cfg = v, (vi, 1.0)
    results_val["learned_unsafeguarded"] = best_v2

    best_v3, v3_cfg = float("inf"), None
    for vi in [4.0, 9.0]:
        for tg in [None, 2.5]:
            for mc in [None, 3]:
                v = val_rmse_of(lambda vi=vi, tg=tg, mc=mc: detector.make_safeguarded_fn(vi, 1.0, tg, mc))
                if v < best_v3:
                    best_v3, v3_cfg = v, (vi, tg, mc)
    print(f"  [learned + safeguard] best {v3_cfg}  val_rmse={best_v3:.3f}")
    results_val["learned_safeguarded"] = best_v3

    winner = min(results_val, key=results_val.get)
    print(f"  -> best policy on validation: {winner} ({results_val[winner]:.3f}); null was {null_val:.3f}")

    policies = {
        "iae": lambda: make_classical_iae_fn(*iae_cfg),
        "ankf": lambda: detector.make_r_scale_fn(*v2_cfg),
        "v3": lambda: detector.make_safeguarded_fn(v3_cfg[0], 1.0, v3_cfg[1], v3_cfg[2]),
    }

    print("\nEvaluating on held-out test trajectories...")
    rows = evaluate(test_data, lstm_model, ankf_model, policies)

    methods = [("Classical EKF (tuned Q, fixed R)", "ekf"),
               ("Classical IAE adaptive (no ML)", "iae"),
               ("LSTM-only (no physics)", "lstm"),
               ("ANKF v1 residual-corr", "resid"),
               ("ANKF v2 learned-R", "ankf"),
               ("ANKF v3 learned-R + safeguard", "v3"),
               ("[oracle-R upper bound]", "oracle")]

    summary = dict(n_test=len(rows), n_train=len(TRAIN_SEEDS), n_val=len(VAL_SEEDS),
                   tuned_Q=QCFG, tuned_Q_val_rmse=tuned_val,
                   validation_selection=results_val, winner_on_validation=winner,
                   cfg=dict(iae=list(iae_cfg), v2=list(v2_cfg),
                            v3=[v3_cfg[0], v3_cfg[1], v3_cfg[2]]),
                   per_trajectory=rows)

    for _, k in methods:
        vals = [r[f"{k}_rmse"] for r in rows]
        summary[f"{k}_rmse"] = float(np.mean(vals))
        summary[f"{k}_median_rmse"] = float(np.median(vals))
        summary[f"{k}_outage_rmse"] = float(np.mean([r[f"{k}_outage_rmse"] for r in rows]))
        ratios = [r[f"{k}_rmse"] / r["ekf_rmse"] for r in rows]
        summary[f"{k}_worst_ratio_vs_ekf"] = float(np.max(ratios))
        summary[f"{k}_wins_vs_ekf"] = int(sum(1 for x in ratios if x < 1.0))

    print(f"\n===== TEST RESULTS ({len(rows)} held-out trajectories) =====")
    print(f"{'Method':<34}{'RMSE':>8}{'median':>9}{'outage':>9}{'wins':>8}{'worst':>9}")
    for name, k in methods:
        w = "  --" if k == "ekf" else f"{summary[k+'_wins_vs_ekf']:>3}/{len(rows)}"
        print(f"{name:<34}{summary[k+'_rmse']:>8.2f}{summary[k+'_median_rmse']:>9.2f}"
              f"{summary[k+'_outage_rmse']:>9.2f}{w:>8}{summary[k+'_worst_ratio_vs_ekf']:>8.2f}x")

    base = summary["ekf_rmse"]
    print("\nvs tuned classical EKF (mean RMSE):")
    for name, k in methods[1:]:
        print(f"  {name:<34}{100*(base-summary[k+'_rmse'])/base:+8.1f}%")
    print(f"\nOracle headroom: {100*(base-summary['oracle_rmse'])/base:.1f}%")

    with open("../results/results.json", "w") as f:
        json.dump(summary, f, indent=2)
    torch.save(lstm_model.state_dict(), "../results/lstm_only.pt")
    torch.save(ankf_model.state_dict(), "../results/ankf_corrector.pt")
    torch.save(detector.state_dict(), "../results/ankf_detector.pt")
    print("\nWrote ../results/results.json")
    return summary


if __name__ == "__main__":
    main()
