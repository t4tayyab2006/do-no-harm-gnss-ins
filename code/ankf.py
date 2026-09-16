"""
ankf.py
-------
ANKF (Adaptive Neuro-Kalman Fusion) -- the proposed hybrid, v2.

WHY v2: the first design (a GRU predicting the EKF's *position residual* from
causal features, kept in git history / documented as a negative result) failed
to generalize -- test RMSE was 14.6% WORSE than the classical EKF while the
training loss fell steadily. The reason is fundamental, not a tuning problem:
during multipath the EKF's position residual is dominated by the injected
zero-mean GPS noise, which is information-theoretically unpredictable. The
network could only memorize training-set noise realizations.

The literature does not do this. KalmanNet, DeepUKF-VIN and the adaptive-UKF
work all learn the *Kalman gain / noise covariance*, not the residual. v2
follows that: a small network learns to DETECT the degraded-measurement regime
from the innovation statistics, and inflates R accordingly inside the filter.

This is learnable for a sound reason: an individual GPS error is unpredictable,
but the *regime* (sigma = 2.5 m open-sky vs sigma = 12 m multipath) is easily
detectable from the normalized innovation squared (NIS), whose expectation is
dim(z) = 2 under a correct noise model and >> 2 when R is understated. And
inflating R in that regime provably reduces filter error -- this is classical
adaptive Kalman filtering, with the detection step learned rather than
hand-thresholded.
"""
import numpy as np
import torch
import torch.nn as nn

FEAT_DIM = 5      # see ekf.run_ekf: [log1p(NIS), log1p(mean NIS hist), log1p(max NIS hist), |y|/10, log1p(tr P)]
MAX_R_SCALE = 6.0  # R can be inflated up to 6x in std (36x in variance)


class DegradationDetector(nn.Module):
    """Maps causal innovation statistics of a GPS fix -> P(fix is degraded).
    Deliberately small (a few hundred parameters): the decision is a regime
    classification, not a memorization task, so capacity is not the bottleneck
    and a small model generalizes from few trajectories."""

    def __init__(self, in_dim=FEAT_DIM, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # logit

    def prob(self, feat_np):
        with torch.no_grad():
            return float(torch.sigmoid(self.forward(torch.tensor(feat_np, dtype=torch.float32))))

    def make_safeguarded_fn(self, var_inflation, sharpen, trace_gate=None, max_consecutive=None):
        """ANKF v3: the learned adaptation, wrapped in two safeguards derived
        from the identifiability problem.

        A large innovation is AMBIGUOUS: it can mean the measurement is bad
        (-> inflate R, correct) or that the STATE estimate is bad (-> trust the
        measurement more, the opposite action). Innovation-based adaptation
        cannot distinguish these -- this is the classical Q/R identifiability
        problem (see papers/NOVELTY_ASSESSMENT.md). Unsafeguarded, it produces a
        divergence loop: detector fires -> R inflated -> GPS ignored -> IMU
        drifts -> innovations grow -> detector fires harder. We observed exactly
        this (test seed 204: 2.23x WORSE than the classical filter, while 7/8
        other trajectories improved).

        Safeguard 1 (trace_gate): suppress adaptation when the filter's own
        covariance is already large -- there the ambiguity resolves toward
        "bad state", so inflating R is the wrong action.
        Safeguard 2 (max_consecutive): bound how many consecutive fixes may be
        down-weighted, so the filter can never ignore GPS indefinitely.

        Returns a FRESH stateful closure per call (per-trajectory state).
        """
        state = {"consec": 0}

        def fn(feat_np):
            p = self.prob(feat_np) ** sharpen
            # feat index 4 is log1p(trace P) -- see ekf.run_ekf
            if trace_gate is not None and feat_np[4] > trace_gate:
                p = 0.0
            if p > 0.5:
                state["consec"] += 1
            else:
                state["consec"] = 0
            if max_consecutive is not None and state["consec"] > max_consecutive:
                p = 0.0
            return float(np.sqrt(1.0 + p * (var_inflation - 1.0)))

        return fn

    def make_r_scale_fn(self, var_inflation, sharpen):
        """Returns the R-std scaling callback plugged into the EKF.

        var_inflation: how much larger the degraded-regime variance is assumed
            to be than nominal. Fitted on TRAINING/VALIDATION trajectories only.
        sharpen: exponent applied to the detector probability. The first
            version used the raw sigmoid (sharpen=1), which inflated R even on
            clean fixes the detector was merely unsure about (p~0.3 -> 2.5x
            std) -- the filter then under-used good GPS everywhere and overall
            RMSE got worse despite 93% detection accuracy. Sharpening
            concentrates the inflation on confident detections.

        Mixture form: E[var] = (1-p)*var_nom + p*var_degraded, so the std
        scale is sqrt(1 + p*(var_inflation - 1)).
        """
        def fn(feat_np):
            p = self.prob(feat_np) ** sharpen
            return float(np.sqrt(1.0 + p * (var_inflation - 1.0)))
        return fn


def make_classical_iae_fn(var_inflation, nis_threshold=9.21):
    """Classical (NON-learned) innovation-based adaptive estimation baseline.

    This is the baseline a learned adaptive method must actually beat. Chi-square
    consistency test on the NIS: under a correct noise model NIS ~ chi2 with
    dim(z)=2 degrees of freedom, so NIS > 9.21 is the 99th percentile -- flag the
    fix as inconsistent and inflate R. No learning, two hand-set constants.

    Omitting this baseline would make "we beat the fixed-R EKF" a misleadingly
    weak claim: most of the available gain may be obtainable without any
    learning at all.
    """
    def fn(feat_np):
        nis = np.expm1(feat_np[0])  # feat[0] = log1p(NIS)
        return float(np.sqrt(var_inflation)) if nis > nis_threshold else 1.0
    return fn


def make_classical_quality_fn(var_inflation, q_threshold, nis_threshold=9.21):
    """Classical rule that ALSO uses the receiver quality indicator (feature 5):
    inflate R if the chi-square test fails OR the indicator exceeds a threshold.
    This is the fair classical counterpart to a learned model that sees the
    indicator -- the standard engineering practice of down-weighting fixes with
    poor reported quality (e.g. high HDOP). q_threshold = inf recovers pure IAE."""
    def fn(feat_np):
        nis = np.expm1(feat_np[0])
        bad = nis > nis_threshold or (len(feat_np) > 5 and feat_np[5] > q_threshold)
        return float(np.sqrt(var_inflation)) if bad else 1.0
    return fn


def collect_fix_dataset(datasets):
    """Build the (features, label) training set for the detector by running the
    NOMINAL classical EKF and labelling each fix with whether it actually fell
    in a degraded window. The label is available at TRAINING time only (we
    injected the degradation); at test time the detector sees innovations only."""
    X, Y = [], []
    for d in datasets:
        sens, ekf_out = d["sens"], d["ekf"]
        outage = sens["outage_mask_full"]
        for rec in ekf_out["fix_records"]:
            X.append(rec["feat"])
            Y.append(1.0 if outage[rec["k"]] else 0.0)
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def train_detector(Xtr, Ytr, Xva, Yva, epochs=400, lr=5e-3, verbose=True):
    model = DegradationDetector(in_dim=Xtr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # class imbalance: degraded fixes are the minority
    pos_weight = torch.tensor(max(1.0, (len(Ytr) - Ytr.sum()) / max(Ytr.sum(), 1.0)))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    Xtr_t, Ytr_t = torch.tensor(Xtr), torch.tensor(Ytr)
    Xva_t, Yva_t = torch.tensor(Xva), torch.tensor(Yva)

    best_val, best_state = float("inf"), None
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        loss = loss_fn(model(Xtr_t), Ytr_t)
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vloss = loss_fn(model(Xva_t), Yva_t).item()
            vacc = ((torch.sigmoid(model(Xva_t)) > 0.5).float() == Yva_t).float().mean().item()
        if vloss < best_val:
            best_val, best_state = vloss, {k: v.clone() for k, v in model.state_dict().items()}
        if verbose and ((ep + 1) % 100 == 0 or ep == 0):
            print(f"  [detector] epoch {ep+1:4d}/{epochs}  train={loss.item():.4f}  val={vloss:.4f}  val_acc={vacc:.3f}")
    model.load_state_dict(best_state)
    return model
