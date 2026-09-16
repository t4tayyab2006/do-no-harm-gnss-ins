"""
gated.py
--------
Proposed method: CERTIFIED SOLUTION-SEPARATION GATING of a learned adaptive filter.

Two filters run in lockstep on the same IMU/GNSS stream:
  F_C  classical adaptive filter (chi-square IAE)  -- the trusted fallback
  F_L  learned adaptive filter (learned R inflation) -- better on average,
       but with an unbounded tail (3.15x blow-up observed; see paper_draft.md)

At every GNSS epoch a ground-truth-free SEPARATION statistic compares them:

    d = || p_L - p_C ||  /  sqrt(tr(P_C,pos) + tr(P_L,pos))

(p = position, P_pos = position covariance block). This mirrors classical
solution-separation integrity monitoring, but separates a LEARNED solution from
a CLASSICAL one instead of sensor subsets. If d > lambda, the output falls back
to F_C and F_L is RESET to F_C's state, which breaks the innovation-driven
divergence loop (a diverged learned filter otherwise keeps feeding its own large
innovations back into the detector).

The threshold lambda is not set from a Gaussian assumption (which a learned
covariance violates); it is calibrated by Learn-then-Test in certify.py, giving
a finite-sample, distribution-free bound on how often the gated system is worse
than the classical fallback.
"""
import numpy as np
from simulate import DT, GPS_NOISE_STD

STATE_DIM = 6
H = np.zeros((2, STATE_DIM)); H[0, 0] = 1; H[1, 1] = 1


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _floor_psd(M, floor):
    """Symmetrize and floor the eigenvalues of a 2x2 covariance estimate."""
    M = 0.5 * (M + M.T)
    w, V = np.linalg.eigh(M)
    return (V * np.maximum(w, floor)) @ V.T


class SageHusaR:
    """Classical Sage-Husa adaptive estimate of the measurement covariance, with
    fading factor b:  R_k = (1 - d_k) R_{k-1} + d_k (y y^T - H P^- H^T),
    d_k = (1 - b) / (1 - b^{k+1}). Eigenvalues floored for positive definiteness
    (the standard safeguard against the estimate going indefinite)."""

    def __init__(self, b, r0=GPS_NOISE_STD ** 2, floor=1.0):
        self.b, self.R, self.k, self.floor = b, np.eye(2) * r0, 0, floor

    def estimate(self, y, HPH):
        d = (1 - self.b) / (1 - self.b ** (self.k + 1))
        self.R = _floor_psd((1 - d) * self.R + d * (np.outer(y, y) - HPH), self.floor)
        self.k += 1
        return self.R


class CovMatchR:
    """Innovation-based covariance matching (Mohamed & Schwarz 1999):
    R_k = mean_{last N}(y y^T) - H P^- H^T, floored; nominal R until N innovations."""

    def __init__(self, N, r0=GPS_NOISE_STD ** 2, floor=1.0):
        self.N, self.r0, self.floor, self.buf = N, r0, floor, []

    def estimate(self, y, HPH):
        self.buf.append(np.outer(y, y))
        self.buf = self.buf[-self.N:]
        if len(self.buf) < self.N:
            return np.eye(2) * self.r0
        return _floor_psd(np.mean(self.buf, axis=0) - HPH, self.floor)


class StepEKF:
    """Step-able copy of ekf.run_ekf (identical math), so two filters can run in
    lockstep and one can be reset to the other's state mid-trajectory."""

    def __init__(self, q_accel, q_gyro, q_bias=1e-5, r_std=GPS_NOISE_STD, r_adapt_fn=None,
                 x0=None, P0=None, r_estimator=None):
        self.x = np.zeros(STATE_DIM) if x0 is None else np.array(x0, dtype=float)
        self.P = np.diag([1.0, 1.0, 0.1, 1.0, 0.01, 0.01]) if P0 is None else np.array(P0, dtype=float)
        self.Q = np.diag([0, 0, q_gyro * DT, q_accel * DT, q_bias, q_bias])
        self.r_std = r_std
        self.r_adapt_fn = r_adapt_fn
        self.r_estimator = r_estimator  # classical full-R estimator (Sage-Husa / covariance matching)
        self.nis_history = []

    def predict(self, accel, gyro):
        x = self.x
        theta, v, ab, gb = x[2], x[3], x[4], x[5]
        xp = x.copy()
        xp[2] = wrap_angle(theta + (gyro - gb) * DT)
        xp[3] = v + (accel - ab) * DT
        c, s = np.cos(xp[2]), np.sin(xp[2])
        xp[0] = x[0] + xp[3] * c * DT
        xp[1] = x[1] + xp[3] * s * DT
        F = np.eye(STATE_DIM)
        F[2, 5] = -DT; F[3, 4] = -DT
        F[0, 2] = -xp[3] * s * DT; F[0, 3] = c * DT
        F[1, 2] = xp[3] * c * DT;  F[1, 3] = s * DT
        self.P = F @ self.P @ F.T + self.Q
        self.x = xp

    def update(self, z, true_std=None, quality=None):
        """quality: optional receiver quality indicator for this fix; if given it
        is appended as feature 5 (after the five innovation statistics).
        Returns the feature vector the adaptation function saw."""
        std = self.r_std if true_std is None else true_std
        R = np.diag([std ** 2, std ** 2])
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        nis = float(y @ np.linalg.solve(S, y))
        hist = self.nis_history
        h3 = (hist + [0.0, 0.0, 0.0])[-3:] if len(hist) < 3 else hist[-3:]
        feat = [np.log1p(nis), np.log1p(np.mean(h3)), np.log1p(max(h3)),
                np.linalg.norm(y) / 10.0, np.log1p(np.trace(self.P))]
        if quality is not None:
            feat.append(float(quality))
        feat = np.array(feat)
        hist.append(nis)
        if self.r_adapt_fn is not None:
            R = R * float(self.r_adapt_fn(feat)) ** 2
            S = H @ self.P @ H.T + R
        elif self.r_estimator is not None:
            R = self.r_estimator.estimate(y, H @ self.P @ H.T)
            S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(STATE_DIM) - K @ H) @ self.P
        return feat

    def reset_to(self, other):
        self.x = other.x.copy()
        self.P = other.P.copy()
        self.nis_history = list(other.nis_history)  # else stale large NIS re-triggers the detector


def separation(fl, fc):
    dp = fl.x[:2] - fc.x[:2]
    pl = fl.P[:2, :2] if fl.P is not None else fc.P[:2, :2]  # covariance-free learned filter
    scale = np.sqrt(np.trace(pl) + np.trace(fc.P[:2, :2]))
    return float(np.linalg.norm(dp) / scale)


def run_single(sens, n, q, make_fn=None, oracle=False, collect=False, make_estimator=None):
    """One filter over a trajectory. make_fn=None -> fixed R; oracle -> true
    per-fix sigma; make_estimator -> classical full-R estimator (e.g. SageHusaR).
    collect=True also returns (k, feature) per available fix, used to build the
    detector's training set from a NOMINAL (non-adaptive) run."""
    f = StepEKF(q["q_accel"], q["q_gyro"], q_bias=q.get("q_bias", 1e-5),
                r_adapt_fn=make_fn() if make_fn else None, x0=sens.get("x0"), P0=sens.get("P0"),
                r_estimator=make_estimator() if make_estimator else None)
    gps = {idx: j for j, idx in enumerate(sens["gps_idx"])}
    acc, gyr = sens["accel_meas"], sens["gyro_meas"]
    track = np.zeros((n, 2)); feats = []
    for k in range(n):
        f.predict(acc[k], gyr[k])
        if k in gps:
            j = gps[k]
            if sens["gps_available"][j]:
                z = np.array([sens["gps_x"][j], sens["gps_y"][j]])
                qj = sens["gps_quality"][j] if "gps_quality" in sens else None
                feat = f.update(z, true_std=sens["gps_std"][j] if oracle else None, quality=qj)
                if collect:
                    feats.append((k, feat))
        track[k] = f.x[:2]
    return (track, feats) if collect else track


GATE_STATS = ("sep_norm", "sep_m", "pred_resid", "gauss_ss", "pred_switch")
# pred_switch: the IMM-style predictive rule made certifiable. At every GNSS epoch the
# output is F_L iff s_ema <= lam, where s_ema is the EMA of the difference of squared
# PRE-update predicted residuals (as in pred_resid); no reset by default. lam = -inf is
# pure classical (harm 0, the certifiable fallback), lam = +inf pure learned, lam = 0
# "use whichever filter predicted better"; negative lam demands the learned filter be
# clearly better. Unlike sep_m it judges WHO is right, not merely whether they disagree.
CHI2_2DOF_999 = 13.8155  # 99.9% quantile of chi-square(2): the usual integrity-style threshold


def run_gated(sens, n, q, make_learned_fn, make_classical_fn, lam, reset=True,
              stat="sep_norm", beta=0.7, learned_is_oracle=False, make_learned_filter=None):
    """Runs F_C and F_L in lockstep with a gate at threshold `lam`.
    lam = 0  -> always output F_C (pure classical fallback)
    lam = inf -> always output F_L (pure learned, never gated)

    Gate statistics (all ground-truth-free and causal):
      sep_norm   position separation / sqrt(tr P_L + tr P_C)   [first version]
                 -- blind during outages: both covariances balloon, so large
                 disagreements look small exactly when divergence happens.
      sep_m      position separation in metres.
      pred_resid which filter predicts the incoming GNSS fix better: EMA over
                 fixes of (|z - p_L|^2 - |z - p_C|^2) / R_nom, using PRE-update
                 predicted positions. Judges who is right, not merely whether
                 the two disagree (cf. IMM model likelihoods).
      gauss_ss   classical Gaussian solution separation: squared Mahalanobis
                 distance dp^T (P_L,pos + P_C,pos)^-1 dp, used with the fixed
                 chi-square threshold CHI2_2DOF_999 (benchmark; assumes both
                 covariances are correct). A covariance-free learned filter
                 (e.g. KalmanNet) contributes 2 P_C,pos instead.

    make_learned_filter: optional factory returning ANY learned filter object with
    predict(acc, gyr), update(z, quality=...), reset_to(other), attributes x and P
    (P may be None) -- e.g. a KalmanNet-style filter. Overrides make_learned_fn.

    Returns gated / learned / classical tracks and the fraction of GNSS epochs
    at which the gate was open (learned output in use)."""
    # learned_is_oracle: the "learned" branch is a PERFECT noise predictor (told
    # each fix's true sigma). Used to test pass-through: does the gate keep the
    # gains of a component that genuinely has them?
    qb = q.get("q_bias", 1e-5)
    fc = StepEKF(q["q_accel"], q["q_gyro"], q_bias=qb, r_adapt_fn=make_classical_fn(),
                 x0=sens.get("x0"), P0=sens.get("P0"))
    if make_learned_filter is not None:
        fl = make_learned_filter(sens)
    else:
        fl = StepEKF(q["q_accel"], q["q_gyro"], q_bias=qb,
                     r_adapt_fn=None if learned_is_oracle else make_learned_fn(),
                     x0=sens.get("x0"), P0=sens.get("P0"))
    gps = {idx: j for j, idx in enumerate(sens["gps_idx"])}
    acc, gyr = sens["accel_meas"], sens["gyro_meas"]
    r_nom = GPS_NOISE_STD ** 2

    out = np.zeros((n, 2)); pl = np.zeros((n, 2)); pc = np.zeros((n, 2))
    use_learned = (0.0 <= lam) if stat == "pred_switch" else (lam > 0)
    open_epochs, epochs = 0, 0
    s_ema = 0.0
    for k in range(n):
        fc.predict(acc[k], gyr[k]); fl.predict(acc[k], gyr[k])
        if k in gps:
            j = gps[k]
            if sens["gps_available"][j]:
                z = np.array([sens["gps_x"][j], sens["gps_y"][j]])
                if stat in ("pred_resid", "pred_switch"):
                    d = (np.sum((z - fl.x[:2]) ** 2) - np.sum((z - fc.x[:2]) ** 2)) / r_nom
                    s_ema = beta * s_ema + (1 - beta) * d
                qj = sens["gps_quality"][j] if "gps_quality" in sens else None
                fc.update(z, quality=qj)
                if learned_is_oracle:
                    fl.update(z, quality=qj, true_std=sens["gps_std"][j])
                else:
                    fl.update(z, quality=qj)
            epochs += 1
            if stat == "pred_switch":
                use_learned = bool(s_ema <= lam)  # -inf: never; +inf: always
                if reset and not use_learned:
                    fl.reset_to(fc)
                    s_ema = 0.0
                open_epochs += int(use_learned)
                out[k] = fl.x[:2] if use_learned else fc.x[:2]
                pl[k] = fl.x[:2]; pc[k] = fc.x[:2]
                continue
            if np.isinf(lam):
                score = 0.0  # gate always open: no statistic needed
            elif stat == "sep_norm":
                score = separation(fl, fc)
            elif stat == "sep_m":
                score = float(np.linalg.norm(fl.x[:2] - fc.x[:2]))
            elif stat == "gauss_ss":
                dp = fl.x[:2] - fc.x[:2]
                Ppos = fc.P[:2, :2] + (fl.P[:2, :2] if fl.P is not None else fc.P[:2, :2])
                score = float(dp @ np.linalg.solve(Ppos, dp))
            else:
                score = s_ema
            if np.isinf(lam):
                use_learned = True
            elif score > lam:
                use_learned = False
                if reset:
                    fl.reset_to(fc)
                    s_ema = 0.0
            else:
                use_learned = lam > 0
            open_epochs += int(use_learned)
        out[k] = fl.x[:2] if use_learned else fc.x[:2]
        pl[k] = fl.x[:2]; pc[k] = fc.x[:2]
    return {"gated": out, "learned": pl, "classical": pc,
            "open_frac": open_epochs / max(epochs, 1)}
