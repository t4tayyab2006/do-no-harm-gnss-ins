"""
ekf.py
------
Classical loosely-coupled IMU/GPS Extended Kalman Filter (baseline).

State: [x, y, theta, v, accel_bias, gyro_bias]  (6-dim)
Process model: bicycle/unicycle kinematics driven by (bias-corrected) IMU.
Measurement: GPS position (x, y), when available.

Fixed process/measurement noise covariances (Q, R) — the classical approach,
and the thing the proposed hybrid improves on by *adapting* to conditions.
"""
import numpy as np
from simulate import DT, GPS_EVERY, GPS_NOISE_STD

STATE_DIM = 6  # x, y, theta, v, ab, gb


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def run_ekf(sens, n, q_accel=0.05, q_gyro=0.02, q_bias=1e-5, r_gps_base=GPS_NOISE_STD,
            r_adapt_fn=None):
    """NOTE on r_gps_base: defaults to the FIXED, nominal GPS noise std -- a
    real-world classical filter does not know in advance when a given GPS
    fix is degraded by multipath, so it cannot inflate R for those fixes.
    This is what makes the fixed-covariance filter a realistic baseline: it
    over-trusts corrupted fixes during multipath windows exactly like a real
    deployed EKF would. Pass r_gps_base=None to instead use the (unrealistic,
    oracle) true per-fix noise std from the simulator, for comparison."""
    """Runs the fixed-covariance EKF over the full IMU stream, fusing GPS
    whenever available. Returns per-timestep state estimates and covariance
    trace (a simple confidence signal used later as a feature)."""
    x = np.zeros(STATE_DIM)  # start at origin, zero heading/vel/bias
    P = np.diag([1.0, 1.0, 0.1, 1.0, 0.01, 0.01])

    Q = np.diag([0, 0, 0, q_accel * DT, q_bias, q_bias])  # process noise injected on v/biases
    accel_meas, gyro_meas = sens["accel_meas"], sens["gyro_meas"]
    gps_idx_set = {idx: k for k, idx in enumerate(sens["gps_idx"])}

    xs = np.zeros((n, STATE_DIM))
    trace_P = np.zeros(n)
    innovation_mag = np.zeros(n)  # 0 unless a GPS update happens at this step
    fix_records = []  # per-GPS-fix causal statistics (for the adaptive detector)
    nis_history = []  # normalized innovation squared, most recent last

    H = np.zeros((2, STATE_DIM)); H[0, 0] = 1; H[1, 1] = 1  # GPS observes x,y directly

    for k in range(n):
        # ---- predict ----
        theta, v, ab, gb = x[2], x[3], x[4], x[5]
        a_corr = accel_meas[k] - ab
        w_corr = gyro_meas[k] - gb

        x_pred = x.copy()
        x_pred[2] = wrap_angle(theta + w_corr * DT)
        x_pred[3] = v + a_corr * DT
        x_pred[0] = x[0] + x_pred[3] * np.cos(x_pred[2]) * DT
        x_pred[1] = x[1] + x_pred[3] * np.sin(x_pred[2]) * DT
        # biases: random walk (unchanged in mean)

        F = np.eye(STATE_DIM)
        F[2, 5] = -DT
        F[3, 4] = -DT
        F[0, 2] = -x_pred[3] * np.sin(x_pred[2]) * DT
        F[0, 3] = np.cos(x_pred[2]) * DT
        F[1, 2] = x_pred[3] * np.cos(x_pred[2]) * DT
        F[1, 3] = np.sin(x_pred[2]) * DT

        Qk = Q.copy()
        Qk[2, 2] = q_gyro * DT
        P = F @ P @ F.T + Qk
        x = x_pred

        # ---- update (if GPS available at this step) ----
        if k in gps_idx_set:
            j = gps_idx_set[k]
            if sens["gps_available"][j]:
                z = np.array([sens["gps_x"][j], sens["gps_y"][j]])
                std = sens["gps_std"][j] if r_gps_base is None else r_gps_base
                R = np.diag([std ** 2, std ** 2])
                y = z - H @ x
                S = H @ P @ H.T + R

                # --- causal statistics available BEFORE deciding how much to
                # trust this fix. NIS (normalized innovation squared) is the
                # classical consistency statistic: under a correct noise model
                # E[NIS] = dim(z) = 2, so NIS >> 2 signals the fix is far
                # noisier than R claims (multipath), which is detectable even
                # though the individual error is not predictable.
                nis = float(y @ np.linalg.inv(S) @ y)
                hist = (nis_history + [0.0, 0.0, 0.0])[-3:] if len(nis_history) < 3 else nis_history[-3:]
                feat = np.array([
                    np.log1p(nis),
                    np.log1p(np.mean(hist)),
                    np.log1p(max(hist)),
                    np.linalg.norm(y) / 10.0,
                    np.log1p(np.trace(P)),
                ])
                fix_records.append({"k": k, "j": j, "feat": feat, "nis": nis})
                nis_history.append(nis)

                # --- adaptive measurement-noise inflation (the learned part) ---
                if r_adapt_fn is not None:
                    scale = float(r_adapt_fn(feat))
                    R = R * (scale ** 2)
                    S = H @ P @ H.T + R

                K = P @ H.T @ np.linalg.inv(S)
                x = x + K @ y
                P = (np.eye(STATE_DIM) - K @ H) @ P
                innovation_mag[k] = np.linalg.norm(y)

        xs[k] = x
        trace_P[k] = np.trace(P)

    return {"x": xs[:, 0], "y": xs[:, 1], "theta": xs[:, 2], "v": xs[:, 3],
            "trace_P": trace_P, "innovation_mag": innovation_mag,
            "fix_records": fix_records}
