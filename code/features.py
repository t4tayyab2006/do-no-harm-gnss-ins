"""
features.py
-----------
Causal (no future-leakage) feature engineering shared by both learned models.
"""
import numpy as np
from simulate import DT, GPS_EVERY

# The recurrent models (GRU/LSTM) are trained on a DECIMATED sequence rather
# than the raw 100 Hz IMU rate: full backprop-through-time over a 9000-step
# sequence is prohibitively slow/memory-heavy on CPU, and GPS itself only
# arrives at 1 Hz, so nothing decision-relevant is lost by summarizing IMU
# behaviour (rolling stats) at 10 Hz before feeding the recurrent nets.
DECIMATE = 10


def decimate_idx(n):
    return np.arange(0, n, DECIMATE)


def rolling_std_causal(x, w):
    """Vectorized causal rolling std (uses only current & past samples)."""
    pad = np.concatenate([np.full(w - 1, x[0]), x])
    kernel = np.ones(w) / w
    mean = np.convolve(pad, kernel, mode="valid")
    mean_sq = np.convolve(pad ** 2, kernel, mode="valid")
    var = np.maximum(mean_sq - mean ** 2, 0)
    return np.sqrt(var)


def forward_fill_nonzero(x):
    out = np.zeros_like(x)
    last = 0.0
    for k in range(len(x)):
        if x[k] != 0:
            last = x[k]
        out[k] = last
    return out


def time_since_last_fix(n, gps_idx, gps_available):
    t_since = np.zeros(n)
    last_fix_step = -10_000
    gps_map = {idx: avail for idx, avail in zip(gps_idx, gps_available)}
    for k in range(n):
        if k in gps_map and gps_map[k]:
            last_fix_step = k
        t_since[k] = (k - last_fix_step) * DT
    return t_since


def build_ankf_features(ref, sens, ekf_out):
    """6-dim causal feature vector per timestep for the ANKF corrector GRU."""
    n = len(ref["t"])
    t_since = time_since_last_fix(n, sens["gps_idx"], sens["gps_available"])
    accel_std = rolling_std_causal(sens["accel_meas"], w=100)
    gyro_std = rolling_std_causal(sens["gyro_meas"], w=100)
    log_trace_p = np.log1p(ekf_out["trace_P"])
    last_innov = forward_fill_nonzero(ekf_out["innovation_mag"])
    v = ekf_out["v"]

    feats = np.stack([
        np.clip(t_since / 5.0, 0, 3.0),
        accel_std * 10.0,
        gyro_std * 50.0,
        log_trace_p / 10.0,
        last_innov / 10.0,
        v / 10.0,
    ], axis=1)
    target = np.stack([ref["x"] - ekf_out["x"], ref["y"] - ekf_out["y"]], axis=1)

    idx = decimate_idx(n)
    return feats[idx].astype(np.float32), target[idx].astype(np.float32)


def build_lstm_inputs(ref, sens):
    """5-dim raw-sensor input sequence (no physics model) for the pure end-to-end
    LSTM baseline: accel, gyro, held-last-GPS-x, held-last-GPS-y, gps-age."""
    n = len(ref["t"])
    gps_idx, gps_x, gps_y = sens["gps_idx"], sens["gps_x"], sens["gps_y"]
    avail = sens["gps_available"]

    held_x = np.zeros(n); held_y = np.zeros(n)
    last_x, last_y = 0.0, 0.0
    gps_map = {idx: j for j, idx in enumerate(gps_idx)}
    for k in range(n):
        if k in gps_map and avail[gps_map[k]]:
            last_x, last_y = gps_x[gps_map[k]], gps_y[gps_map[k]]
        held_x[k], held_y[k] = last_x, last_y

    t_since = time_since_last_fix(n, gps_idx, avail)

    feats = np.stack([
        sens["accel_meas"],
        sens["gyro_meas"] * 10.0,
        held_x / 100.0,
        held_y / 100.0,
        np.clip(t_since / 5.0, 0, 3.0),
    ], axis=1)
    target = np.stack([ref["x"] / 100.0, ref["y"] / 100.0], axis=1)

    idx = decimate_idx(n)
    return feats[idx].astype(np.float32), target[idx].astype(np.float32)


def decimate_array(x):
    return x[decimate_idx(len(x))]
