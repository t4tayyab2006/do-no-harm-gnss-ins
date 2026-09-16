"""
simulate.py
-----------
Generates physically-realistic ground-vehicle trajectories with simulated
MEMS-grade IMU (accelerometer + gyroscope) and GPS measurements, including
GPS outage / multipath segments (the classic "urban canyon" problem).

Sensor noise parameters are set to realistic consumer-grade MEMS IMU values
(comparable to an InvenSense MPU-6050 / Bosch BMI160 class sensor) and a
consumer GPS/GNSS receiver (~2-5 m CEP), consistent with the noise regimes
used in the IMU/GPS fusion literature (see ../papers/references.md).

This plays the same role that the open-source `gnss-ins-sim` simulator
(Aceinna/gnss-ins-sim) plays in several of the reviewed papers: a controlled,
reproducible benchmark with realistic error models, used because real
recorded datasets (KITTI raw, UrbanNav, PPC-Dataset) are multi-hundred-MB to
multi-GB downloads gated behind OneDrive/Dropbox/Baidu links that are not
practically fetchable in this environment.
"""
import numpy as np

DT = 0.01          # IMU rate: 100 Hz
GPS_DT = 1.0        # GPS rate: 1 Hz
GPS_EVERY = int(round(GPS_DT / DT))

# Realistic consumer-grade MEMS IMU noise (continuous-time densities, discretized)
ACCEL_NOISE_STD = 0.03      # m/s^2  (~150 ug/sqrt(Hz)-ish integrated over 100Hz)
GYRO_NOISE_STD = np.deg2rad(0.5)   # rad/s
ACCEL_BIAS_WALK = 0.0008    # m/s^2 per sqrt(s), random-walk bias instability
GYRO_BIAS_WALK = np.deg2rad(0.02)  # rad/s per sqrt(s)

GPS_NOISE_STD = 2.5          # meters, 1-sigma (consumer GNSS, open sky)
GPS_MULTIPATH_STD = 12.0     # meters, 1-sigma during degraded/urban-canyon segments


def generate_reference_trajectory(duration_s, seed):
    """Generate a smooth, randomised ground-vehicle path (bicycle-model-like):
    varying speed + varying yaw rate 'maneuvers' (accel/decel, turns, cruise)."""
    rng = np.random.default_rng(seed)
    n = int(duration_s / DT)
    t = np.arange(n) * DT

    # Build true yaw-rate and forward-accel "control" signals as a sum of
    # smooth random maneuvers, so every trajectory is different but plausible
    # for a road vehicle.
    n_segments = rng.integers(6, 12)
    seg_bounds = np.sort(rng.choice(np.arange(1, n - 1), size=n_segments - 1, replace=False))
    seg_bounds = np.concatenate([[0], seg_bounds, [n]])

    true_accel = np.zeros(n)   # forward specific force (m/s^2)
    true_yawrate = np.zeros(n)  # rad/s

    for i in range(len(seg_bounds) - 1):
        a, b = seg_bounds[i], seg_bounds[i + 1]
        true_accel[a:b] = rng.uniform(-1.2, 1.5)
        true_yawrate[a:b] = np.deg2rad(rng.uniform(-25, 25))

    # smooth with a short moving-average so IMU signals are continuous, not blocky
    k = 25
    kernel = np.ones(k) / k
    true_accel = np.convolve(true_accel, kernel, mode="same")
    true_yawrate = np.convolve(true_yawrate, kernel, mode="same")

    # integrate kinematics: state = [x, y, theta, v]
    x = np.zeros(n); y = np.zeros(n); theta = np.zeros(n); v = np.zeros(n)
    v[0] = 3.0  # start moving at 3 m/s
    for i in range(1, n):
        v[i] = max(0.0, v[i - 1] + true_accel[i - 1] * DT)
        theta[i] = theta[i - 1] + true_yawrate[i - 1] * DT
        x[i] = x[i - 1] + v[i] * np.cos(theta[i]) * DT
        y[i] = y[i - 1] + v[i] * np.sin(theta[i]) * DT

    return {
        "t": t, "x": x, "y": y, "theta": theta, "v": v,
        "true_accel": true_accel, "true_yawrate": true_yawrate,
    }


def add_sensors(ref, seed, n_outages=3, outage_len_s=(4, 12)):
    """Corrupt the reference trajectory with realistic IMU + GPS sensor models,
    and inject GPS outage / heavy-multipath windows."""
    rng = np.random.default_rng(seed + 1000)
    n = len(ref["t"])

    # --- IMU: additive white noise + slowly-varying random-walk bias ---
    accel_bias = np.cumsum(rng.normal(0, ACCEL_BIAS_WALK * np.sqrt(DT), n))
    gyro_bias = np.cumsum(rng.normal(0, GYRO_BIAS_WALK * np.sqrt(DT), n))
    accel_meas = ref["true_accel"] + accel_bias + rng.normal(0, ACCEL_NOISE_STD, n)
    gyro_meas = ref["true_yawrate"] + gyro_bias + rng.normal(0, GYRO_NOISE_STD, n)

    # --- GPS: sparse, noisy position fixes, with outage/multipath windows ---
    gps_idx = np.arange(0, n, GPS_EVERY)
    gps_available = np.ones(len(gps_idx), dtype=bool)
    gps_std = np.full(len(gps_idx), GPS_NOISE_STD)

    # mark random outage windows (both full dropout and heavy multipath)
    outage_mask_full = np.zeros(n, dtype=bool)   # for reporting
    for _ in range(n_outages):
        start = rng.integers(int(0.1 * n), int(0.85 * n))
        length = int(rng.uniform(*outage_len_s) / DT)
        end = min(n, start + length)
        outage_mask_full[start:end] = True
        kind = rng.choice(["dropout", "multipath"])
        sel = (gps_idx >= start) & (gps_idx < end)
        if kind == "dropout":
            gps_available[sel] = False
        else:
            gps_std[sel] = GPS_MULTIPATH_STD

    gps_x = ref["x"][gps_idx] + rng.normal(0, 1.0, len(gps_idx)) * gps_std
    gps_y = ref["y"][gps_idx] + rng.normal(0, 1.0, len(gps_idx)) * gps_std

    return {
        "accel_meas": accel_meas, "gyro_meas": gyro_meas,
        "gps_idx": gps_idx, "gps_x": gps_x, "gps_y": gps_y,
        "gps_available": gps_available, "gps_std": gps_std,
        "outage_mask_full": outage_mask_full,
    }


def add_quality_indicator(sens, seed, kappa):
    """Per-fix receiver quality indicator (stand-in for HDOP / C/N0 / satellite
    count, which every real receiver reports -- even plain NMEA carries HDOP).

    Standardized signal-detection model:  q = kappa * degraded + N(0, 1)
    kappa is the detectability d': 0 = uninformative, 1 = weak, 2 = moderate,
    3 = strong. The Gaussian overlap produces realistic false alarms (indicator
    looks bad on a clean fix) and misses (multipath with a good-looking
    indicator). Drawn from its OWN random stream, so every other sensor value is
    bit-identical to the kappa-free dataset.
    """
    rng = np.random.default_rng(seed + 7000)
    degraded = sens["outage_mask_full"][sens["gps_idx"]].astype(float)
    sens["gps_quality"] = kappa * degraded + rng.normal(0.0, 1.0, len(degraded))
    return sens


def make_dataset(seed, duration_s=90.0, n_outages=3, kappa=None):
    ref = generate_reference_trajectory(duration_s, seed)
    sens = add_sensors(ref, seed, n_outages=n_outages)
    if kappa is not None:
        sens = add_quality_indicator(sens, seed, kappa)
    return ref, sens


if __name__ == "__main__":
    ref, sens = make_dataset(seed=0)
    print(f"Generated {len(ref['t'])} IMU samples, {len(sens['gps_idx'])} GPS fixes, "
          f"{(~sens['gps_available']).sum()} GPS fixes dropped, "
          f"{sens['outage_mask_full'].sum() * DT:.1f}s under degraded GPS.")
