"""
realdata.py
-----------
Semi-real evaluation data from the PPC-Dataset (Suzuki, 2025; MIT licence;
github.com/taroz/PPC-Dataset): REAL 100 Hz IMU (Analog Devices ADIS16505-2) and REAL
vehicle motion in urban Nagoya and Tokyo, with Applanix POS LV ground truth.

GNSS degradations are INJECTED onto the true trajectory -- a standard methodology
for GNSS-outage studies on real IMU data -- because the dataset's GNSS is raw
RINEX that would need a separate positioning engine. So: real inertial sensing,
real vehicle dynamics, real road geometry; synthetic GNSS errors with the same
error model as the simulator (nominal sigma 2.5 m, multipath sigma 12 m or dropout).

Conventions verified empirically against ground truth (tokyo/run1):
  * yaw rate in our ENU frame (theta counter-clockwise from east) = +gyro_z
    (correlation 1.000; the README's "Z down" would suggest the opposite sign)
  * forward acceleration = acc_x (correlation 0.968 with d|v|/dt); a mean offset of
    ~0.45 m/s^2 is gravity leaking through ~3 deg road pitch, absorbed by the
    filter's accelerometer-bias state
Lever arms between IMU, antenna and reference point (<1.5 m) are ignored; they are
small relative to the injected 2.5 m / 12 m GNSS noise.

Each run is cut into non-overlapping SEG_S-second segments; each segment is one
"trajectory" for the certificate, with exactly one degradation window. Every filter
starts from the same perturbed ground-truth state (initial alignment assumed, as is
usual in outage evaluations).
"""
import os
import numpy as np
from simulate import DT, GPS_EVERY, GPS_NOISE_STD, GPS_MULTIPATH_STD, add_quality_indicator

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "PPC-Dataset")
SEG_S = 30.0
R_EARTH = 6378137.0


def load_run(city, run):
    d = os.path.join(ROOT, city, f"run{run}")
    imu = np.loadtxt(os.path.join(d, "imu.csv"), delimiter=",", skiprows=1)
    ref = np.loadtxt(os.path.join(d, "reference.csv"), delimiter=",", skiprows=1)
    t = imu[:, 0]
    keep = (t >= ref[0, 0]) & (t <= ref[-1, 0])
    imu, t = imu[keep], t[keep]
    lat0, lon0 = np.deg2rad(ref[0, 2]), np.deg2rad(ref[0, 3])
    east = (np.deg2rad(ref[:, 3]) - lon0) * R_EARTH * np.cos(lat0)
    north = (np.deg2rad(ref[:, 2]) - lat0) * R_EARTH
    theta = np.unwrap(np.deg2rad(90.0 - ref[:, 10]))
    speed = np.hypot(ref[:, 11], ref[:, 12])
    interp = lambda a: np.interp(t, ref[:, 0], a)
    return {"t": t, "acc": imu[:, 2], "gyr": np.deg2rad(imu[:, 7]),
            "x": interp(east), "y": interp(north), "theta": interp(theta), "v": interp(speed),
            "city": city, "run": run}


def segments(run_data, seed_base, kappa=None, seg_s=SEG_S):
    """Cut a run into segments and inject one GNSS degradation window per segment.
    Returns a list of (ref, sens) pairs in the same format as simulate.make_dataset."""
    n_seg = int(round(seg_s / DT))
    total = len(run_data["t"]) // n_seg
    out = []
    for s in range(total):
        a, b = s * n_seg, (s + 1) * n_seg
        seed = seed_base + s
        rng = np.random.default_rng(seed)
        x = run_data["x"][a:b] - run_data["x"][a]
        y = run_data["y"][a:b] - run_data["y"][a]
        ref = {"t": np.arange(n_seg) * DT, "x": x, "y": y,
               "theta": run_data["theta"][a:b], "v": run_data["v"][a:b]}

        gps_idx = np.arange(0, n_seg, GPS_EVERY)
        avail = np.ones(len(gps_idx), bool)
        std = np.full(len(gps_idx), GPS_NOISE_STD)
        mask = np.zeros(n_seg, bool)
        length = int(rng.uniform(4.0, 12.0) / DT)
        start = int(rng.integers(int(3.0 / DT), n_seg - length - int(2.0 / DT)))
        mask[start:start + length] = True
        sel = (gps_idx >= start) & (gps_idx < start + length)
        if rng.random() < 0.5:
            avail[sel] = False
        else:
            std[sel] = GPS_MULTIPATH_STD
        gx = x[gps_idx] + rng.normal(0, 1.0, len(gps_idx)) * std
        gy = y[gps_idx] + rng.normal(0, 1.0, len(gps_idx)) * std

        # identical perturbed initial state for every method
        x0 = np.array([rng.normal(0, GPS_NOISE_STD), rng.normal(0, GPS_NOISE_STD),
                       ref["theta"][0] + rng.normal(0, np.deg2rad(5.0)),
                       max(0.0, ref["v"][0] + rng.normal(0, 0.5)), 0.0, 0.0])
        P0 = np.diag([GPS_NOISE_STD ** 2, GPS_NOISE_STD ** 2, np.deg2rad(5.0) ** 2, 0.5 ** 2,
                      0.5 ** 2, np.deg2rad(0.5) ** 2])
        sens = {"accel_meas": run_data["acc"][a:b], "gyro_meas": run_data["gyr"][a:b],
                "gps_idx": gps_idx, "gps_x": gx, "gps_y": gy, "gps_available": avail,
                "gps_std": std, "outage_mask_full": mask, "x0": x0, "P0": P0,
                "tag": f"{run_data['city']}{run_data['run']}-{s}"}
        if kappa is not None:
            sens = add_quality_indicator(sens, seed, kappa)
        out.append((ref, sens))
    return out


def city_segments(city, kappa=None):
    """All segments of all three runs of a city. Seeds are disjoint per run."""
    base = {"nagoya": 50_000, "tokyo": 90_000}[city]
    segs = []
    for run in (1, 2, 3):
        segs += segments(load_run(city, run), base + 1000 * run, kappa=kappa)
    return segs


if __name__ == "__main__":
    for c in ("nagoya", "tokyo"):
        s = city_segments(c)
        print(f"{c}: {len(s)} segments of {SEG_S:.0f} s; "
              f"{sum((~x[1]['gps_available']).sum() > 0 for x in s)} with a dropout, "
              f"{sum((x[1]['gps_std'] > GPS_NOISE_STD).sum() > 0 for x in s)} with multipath")
